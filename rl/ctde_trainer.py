# rl/ctde_trainer.py
"""
CTDE trainer for the decentralized GNN.

At each step:
  - build the LOCAL graph (actor input) and GLOBAL graph (critic input),
  - actor samples actions from the local graph,
  - critic estimates value from the global graph,
  - store both graphs + RL quantities.
GAE uses centralized-critic values; the update trains the local actor and the
global critic jointly. At execution only the local actor is used.
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import List
import numpy as np
import torch

from environment.make_env import make_env
from rl.graph_batch import build_graph_from_global_obs, collate_graphs
from rl.local_graph import build_local_graph
from rl.action_masking import node_action_mask
from rl.ppo import PPOConfig
from rl.ctde_buffer import CTDERolloutBuffer
from rl.ctde_ppo import ctde_ppo_update
from controllers.decentralized_gnn import DecentralizedActorCritic
from evaluation.metrics import save_training_log, plot_training_curves
from rl.vec_env import make_vec_env, build_task_samplers


@dataclass
class CTDETrainConfig:
    n_modules_choices: List[int] = field(default_factory=lambda: [4, 6])
    mission_choices: List[str] = field(default_factory=lambda: ["power"])
    shape: str = "line"
    max_steps: int = 40
    sun_direction: tuple = (0.0, 0.0, 1.0)
    earth_direction: tuple = (0.0, 1.0, 0.0)
    reward_mode: str = "improvement"

    total_steps: int = 300_000
    rollout_len: int = 2048
    n_envs: int = 32
    n_workers: int = 8
    use_subproc: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    hidden_dim: int = 128
    actor_layers: int = 2      # SMALL => local reasoning (decentralized)
    critic_layers: int = 4     # LARGE => global value estimate
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    log_every: int = 1
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "gnn_decentralized.pt"


class CTDETrainer:
    def __init__(self, cfg: CTDETrainConfig, ppo_cfg: PPOConfig = None):
        self.cfg = cfg
        self.ppo_cfg = ppo_cfg or PPOConfig()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.ac = DecentralizedActorCritic(
            hidden_dim=cfg.hidden_dim,
            actor_layers=cfg.actor_layers,
            critic_layers=cfg.critic_layers,
            device=cfg.device,
        )
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr)
        self._episode_seed = cfg.seed

    def _new_env(self):
        n = int(self.rng.choice(self.cfg.n_modules_choices))
        mission = str(self.rng.choice(self.cfg.mission_choices))
        self._episode_seed += 1
        return make_env(
            n_modules=n, shape=self.cfg.shape, mission=mission,
            sun_direction=self.cfg.sun_direction, max_steps=self.cfg.max_steps,
            reward_mode=self.cfg.reward_mode, seed=self._episode_seed,
        )

    def _step(self, env):
        """One env step: actor acts from LOCAL graph, critic values GLOBAL graph."""
        cfg = self.cfg
        mission = env.mission.config.primary_objective

        # LOCAL graph (actor input) — from local observations only.
        all_local = env.all_local_observations()
        lgraph = build_local_graph(all_local, mission, env.mission.config.sun_direction, env.mission.config.earth_direction)
        lnf, lei, lef, lbatch, _ = collate_graphs([lgraph])
        mask = node_action_mask(env, lgraph.id_order)

        # GLOBAL graph (critic input) — from the global observation.
        gobs = env.global_observation()
        ggraph = build_graph_from_global_obs(gobs)
        gnf, gei, gef, gbatch, _ = collate_graphs([ggraph])

        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=cfg.device)
        with torch.no_grad():
            actions, log_prob = self.ac.act(
                t(lnf, torch.float32), t(lei, torch.int64), t(lef, torch.float32),
                t(lbatch, torch.int64), num_graphs=1,
                mask=t(mask, torch.float32), deterministic=False)
            value = self.ac.value(
                t(gnf, torch.float32), t(gei, torch.int64), t(gef, torch.float32),
                t(gbatch, torch.int64), num_graphs=1)

        actions_np = actions.cpu().numpy()
        joint = {mid: int(actions_np[i]) for i, mid in enumerate(lgraph.id_order)}
        next_obs, reward, term, trunc, info = env.step(joint)
        done = float(term or trunc)

        transition = dict(
            local_graph=lgraph, global_graph=ggraph, mask=mask,
            action=actions_np, log_prob=float(log_prob.item()),
            reward=reward, value=float(value.item()), done=done,
        )
        return transition, done, info

    def _bootstrap_value(self, env):
        """Centralized-critic value of the current state (for GAE tail)."""
        cfg = self.cfg
        gobs = env.global_observation()
        ggraph = build_graph_from_global_obs(gobs)
        gnf, gei, gef, gbatch, _ = collate_graphs([ggraph])
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=cfg.device)
        with torch.no_grad():
            v = self.ac.value(t(gnf, torch.float32), t(gei, torch.int64),
                              t(gef, torch.float32), t(gbatch, torch.int64),
                              num_graphs=1)
        return float(v.item())

    def train(self):
        cfg = self.cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        buffer = CTDERolloutBuffer()

        # --- Vector env ---
        env_fns = build_task_samplers(cfg, cfg.n_envs)
        vec = make_vec_env(env_fns, n_workers=cfg.n_workers,
                           use_subproc=cfg.use_subproc)
        state = vec.reset()

        global_step = 0
        ep_return = np.zeros(cfg.n_envs)
        ep_returns, ep_objectives = [], []
        update_idx = 0
        t_start = time.perf_counter()
        steps_per_rollout = max(1, cfg.rollout_len // cfg.n_envs)
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=cfg.device)

        history = {"step": [], "ep_return": [], "ep_objective": [],
                   "policy_loss": [], "value_loss": [], "entropy": [],
                   "kl": [], "clip_frac": []}

        while global_step < cfg.total_steps:
            buffer.reset()

            for _ in range(steps_per_rollout):
                # LOCAL graphs (actor) and GLOBAL graphs (critic) from state.
                local_graphs, global_graphs = [], []
                for s in state:
                    gobs, lobs = s[0], s[1]
                    mission = gobs["mission"]
                    local_graphs.append(build_local_graph(
                        lobs, mission,
                        tuple(gobs["sun_direction"]),
                        tuple(gobs["earth_direction"]),
                    ))
                    global_graphs.append(build_graph_from_global_obs(gobs))

                masks = [s[2] for s in state]
                id_orders = [s[3] for s in state]

                lnf, lei, lef, lbatch, _ = collate_graphs(local_graphs)
                gnf, gei, gef, gbatch, _ = collate_graphs(global_graphs)
                mask_cat = np.concatenate(masks, axis=0)

                with torch.no_grad():
                    actions, log_prob = self.ac.act(
                        t(lnf, torch.float32), t(lei, torch.int64),
                        t(lef, torch.float32), t(lbatch, torch.int64),
                        num_graphs=cfg.n_envs,
                        mask=t(mask_cat, torch.float32), deterministic=False)
                    value = self.ac.value(
                        t(gnf, torch.float32), t(gei, torch.int64),
                        t(gef, torch.float32), t(gbatch, torch.int64),
                        num_graphs=cfg.n_envs)

                actions_np = actions.cpu().numpy()
                logp_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy()

                joint_actions, slices = [], []
                off = 0
                for i, lg in enumerate(local_graphs):
                    n_i = lg.node_features.shape[0]
                    a_i = actions_np[off: off + n_i]
                    joint_actions.append(
                        {mid: int(a_i[r]) for r, mid in enumerate(id_orders[i])})
                    slices.append((off, n_i))
                    off += n_i

                results = vec.step(joint_actions)

                for i in range(cfg.n_envs):
                    o, n_i = slices[i]
                    reward = results[i][4]
                    done = results[i][5]
                    objective = results[i][6]
                    buffer.add(
                        local_graph=local_graphs[i],
                        global_graph=global_graphs[i],
                        mask=masks[i], action=actions_np[o: o + n_i],
                        log_prob=float(logp_np[i]), reward=reward,
                        value=float(value_np[i]), done=float(done))
                    ep_return[i] += reward
                    global_step += 1
                    if done:
                        ep_returns.append(float(ep_return[i]))
                        ep_objectives.append(objective)
                        ep_return[i] = 0.0

                state = results

            # --- Bootstrap (batched critic forward) ---
            global_graphs = [build_graph_from_global_obs(s[0]) for s in state]
            gnf, gei, gef, gbatch, _ = collate_graphs(global_graphs)
            with torch.no_grad():
                last_values = self.ac.value(
                    t(gnf, torch.float32), t(gei, torch.int64),
                    t(gef, torch.float32), t(gbatch, torch.int64),
                    num_graphs=cfg.n_envs)
            buffer.compute_gae(float(last_values.mean().item()),
                               cfg.gamma, cfg.gae_lambda)

            stats = ctde_ppo_update(self.ac, self.optimizer, buffer,
                                    self.ppo_cfg, device=cfg.device)
            update_idx += 1

            if update_idx % cfg.log_every == 0:
                recent_ret = np.mean(ep_returns[-10:]) if ep_returns else float("nan")
                recent_obj = np.mean(ep_objectives[-10:]) if ep_objectives else float("nan")
                sps = global_step / max(time.perf_counter() - t_start, 1e-9)

                history["step"].append(global_step)
                history["ep_return"].append(float(recent_ret))
                history["ep_objective"].append(float(recent_obj))
                history["policy_loss"].append(stats["policy_loss"])
                history["value_loss"].append(stats["value_loss"])
                history["entropy"].append(stats["entropy"])
                history["kl"].append(stats["kl"])
                history["clip_frac"].append(stats["clip_frac"])
                
                print(f"[upd {update_idx:4d} | step {global_step:>8d}] "
                      f"ep_ret={recent_ret:7.3f} ep_obj={recent_obj:7.3f} "
                      f"ploss={stats['policy_loss']:+.4f} "
                      f"vloss={stats['value_loss']:.4f} "
                      f"ent={stats['entropy']:.3f} kl={stats['kl']:.4f} "
                      f"| {sps:.0f} steps/s")

        ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        self.ac.save(ckpt_path, cfg.__dict__)
        print(f"\nTraining complete. Checkpoint saved to {ckpt_path}")
        base = os.path.splitext(ckpt_path)[0]
        save_training_log(base + "_log.csv", history)
        plot_training_curves(base + "_curves.png", history)
        print(f"Training log + curves saved to {base}_log.csv / _curves.png")
        return ckpt_path

    def _make_env_fn(self):
        """Return a picklable-ish factory that samples a random (N, mission).

        NOTE: uses a fresh RNG per call seeded from a counter so subprocess
        workers produce varied, reproducible tasks.
        """
        cfg = self.cfg
        base_seed = cfg.seed
        counter = {"i": 0}
        choices_n = list(cfg.n_modules_choices)
        choices_m = list(cfg.mission_choices)

        def fn():
            counter["i"] += 1
            rng = np.random.default_rng(base_seed * 100000 + counter["i"])
            n = int(rng.choice(choices_n))
            mission = str(rng.choice(choices_m))
            seed = base_seed * 100000 + counter["i"]
            return make_env(
                n_modules=n, shape=cfg.shape, mission=mission,
                sun_direction=cfg.sun_direction, max_steps=cfg.max_steps,
                reward_mode=cfg.reward_mode, seed=seed,
            )
        return fn