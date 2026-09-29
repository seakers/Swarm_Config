# rl/gnn_trainer.py
"""
GNN PPO trainer (centralized). Collects rollouts using the GNNPolicy over the
global spacecraft graph and runs PPO updates (reusing rl/ppo.py's clipped
objective via a GNN-aware update).

Because the GNN handles any N, training can randomize N and mission per episode,
producing one checkpoint conditioned on both (mission is a per-node feature; N
is handled naturally by the graph).
"""

from __future__ import annotations
import os
import time
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn

from environment.make_env import make_env
from rl.graph_batch import (
    build_graph_from_global_obs, collate_graphs,
    node_feature_dim, EDGE_FEATURE_DIM, MISSION_LIST,
)
from rl.action_masking import node_action_mask
from rl.ppo import PPOConfig
from controllers.gnn_policy import GNNPolicy
from evaluation.metrics import save_training_log, plot_training_curves
from rl.vec_env import make_vec_env, build_task_samplers


@dataclass
class GNNTrainConfig:
    # Task family (randomized per episode if lists have >1 entry).
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
    n_layers: int = 3          # MP layers: large => centralized reasoning
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    log_every: int = 1
    checkpoint_dir: str = "checkpoints"
    checkpoint_name: str = "gnn_centralized.pt"


class GNNPPOTrainer:
    def __init__(self, cfg: GNNTrainConfig, ppo_cfg: PPOConfig = None):
        self.cfg = cfg
        self.ppo_cfg = ppo_cfg or PPOConfig()
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)

        self.node_dim = node_feature_dim()
        self.edge_dim = EDGE_FEATURE_DIM
        self.policy = GNNPolicy(
            node_dim=self.node_dim, edge_dim=self.edge_dim,
            hidden=cfg.hidden_dim, n_layers=cfg.n_layers,
        ).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cfg.lr)

        self._episode_seed = cfg.seed

    def _new_env(self):
        """Sample a fresh (N, mission) task and build an env for it."""
        n = int(self.rng.choice(self.cfg.n_modules_choices))
        mission = str(self.rng.choice(self.cfg.mission_choices))
        self._episode_seed += 1
        env = make_env(
            n_modules=n, shape=self.cfg.shape, mission=mission,
            sun_direction=self.cfg.sun_direction, max_steps=self.cfg.max_steps,
            reward_mode=self.cfg.reward_mode, seed=self._episode_seed,
        )
        return env

    def _rollout_step(self, env, obs_struct):
        """Take one env step under the current policy; return transition data."""
        graph = build_graph_from_global_obs(obs_struct)
        (nf, ei, ef, batch, sizes) = collate_graphs([graph])
        mask = node_action_mask(env, graph.id_order)

        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.cfg.device)
        with torch.no_grad():
            actions, log_prob, value = self.policy.act(
                t(nf, torch.float32), t(ei, torch.int64), t(ef, torch.float32),
                t(batch, torch.int64), num_graphs=1,
                mask=t(mask, torch.float32), deterministic=False,
            )
        actions_np = actions.cpu().numpy()
        joint = {mid: int(actions_np[i]) for i, mid in enumerate(graph.id_order)}
        next_obs, reward, term, trunc, info = env.step(joint)
        done = float(term or trunc)
        transition = dict(graph=graph, mask=mask, action=actions_np,
                          log_prob=float(log_prob.item()), reward=reward,
                          value=float(value.item()), done=done)
        return transition, next_obs, done, info

    def train(self):
        from rl.gnn_buffer import GNNRolloutBuffer
        from rl.gnn_ppo import gnn_ppo_update

        cfg = self.cfg
        os.makedirs(cfg.checkpoint_dir, exist_ok=True)
        buffer = GNNRolloutBuffer()

        # --- Vector env ---
        env_fns = build_task_samplers(cfg, cfg.n_envs)
        vec = make_vec_env(env_fns, n_workers=cfg.n_workers,
                           use_subproc=cfg.use_subproc)

        # state[i] = (gobs, lobs, mask, id_order, reward, done, objective)
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
                # --- Act on the CURRENT state ---
                graphs = [build_graph_from_global_obs(s[0]) for s in state]
                masks = [s[2] for s in state]
                id_orders = [s[3] for s in state]

                nf, ei, ef, batch, _ = collate_graphs(graphs)
                mask_cat = np.concatenate(masks, axis=0)

                with torch.no_grad():
                    actions, log_prob, value = self.policy.act(
                        t(nf, torch.float32), t(ei, torch.int64),
                        t(ef, torch.float32), t(batch, torch.int64),
                        num_graphs=cfg.n_envs, mask=t(mask_cat, torch.float32))
                actions_np = actions.cpu().numpy()
                logp_np = log_prob.cpu().numpy()
                value_np = value.cpu().numpy()

                # Slice per-env actions and build joint actions.
                joint_actions, slices = [], []
                off = 0
                for i, g in enumerate(graphs):
                    n_i = g.node_features.shape[0]
                    a_i = actions_np[off: off + n_i]
                    joint_actions.append(
                        {mid: int(a_i[r]) for r, mid in enumerate(id_orders[i])})
                    slices.append((off, n_i))
                    off += n_i

                # --- Step all envs in parallel ---
                results = vec.step(joint_actions)

                # --- Store transitions (graph/mask/action we ACTED on) ---
                for i in range(cfg.n_envs):
                    o, n_i = slices[i]
                    reward = results[i][4]
                    done = results[i][5]
                    objective = results[i][6]
                    buffer.add(
                        graph=graphs[i], mask=masks[i],
                        action=actions_np[o: o + n_i],
                        log_prob=float(logp_np[i]), reward=reward,
                        value=float(value_np[i]), done=float(done))
                    ep_return[i] += reward
                    global_step += 1
                    if done:
                        ep_returns.append(float(ep_return[i]))
                        ep_objectives.append(objective)
                        ep_return[i] = 0.0

                # --- Advance: overwrite held state with the step result ---
                state = results

            # --- Bootstrap values from the CURRENT state ---
            graphs = [build_graph_from_global_obs(s[0]) for s in state]
            nf, ei, ef, batch, _ = collate_graphs(graphs)
            with torch.no_grad():
                _, _, last_values = self.policy.act(
                    t(nf, torch.float32), t(ei, torch.int64),
                    t(ef, torch.float32), t(batch, torch.int64),
                    num_graphs=cfg.n_envs, deterministic=True)
            buffer.compute_gae(float(last_values.mean().item()),
                                    cfg.gamma, cfg.gae_lambda)

            stats = gnn_ppo_update(self.policy, self.optimizer, buffer,
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

        vec.close()
        ckpt_path = os.path.join(cfg.checkpoint_dir, cfg.checkpoint_name)
        self.save(ckpt_path)
        print(f"\nTraining complete. Checkpoint saved to {ckpt_path}")
        base = os.path.splitext(ckpt_path)[0]
        save_training_log(base + "_log.csv", history)
        plot_training_curves(base + "_curves.png", history)
        print(f"Training log + curves saved to {base}_log.csv / _curves.png")
        return ckpt_path

    def save(self, path: str):
        torch.save({
            "state_dict": self.policy.state_dict(),
            "node_dim": self.node_dim,
            "edge_dim": self.edge_dim,
            "train_config": self.cfg.__dict__,
        }, path)

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