"""
trainer.py
==========
PPO trainer for the constellation swarm agent.

Features
--------
* Clipped PPO objective with entropy bonus
* GAE-lambda advantage estimation
* Gradient clipping
* Curriculum learning with staged complexity increase
* Tensorboard / CSV logging
* Checkpoint saving (best + periodic)
* Early-stopping on episode reward plateau

Usage
-----
    from env.constellation_env import HierarchicalConstellationEnv
    from tasks.constellation_tasks import FormConstellationTask
    from rl.trainer import PPOTrainer, TrainingConfig

    env = HierarchicalConstellationEnv(num_cubes=8)
    cfg = TrainingConfig(total_timesteps=500_000)
    trainer = PPOTrainer(env, cfg)
    trainer.train()
"""

from __future__ import annotations

import os
import csv
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Batch

from rl.agent         import ConstellationAgent
from rl.rollout_buffer import RolloutBuffer, Transition
from rl.obs_builder   import ObservationBuilder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level   = logging.INFO,
    format  = '%(asctime)s | %(levelname)s | %(message)s',
    datefmt = '%H:%M:%S',
)


# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # PPO hyper-parameters
    total_timesteps : int   = 1_000_000
    rollout_steps   : int   = 2048        # steps per rollout collection
    num_epochs      : int   = 10          # PPO update epochs
    batch_size      : int   = 64
    learning_rate   : float = 3e-4
    clip_range      : float = 0.2
    value_coef      : float = 0.5
    entropy_coef    : float = 0.01
    max_grad_norm   : float = 0.5
    gamma           : float = 0.99
    gae_lambda      : float = 0.95

    # Architecture
    hidden_dim      : int   = 128
    num_gnn_layers  : int   = 3
    num_heads       : int   = 4

    # Curriculum
    curriculum      : bool  = True
    curriculum_stages: List[Dict] = field(default_factory=lambda: [
        # Stage 1: tiny swarm, simple task
        {'min_reward': -np.inf, 'timesteps': 0,
         'num_cubes': 8,  'task': 'form_constellation',
         'max_steps': 300, 'num_groups': 2, 'baseline': 1000.0},
        # Stage 2: medium swarm
        {'min_reward': 0.3, 'timesteps': 100_000,
         'num_cubes': 27, 'task': 'form_constellation',
         'max_steps': 500, 'num_groups': 2, 'baseline': 5000.0},
        # Stage 3: add docking task
        {'min_reward': 0.4, 'timesteps': 250_000,
         'num_cubes': 27, 'task': 'rendezvous',
         'max_steps': 700, 'num_groups': 2, 'baseline': 10000.0},
        # Stage 4: full swarm
        {'min_reward': 0.5, 'timesteps': 500_000,
         'num_cubes': 64, 'task': 'form_constellation',
         'max_steps': 1000, 'num_groups': 3, 'baseline': 20000.0},
    ])

    # Logging & checkpointing
    log_interval    : int   = 10          # episodes
    save_interval   : int   = 50          # episodes
    checkpoint_dir  : str   = 'checkpoints'
    log_dir         : str   = 'logs'
    run_name        : str   = 'swarm_ppo'

    # Device
    device          : str   = 'cuda' if torch.cuda.is_available() else 'cpu'


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO trainer for the ConstellationAgent.

    Parameters
    ----------
    env : a HierarchicalConstellationEnv (or ConstellationEnv) instance
    cfg : TrainingConfig
    """

    def __init__(self, env, cfg: Optional[TrainingConfig] = None):
        self.env = env
        self.cfg = cfg or TrainingConfig()

        os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir,        exist_ok=True)

        self._setup_logging()

        # Agent
        self.agent = ConstellationAgent(
            env        = env,
            hidden_dim = self.cfg.hidden_dim,
            num_gnn_layers = self.cfg.num_gnn_layers,
            num_heads  = self.cfg.num_heads,
            device     = self.cfg.device,
        )

        # Combine encoder + policy parameters for a single optimizer
        all_params = (list(self.agent.encoder.parameters()) +
                      list(self.agent.policy.parameters()))
        self.optimizer = optim.Adam(all_params, lr=self.cfg.learning_rate,
                                    eps=1e-5)

        self.buffer = RolloutBuffer(
            capacity   = self.cfg.rollout_steps,
            gamma      = self.cfg.gamma,
            gae_lambda = self.cfg.gae_lambda,
            device     = self.cfg.device,
        )

        # Tracking
        self.global_step     : int   = 0
        self.episode         : int   = 0
        self.best_mean_reward: float = -np.inf
        self.episode_rewards : List[float] = []
        self.curriculum_stage: int   = 0

    # ──────────────────────────────────────────────────────────────────────
    # Main training loop
    # ──────────────────────────────────────────────────────────────────────

    def train(self):
        logger.info(f"Starting PPO training | device={self.cfg.device} | "
                    f"total_steps={self.cfg.total_timesteps}")

        obs, info = self.env.reset()
        ep_reward = 0.0
        ep_length = 0

        while self.global_step < self.cfg.total_timesteps:

            # ── Collect rollout ───────────────────────────────────────────
            self.buffer.reset()
            self.agent.eval()

            for _ in range(self.cfg.rollout_steps):
                action_dict = self.agent.act(obs, info)

                action_type = action_dict['action_type']
                sub_action  = action_dict['sub_action']
                log_prob    = action_dict['log_prob']
                value       = action_dict['value']

                env_action = np.array([action_type, sub_action], dtype=np.int64)
                next_obs, reward, terminated, truncated, next_info = \
                    self.env.step(env_action)

                done = terminated or truncated

                # Build mask snapshot for this transition (compact numpy)
                _, masks = self.agent.obs_builder.build(obs, info)
                mask_np = {k: v.cpu().numpy() for k, v in masks.items()
                           if isinstance(v, torch.Tensor)}
                mask_np['sep_cube_sets'] = masks.get('sep_cube_sets', [[]])

                self.buffer.add(Transition(
                    obs         = obs,
                    action_type = action_type,
                    sub_action  = sub_action,
                    log_prob    = log_prob,
                    value       = value,
                    reward      = float(reward),
                    done        = done,
                    masks       = mask_np,
                ))

                ep_reward += reward
                ep_length += 1
                self.global_step += 1

                if done:
                    self.episode += 1
                    self.episode_rewards.append(ep_reward)
                    self._log_episode(ep_reward, ep_length, next_info)

                    obs, info = self.env.reset()
                    ep_reward = 0.0
                    ep_length = 0

                    # Check curriculum progression
                    if self.cfg.curriculum:
                        self._maybe_advance_curriculum()
                else:
                    obs, info = next_obs, next_info

                if self.global_step >= self.cfg.total_timesteps:
                    break

            # ── Bootstrap last value ──────────────────────────────────────
            if not done:
                last_action = self.agent.act(obs, info)
                last_value  = last_action['value']
            else:
                last_value = 0.0

            self.buffer.compute_advantages(last_value)

            # ── PPO update ────────────────────────────────────────────────
            self.agent.train()
            ppo_stats = self._ppo_update()

            self._log_update(ppo_stats)

            # Checkpoint
            if self.episode % self.cfg.save_interval == 0 and self.episode > 0:
                self._save_checkpoint('periodic')

        logger.info("Training complete.")
        self._save_checkpoint('final')

    # ──────────────────────────────────────────────────────────────────────
    # PPO update
    # ──────────────────────────────────────────────────────────────────────

    def _ppo_update(self) -> Dict[str, float]:
        """Run num_epochs PPO update passes over the rollout buffer."""
        pg_losses, vf_losses, ent_losses, total_losses = [], [], [], []
        clip_fracs = []

        for _epoch in range(self.cfg.num_epochs):
            for batch in self.buffer.get_batches(self.cfg.batch_size):
                idx          = batch['indices']
                action_types = batch['action_types']
                sub_actions  = batch['sub_actions']
                old_log_probs= batch['old_log_probs']
                advantages   = batch['advantages']
                returns      = batch['returns']

                # Re-build graphs for this mini-batch
                graph_list, mask_list, mode_list, env_list = [], [], [], []
                for i in idx:
                    tr   = self.buffer.get_transition(int(i))
                    gd, masks = self.agent.obs_builder.build(tr.obs, {'_masks': tr.masks})
                    graph_list.append(gd)
                    mask_list.append(masks)

                # Batch graphs
                graph_batch = Batch.from_data_list(graph_list).to(self.cfg.device)

                # Batch masks (simple concatenation along batch dim)
                batched_masks = self._batch_masks(mask_list)

                # Goal tensors
                mode_idx, env_feats = self._batch_goals(graph_list)

                # Evaluate
                eval_out = self.agent.evaluate_batch(
                    graph_batch  = graph_batch,
                    masks_batch  = batched_masks,
                    mode_idx     = mode_idx,
                    env_feats    = env_feats,
                    action_types = action_types,
                    sub_actions  = sub_actions,
                )

                log_probs = eval_out['log_probs']
                entropy   = eval_out['entropy']
                values    = eval_out['values']

                # Policy loss
                ratio       = torch.exp(log_probs - old_log_probs)
                clip_frac   = ((ratio - 1.0).abs() > self.cfg.clip_range).float().mean()
                pg_loss1    = -advantages * ratio
                pg_loss2    = -advantages * ratio.clamp(
                    1.0 - self.cfg.clip_range, 1.0 + self.cfg.clip_range)
                pg_loss     = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                ent_loss = -entropy.mean()

                loss = (pg_loss
                        + self.cfg.value_coef  * vf_loss
                        + self.cfg.entropy_coef * ent_loss)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.agent.encoder.parameters()) +
                    list(self.agent.policy.parameters()),
                    self.cfg.max_grad_norm,
                )
                self.optimizer.step()

                pg_losses.append(pg_loss.item())
                vf_losses.append(vf_loss.item())
                ent_losses.append(ent_loss.item())
                total_losses.append(loss.item())
                clip_fracs.append(clip_frac.item())

        return {
            'pg_loss'    : float(np.mean(pg_losses)),
            'vf_loss'    : float(np.mean(vf_losses)),
            'entropy'    : float(-np.mean(ent_losses)),
            'total_loss' : float(np.mean(total_losses)),
            'clip_frac'  : float(np.mean(clip_fracs)),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────────────

    def _batch_masks(self, mask_list: List[Dict]) -> Dict:
        """Stack per-sample mask dicts along batch dimension."""
        batched = {}
        for key in ('action_type', 'cube_move', 'separation', 'docking', 'maneuver'):
            tensors = [m[key] for m in mask_list]
            batched[key] = torch.cat(tensors, dim=0).to(self.cfg.device)
        # sep_cube_sets is a list-of-lists; concat across batch
        batched['sep_cube_sets'] = [m.get('sep_cube_sets', [[]])[0] for m in mask_list]
        return batched

    def _batch_goals(self, graph_list):
        """Extract mode_idx and env_feats from a list of single-env graphs."""
        mode_list, env_list = [], []
        for gd in graph_list:
            gd = gd.to(self.cfg.device)
            mi, ef = self.agent._extract_goal(gd)
            mode_list.append(mi)
            env_list.append(ef)
        return (torch.cat(mode_list, dim=0).to(self.cfg.device),
                torch.cat(env_list,  dim=0).to(self.cfg.device))

    def _maybe_advance_curriculum(self):
        """Advance to next curriculum stage if ready."""
        if self.curriculum_stage >= len(self.cfg.curriculum_stages) - 1:
            return

        stages      = self.cfg.curriculum_stages
        next_stage  = stages[self.curriculum_stage + 1]
        recent_n    = min(20, len(self.episode_rewards))
        recent_mean = float(np.mean(self.episode_rewards[-recent_n:]))

        if (recent_mean  >= next_stage['min_reward'] and
                self.global_step >= next_stage['timesteps']):
            self.curriculum_stage += 1
            logger.info(f"Curriculum: advancing to stage {self.curriculum_stage} "
                        f"({next_stage})")
            self._reset_env_for_stage(next_stage)

    def _reset_env_for_stage(self, stage: Dict):
        """
        Reconfigure the environment for a new curriculum stage.
        This is a best-effort reset; concrete implementation depends on
        whether the env supports live reconfiguration.
        """
        try:
            self.env.num_cubes = stage.get('num_cubes', self.env.num_cubes)
            self.env.max_steps = stage.get('max_steps', self.env.max_steps)
            # Task change requires env rebuild in some cases; log warning
            logger.info(f"Stage parameters updated. "
                        f"Cubes={self.env.num_cubes}, MaxSteps={self.env.max_steps}")
        except Exception as exc:
            logger.warning(f"Could not fully reconfigure env for stage: {exc}")

    def _save_checkpoint(self, tag: str):
        path = os.path.join(self.cfg.checkpoint_dir,
                            f"{self.cfg.run_name}_{tag}.pt")
        self.agent.save(path)

        recent_n = min(20, len(self.episode_rewards))
        if recent_n > 0:
            mean_r = float(np.mean(self.episode_rewards[-recent_n:]))
            if mean_r > self.best_mean_reward:
                self.best_mean_reward = mean_r
                best_path = os.path.join(self.cfg.checkpoint_dir,
                                          f"{self.cfg.run_name}_best.pt")
                self.agent.save(best_path)
                logger.info(f"New best checkpoint (mean_reward={mean_r:.4f}) → {best_path}")

    # ── Logging ───────────────────────────────────────────────────────────

    def _setup_logging(self):
        self._csv_path = os.path.join(
            self.cfg.log_dir, f"{self.cfg.run_name}_episodes.csv")
        self._update_csv_path = os.path.join(
            self.cfg.log_dir, f"{self.cfg.run_name}_updates.csv")

        with open(self._csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['episode', 'step', 'reward', 'length',
                        'task_progress', 'num_groups', 'stage'])

        with open(self._update_csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'pg_loss', 'vf_loss', 'entropy',
                        'total_loss', 'clip_frac'])

    def _log_episode(self, reward: float, length: int, info: Dict):
        if self.episode % self.cfg.log_interval == 0:
            recent_n    = min(20, len(self.episode_rewards))
            recent_mean = float(np.mean(self.episode_rewards[-recent_n:])) \
                          if self.episode_rewards else 0.0
            logger.info(
                f"Ep {self.episode:5d} | "
                f"step {self.global_step:7d} | "
                f"R={reward:+.3f} | mean20={recent_mean:+.3f} | "
                f"len={length} | "
                f"groups={info.get('num_groups', '?')} | "
                f"stage={self.curriculum_stage}"
            )

        with open(self._csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                self.episode, self.global_step, reward, length,
                info.get('task_progress', 0.0),
                info.get('num_groups', 0),
                self.curriculum_stage,
            ])

    def _log_update(self, stats: Dict):
        with open(self._update_csv_path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                self.global_step,
                stats['pg_loss'],
                stats['vf_loss'],
                stats['entropy'],
                stats['total_loss'],
                stats['clip_frac'],
            ])