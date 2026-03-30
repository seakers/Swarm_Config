"""
rl_agent.py
===========
Deep RL agent for modular spacecraft swarm reconfiguration.

The agent learns to reconfigure a swarm of cubesats to optimise physics-
driven mission objectives (solar-panel exposure, Earth antenna data-rate,
thermal efficiency, science instrument coverage, etc.) without being told
what *shape* to form.  It generalises across swarm sizes and tasks via a
Graph-Neural-Network-style permutation-invariant observation encoder.

Architecture overview
---------------------
    Observation  ──► Per-cube MLP trunk ──► Mean-pool
                                                 │
                     Mission context ─────────────┤
                                                  ▼
                                            Shared MLP
                                         ┌──────┴──────┐
                                      Actor          Critic

Training
--------
We use Maskable PPO (sb3-contrib) so the policy never wastes logit mass
on physically invalid moves.

Quick start
-----------
    # Train a new agent
    python rl_agent.py --train --steps 1_000_000 --save-path models/ppo_swarm

    # Evaluate a saved agent
    python rl_agent.py --eval --load-path models/ppo_swarm --episodes 20

    # Render one episode interactively
    python rl_agent.py --render --load-path models/ppo_swarm

Dependencies
------------
    pip install stable-baselines3 sb3-contrib gymnasium numpy torch
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

# ── Stable-Baselines3 + MaskablePPO ────────────────────────────────────────
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CheckpointCallback,
)
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

# ── Project imports ─────────────────────────────────────────────────────────
from configs.formations import (
    create_cube_formation,
    create_plane_formation,
    create_line_formation,
)
from core.swarm import Swarm
from env.env import SwarmReconfigurationEnv
from mechanics.moves import MovementSystem, HingeMove
from rewards.metrics import SwarmMetrics, MissionModeScorer
from tasks.tasks import (
    Task,
    MissionModeTask,
    MultiObjectiveMissionTask,
)


# ===========================================================================
# 1.  Mission-mode gymnasium environment
# ===========================================================================

# Each mission scenario has a set of environmental context vectors plus a
# task definition.  We use MissionModeTask / MultiObjectiveMissionTask to
# keep the reward physically grounded.

@dataclass
class MissionScenario:
    """A fully-specified mission scenario sampled each episode."""
    name: str
    task: Task
    sun_direction: Tuple[float, float, float]
    earth_direction: Tuple[float, float, float]
    target_direction: Tuple[float, float, float]
    sun_distance_au: float


def _random_unit_vector(rng: np.random.Generator) -> Tuple[float, float, float]:
    v = rng.standard_normal(3)
    v = v / np.linalg.norm(v)
    return tuple(float(x) for x in v)


def _build_scenario_pool(
    num_cubes: int,
    rng: np.random.Generator,
) -> List[MissionScenario]:
    """
    Return a diverse pool of mission scenarios.

    The pool mixes:
      • Single-mode tasks  (communication, observation, cruise, charging,
                            thermal_emergency, distributed_sensing)
      • Multi-objective tasks  (e.g. observe + communicate simultaneously)
    Environmental vectors (sun / earth / target) are randomised so the agent
    cannot overfit to a fixed geometry.
    """
    scenarios: List[MissionScenario] = []

    # ── Single-mode tasks ───────────────────────────────────────────────────
    single_modes = [
        "communication",
        "observation",
        "cruise",
        "charging",
        "thermal_emergency",
        "distributed_sensing",
    ]

    for mode in single_modes:
        for _ in range(4):          # 4 random contexts per mode
            sun_dir   = _random_unit_vector(rng)
            earth_dir = _random_unit_vector(rng)
            tgt_dir   = _random_unit_vector(rng)
            dist_au   = float(rng.uniform(0.5, 30.0))

            scenarios.append(MissionScenario(
                name=mode,
                task=MissionModeTask(
                    mode=mode,
                    sun_direction=sun_dir,
                    earth_direction=earth_dir,
                    target_direction=tgt_dir,
                    sun_distance_au=dist_au,
                    target_score=0.75,
                ),
                sun_direction=sun_dir,
                earth_direction=earth_dir,
                target_direction=tgt_dir,
                sun_distance_au=dist_au,
            ))

    # ── Multi-objective tasks ───────────────────────────────────────────────
    multi_configs = [
        {"observation": 0.6, "communication": 0.4},
        {"communication": 0.5, "charging": 0.5},
        {"observation": 0.4, "cruise": 0.3, "communication": 0.3},
        {"charging": 0.5, "thermal_emergency": 0.5},
        {"distributed_sensing": 0.6, "communication": 0.4},
    ]

    for obj_dict in multi_configs:
        for _ in range(3):
            sun_dir   = _random_unit_vector(rng)
            earth_dir = _random_unit_vector(rng)
            tgt_dir   = _random_unit_vector(rng)
            dist_au   = float(rng.uniform(0.5, 30.0))

            name = "+".join(f"{k}({v:.1f})" for k, v in obj_dict.items())
            scenarios.append(MissionScenario(
                name=name,
                task=MultiObjectiveMissionTask(
                    objectives=obj_dict,
                    sun_direction=sun_dir,
                    earth_direction=earth_dir,
                    target_direction=tgt_dir,
                    sun_distance_au=dist_au,
                    target_score=0.70,
                ),
                sun_direction=sun_dir,
                earth_direction=earth_dir,
                target_direction=tgt_dir,
                sun_distance_au=dist_au,
            ))

    return scenarios


class MissionModeEnv(SwarmReconfigurationEnv):
    """
    Extends SwarmReconfigurationEnv for physics-driven mission objectives.

    Key additions
    -------------
    * Per-episode mission scenario sampling (task + environmental context).
    * Richer observation that includes:
        - Per-cube features (position, orientation, face exposure fractions)
        - Swarm-level metrics (compactness, baseline, surface area, ...)
        - Mission context (sun / earth / target unit vectors, distance,
          current mission-mode score breakdown)
    * Observation is structured so the per-cube portion is permutation-
      invariant after mean-pooling (used by the GNN-style extractor below).
    """

    # Per-cube feature dimension (computed in _per_cube_features)
    PER_CUBE_DIM = 15   # 3 pos + 6 orientation (upper-triangle) + 6 face-state

    # Mission context feature dimension
    CONTEXT_DIM = 16    # 3 sun + 3 earth + 3 target + 1 dist + 6 mode scores

    def __init__(
        self,
        num_cubes: int = 27,
        scenario_pool: Optional[List[MissionScenario]] = None,
        max_steps: int = 500,
        step_penalty: float = 5e-4,
        invalid_action_penalty: float = 0.05,
        require_connectivity: bool = False,
        initial_formation: str = "cube",
        render_mode: Optional[str] = None,
    ):
        # We pass a placeholder task; it is replaced on every reset().
        super().__init__(
            num_cubes=num_cubes,
            task=MissionModeTask("cruise"),
            max_steps=max_steps,
            step_penalty=step_penalty,
            invalid_action_penalty=invalid_action_penalty,
            require_connectivity=require_connectivity,
            initial_formation=initial_formation,
            render_mode=render_mode,
        )

        self._rng = np.random.default_rng(0)
        self._scenario_pool = scenario_pool or _build_scenario_pool(num_cubes, self._rng)
        self._current_scenario: Optional[MissionScenario] = None

        # ── Override observation space ──────────────────────────────────────
        obs_dim = self.PER_CUBE_DIM + self.CONTEXT_DIM
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Scenario helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _sample_scenario(self) -> MissionScenario:
        idx = self._rng.integers(len(self._scenario_pool))
        return self._scenario_pool[idx]

    # ──────────────────────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────────────────────

    def _per_cube_features(self) -> np.ndarray:
        """
        Return a (PER_CUBE_DIM,) vector = mean-pooled per-cube features.

        Each cube contributes:
          [0:3]   normalised position relative to centroid  (3)
          [3:9]   upper triangle of 3×3 orientation matrix (6)
          [9:15]  face-state flags: for each of 6 faces,
                  1 if exposed else 0                       (6)
        """
        centroid = np.array(self.swarm.get_center_of_mass(), dtype=np.float32)
        cube_feat = np.zeros((self.num_cubes, self.PER_CUBE_DIM), dtype=np.float32)

        # Build occupied set for exposure check
        occupied = {c.position for c in self.swarm.get_all_cubes()}

        for i, cube in enumerate(self.swarm.get_all_cubes()):
            if i >= self.num_cubes:
                break

            # Relative position (normalised by swarm size)
            pos = (np.array(cube.position, dtype=np.float32) - centroid)
            scale = max(1.0, self.num_cubes ** (1 / 3))
            cube_feat[i, 0:3] = pos / scale

            # Compact orientation: upper-triangle of rotation matrix
            mat = cube.orientation.matrix.flatten().astype(np.float32)
            # mat has 9 elements; take [0,1,2,4,5,8] (upper triangle + diagonal)
            cube_feat[i, 3:9] = mat[[0, 1, 2, 4, 5, 8]]

            # Face exposure: one flag per of the 6 axis-aligned faces
            #  (+x, -x, +y, -y, +z, -z)
            for fi, (dx, dy, dz) in enumerate(
                [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
            ):
                adj = (
                    cube.position[0] + dx,
                    cube.position[1] + dy,
                    cube.position[2] + dz,
                )
                cube_feat[i, 9 + fi] = 0.0 if adj in occupied else 1.0

        # Mean-pool: (num_cubes, PER_CUBE_DIM) → (PER_CUBE_DIM,)
        return cube_feat.mean(axis=0)

    def _context_features(self) -> np.ndarray:
        """
        Return a (CONTEXT_DIM,) mission-context vector:
          [0:3]   sun_direction  (unit vector)
          [3:6]   earth_direction
          [6:9]   target_direction
          [9]     log(sun_distance_au) / 4  (normalised)
          [10:16] per-mode score breakdown for current config
        """
        sc = self._current_scenario
        ctx = np.zeros(self.CONTEXT_DIM, dtype=np.float32)

        ctx[0:3]  = np.array(sc.sun_direction,   dtype=np.float32)
        ctx[3:6]  = np.array(sc.earth_direction, dtype=np.float32)
        ctx[6:9]  = np.array(sc.target_direction, dtype=np.float32)
        ctx[9]    = float(np.log1p(sc.sun_distance_au) / 4.0)

        # Quick per-mode scores (6 values)
        try:
            scorer = MissionModeScorer(self.swarm)
            all_scores = scorer.get_all_mode_scores(
                sc.sun_direction, sc.earth_direction,
                sc.target_direction, sc.sun_distance_au,
            )
            ordered = [
                "communication", "observation", "cruise",
                "charging", "thermal_emergency", "distributed_sensing",
            ]
            for k, mode in enumerate(ordered):
                ctx[10 + k] = float(all_scores.get(mode, 0.0))
        except Exception:
            pass   # Leave zeros if scorer is unavailable

        return ctx

    def _get_observation(self) -> np.ndarray:
        cube_feat = self._per_cube_features()           # (PER_CUBE_DIM,)
        ctx_feat  = self._context_features()            # (CONTEXT_DIM,)
        return np.concatenate([cube_feat, ctx_feat]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Reset / step
    # ──────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        # Sample new scenario
        self._current_scenario = self._sample_scenario()
        self.task = self._current_scenario.task

        obs, info = super().reset(seed=seed, options=options)

        info["scenario_name"] = self._current_scenario.name
        info["sun_distance_au"] = self._current_scenario.sun_distance_au
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = super().step(action)
        info["scenario_name"] = getattr(self._current_scenario, "name", "unknown")
        return obs, reward, terminated, truncated, info


# ===========================================================================
# 2.  Action masking helper (required by MaskablePPO)
# ===========================================================================

def get_action_mask(env: MissionModeEnv) -> np.ndarray:
    """
    Return the boolean action-validity mask for the current env state.
    Called by ActionMasker wrapper on every step / reset.
    """
    return env._get_action_mask()


# ===========================================================================
# 3.  GNN-style features extractor
# ===========================================================================

class SwarmFeaturesExtractor(BaseFeaturesExtractor):
    """
    Permutation-invariant feature extractor for swarm observations.

    The input observation vector is split into:
      • Per-cube mean-pooled features  (already pooled in the env)
      • Mission context features

    Both halves pass through their own MLP before being concatenated
    into a shared features vector consumed by the Actor/Critic heads.

    Because the env already mean-pools the per-cube features, this
    extractor is compatible with variable swarm sizes without any
    architecture changes.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        per_cube_dim: int = MissionModeEnv.PER_CUBE_DIM,
        context_dim:  int = MissionModeEnv.CONTEXT_DIM,
        cube_hidden:  int = 128,
        ctx_hidden:   int = 64,
        shared_hidden: int = 256,
        features_dim: int = 256,
    ):
        super().__init__(observation_space, features_dim)

        self.per_cube_dim = per_cube_dim
        self.context_dim  = context_dim

        # ── Per-cube trunk ──────────────────────────────────────────────────
        self.cube_net = nn.Sequential(
            nn.Linear(per_cube_dim, cube_hidden),
            nn.LayerNorm(cube_hidden),
            nn.GELU(),
            nn.Linear(cube_hidden, cube_hidden),
            nn.LayerNorm(cube_hidden),
            nn.GELU(),
        )

        # ── Context trunk ───────────────────────────────────────────────────
        self.ctx_net = nn.Sequential(
            nn.Linear(context_dim, ctx_hidden),
            nn.LayerNorm(ctx_hidden),
            nn.GELU(),
            nn.Linear(ctx_hidden, ctx_hidden),
            nn.LayerNorm(ctx_hidden),
            nn.GELU(),
        )

        # ── Shared head ─────────────────────────────────────────────────────
        self.shared = nn.Sequential(
            nn.Linear(cube_hidden + ctx_hidden, shared_hidden),
            nn.LayerNorm(shared_hidden),
            nn.GELU(),
            nn.Linear(shared_hidden, features_dim),
            nn.LayerNorm(features_dim),
            nn.GELU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cube_obs = observations[:, : self.per_cube_dim]
        ctx_obs  = observations[:, self.per_cube_dim :]

        cube_feat = self.cube_net(cube_obs)
        ctx_feat  = self.ctx_net(ctx_obs)

        combined = torch.cat([cube_feat, ctx_feat], dim=-1)
        return self.shared(combined)


# ===========================================================================
# 4.  Training callbacks
# ===========================================================================

class MissionProgressCallback(BaseCallback):
    """
    Logs per-episode mission progress and scenario diversity metrics to
    stdout and (optionally) to a CSV for plotting.
    """

    def __init__(
        self,
        log_freq: int = 5_000,
        csv_path: Optional[str] = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq  = log_freq
        self.csv_path  = csv_path
        self._csv_file = None
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int]   = []
        self._scenario_names:  List[str]   = []
        self._last_log_step = 0

    def _on_training_start(self) -> None:
        if self.csv_path:
            os.makedirs(os.path.dirname(self.csv_path) or ".", exist_ok=True)
            self._csv_file = open(self.csv_path, "w", buffering=1)
            self._csv_file.write(
                "step,mean_reward,mean_ep_len,num_episodes,scenarios\n"
            )

    def _on_step(self) -> bool:
        # Collect episode completions from the Monitor wrapper info
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self._episode_rewards.append(ep_info["r"])
                self._episode_lengths.append(ep_info["l"])
                self._scenario_names.append(info.get("scenario_name", "?"))

        # Periodic logging
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._last_log_step = self.num_timesteps
            n = len(self._episode_rewards)
            if n > 0:
                mean_r  = np.mean(self._episode_rewards[-100:])
                mean_l  = np.mean(self._episode_lengths[-100:])
                unique_s = len(set(self._scenario_names[-100:]))
                if self.verbose >= 1:
                    print(
                        f"[{self.num_timesteps:>9,d} steps]  "
                        f"mean_reward={mean_r:+.4f}  "
                        f"mean_ep_len={mean_l:.1f}  "
                        f"unique_scenarios={unique_s}  "
                        f"episodes={n}"
                    )
                if self._csv_file:
                    self._csv_file.write(
                        f"{self.num_timesteps},{mean_r:.6f},"
                        f"{mean_l:.2f},{n},{unique_s}\n"
                    )
        return True

    def _on_training_end(self) -> None:
        if self._csv_file:
            self._csv_file.close()


# ===========================================================================
# 5.  Agent builder
# ===========================================================================

@dataclass
class AgentConfig:
    """Hyperparameter bundle – tweak from CLI or in code."""

    # Environment
    num_cubes:          int   = 27          # 3×3×3 default (perfect cube)
    max_steps:          int   = 400
    n_envs:             int   = 8           # parallel envs for rollout
    initial_formation:  str   = "cube"

    # PPO hyperparameters
    total_timesteps:    int   = 1_000_000
    n_steps:            int   = 512         # rollout length per env
    batch_size:         int   = 256
    n_epochs:           int   = 10
    learning_rate:      float = 3e-4
    gamma:              float = 0.995
    gae_lambda:         float = 0.95
    clip_range:         float = 0.20
    ent_coef:           float = 0.01        # encourages exploration
    vf_coef:            float = 0.5
    max_grad_norm:      float = 0.5

    # Network
    cube_hidden:        int   = 128
    ctx_hidden:         int   = 64
    shared_hidden:      int   = 256
    features_dim:       int   = 256
    pi_layers:          List[int] = field(default_factory=lambda: [256, 128])
    vf_layers:          List[int] = field(default_factory=lambda: [256, 128])

    # Logging / checkpointing
    log_freq:           int   = 10_000
    checkpoint_freq:    int   = 50_000
    eval_freq:          int   = 25_000
    eval_episodes:      int   = 10
    save_path:          str   = "models/ppo_swarm_mission"
    tensorboard_log:    Optional[str] = "runs/swarm_ppo"


def make_env(cfg: AgentConfig, seed: int = 0) -> MissionModeEnv:
    """Factory that creates one wrapped environment."""
    rng  = np.random.default_rng(seed)
    pool = _build_scenario_pool(cfg.num_cubes, rng)

    env = MissionModeEnv(
        num_cubes=cfg.num_cubes,
        scenario_pool=pool,
        max_steps=cfg.max_steps,
        initial_formation=cfg.initial_formation,
    )
    env = ActionMasker(env, get_action_mask)   # plug in masking
    env = Monitor(env)
    return env


def build_agent(cfg: AgentConfig) -> MaskablePPO:
    """
    Build and return a MaskablePPO agent with the GNN-style extractor.
    """
    # ── Vectorised environments ─────────────────────────────────────────────
    vec_env = DummyVecEnv([
        (lambda s: lambda: make_env(cfg, seed=s))(seed)
        for seed in range(cfg.n_envs)
    ])
    vec_env = VecMonitor(vec_env)

    # ── Policy kwargs ────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class=SwarmFeaturesExtractor,
        features_extractor_kwargs=dict(
            per_cube_dim=MissionModeEnv.PER_CUBE_DIM,
            context_dim=MissionModeEnv.CONTEXT_DIM,
            cube_hidden=cfg.cube_hidden,
            ctx_hidden=cfg.ctx_hidden,
            shared_hidden=cfg.shared_hidden,
            features_dim=cfg.features_dim,
        ),
        net_arch=dict(
            pi=cfg.pi_layers,
            vf=cfg.vf_layers,
        ),
        activation_fn=nn.GELU,
    )

    # ── Agent ────────────────────────────────────────────────────────────────
    agent = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        max_grad_norm=cfg.max_grad_norm,
        policy_kwargs=policy_kwargs,
        tensorboard_log=cfg.tensorboard_log,
        verbose=1,
    )

    return agent


# ===========================================================================
# 6.  Training entry point
# ===========================================================================

def train(cfg: AgentConfig) -> MaskablePPO:
    """Train the agent and return the trained model."""
    print("=" * 65)
    print("  Swarm Mission RL Agent — Training")
    print("=" * 65)
    print(f"  Cubes         : {cfg.num_cubes}")
    print(f"  Total steps   : {cfg.total_timesteps:,}")
    print(f"  Parallel envs : {cfg.n_envs}")
    print(f"  PPO n_steps   : {cfg.n_steps}  batch={cfg.batch_size}")
    print(f"  Save path     : {cfg.save_path}")
    print("=" * 65)

    agent = build_agent(cfg)

    # ── Eval env (single, no Monitor doubling) ───────────────────────────────
    eval_env = DummyVecEnv([lambda: make_env(cfg, seed=999)])

    # ── Callbacks ────────────────────────────────────────────────────────────
    os.makedirs(cfg.save_path, exist_ok=True)
    csv_path = os.path.join(cfg.save_path, "training_log.csv")

    callbacks = [
        MissionProgressCallback(
            log_freq=cfg.log_freq,
            csv_path=csv_path,
            verbose=1,
        ),
        CheckpointCallback(
            save_freq=cfg.checkpoint_freq // cfg.n_envs,
            save_path=cfg.save_path,
            name_prefix="ppo_swarm",
            verbose=0,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(cfg.save_path, "best"),
            log_path=os.path.join(cfg.save_path, "eval"),
            eval_freq=cfg.eval_freq // cfg.n_envs,
            n_eval_episodes=cfg.eval_episodes,
            deterministic=True,
            verbose=1,
        ),
    ]

    t0 = time.time()
    agent.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    elapsed = time.time() - t0
    print(f"\n  Training finished in {elapsed/60:.1f} min.")

    # Save final model
    final_path = os.path.join(cfg.save_path, "final_model")
    agent.save(final_path)
    print(f"  Saved to: {final_path}.zip")

    return agent


# ===========================================================================
# 7.  Evaluation
# ===========================================================================

@dataclass
class EpisodeResult:
    scenario_name: str
    total_reward:  float
    steps:         int
    final_score:   float
    completed:     bool


def evaluate(
    agent: MaskablePPO,
    cfg: AgentConfig,
    n_episodes: int = 20,
    deterministic: bool = True,
    verbose: bool = True,
) -> List[EpisodeResult]:
    """
    Run *n_episodes* evaluation episodes and return per-episode results.
    """
    rng  = np.random.default_rng(42)
    pool = _build_scenario_pool(cfg.num_cubes, rng)

    results: List[EpisodeResult] = []

    for ep in range(n_episodes):
        env = MissionModeEnv(
            num_cubes=cfg.num_cubes,
            scenario_pool=pool,
            max_steps=cfg.max_steps,
            initial_formation=cfg.initial_formation,
        )
        masked_env = ActionMasker(env, get_action_mask)

        obs, info = masked_env.reset()
        scenario_name = info.get("scenario_name", "?")

        ep_reward = 0.0
        step      = 0
        done      = False

        while not done:
            action_masks = get_action_mask(env)
            action, _ = agent.predict(
                obs,
                action_masks=action_masks,
                deterministic=deterministic,
            )
            obs, reward, terminated, truncated, info = masked_env.step(int(action))
            ep_reward += reward
            step      += 1
            done = terminated or truncated

        final_progress = info.get("task_progress", 0.0)
        completed      = info.get("task_complete", False)

        result = EpisodeResult(
            scenario_name=scenario_name,
            total_reward=ep_reward,
            steps=step,
            final_score=final_progress,
            completed=completed,
        )
        results.append(result)

        if verbose:
            tick = "✓" if completed else " "
            print(
                f"  [{tick}] Ep {ep+1:3d}  scenario={scenario_name:<35s}  "
                f"reward={ep_reward:+.3f}  score={final_progress:.3f}  "
                f"steps={step}"
            )

        masked_env.close()

    # Summary
    if verbose:
        print("\n" + "─" * 65)
        rewards  = [r.total_reward  for r in results]
        scores   = [r.final_score   for r in results]
        complete = [r.completed     for r in results]
        print(f"  Mean reward : {np.mean(rewards):+.4f}  ± {np.std(rewards):.4f}")
        print(f"  Mean score  : {np.mean(scores):.4f}  ± {np.std(scores):.4f}")
        print(f"  Completion  : {sum(complete)}/{n_episodes}")

        # Per-scenario breakdown
        scenario_groups: Dict[str, List[float]] = {}
        for r in results:
            base = r.scenario_name.split("(")[0]
            scenario_groups.setdefault(base, []).append(r.final_score)
        print("\n  Score by scenario type:")
        for name, sc in sorted(scenario_groups.items()):
            print(f"    {name:<40s}  {np.mean(sc):.3f}")

    return results


# ===========================================================================
# 8.  Render one episode
# ===========================================================================

def render_episode(
    agent: MaskablePPO,
    cfg: AgentConfig,
    scenario_name: Optional[str] = None,
) -> None:
    """
    Render a single episode interactively using the project's SwarmVisualizer.
    """
    from visualization.renderer import SwarmVisualizer   # late import

    rng  = np.random.default_rng(0)
    pool = _build_scenario_pool(cfg.num_cubes, rng)

    if scenario_name:
        pool = [s for s in pool if scenario_name.lower() in s.name.lower()] or pool

    env = MissionModeEnv(
        num_cubes=cfg.num_cubes,
        scenario_pool=pool,
        max_steps=cfg.max_steps,
        initial_formation=cfg.initial_formation,
    )
    masked_env = ActionMasker(env, get_action_mask)

    obs, info = masked_env.reset()
    print(f"\n  Rendering scenario: {info.get('scenario_name')}")

    viz = SwarmVisualizer(env.swarm)
    viz.render(
        title=f"Step 0 | {info.get('scenario_name')} | score=0.00",
        show_connections=True,
    )

    step, ep_reward, done = 0, 0.0, False
    while not done:
        action_masks = get_action_mask(env)
        action, _ = agent.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = masked_env.step(int(action))
        ep_reward += reward
        step      += 1
        done = terminated or truncated

        if step % 5 == 0 or done:
            score = info.get("task_progress", 0.0)
            viz.swarm = env.swarm
            viz.render(
                title=(
                    f"Step {step} | {info.get('scenario_name')} | "
                    f"score={score:.3f} | reward={ep_reward:+.3f}"
                ),
                show_connections=True,
            )

    print(f"  Episode finished: steps={step}  reward={ep_reward:+.4f}")
    viz.show()
    viz.close()
    masked_env.close()


# ===========================================================================
# 9.  CLI
# ===========================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train / evaluate the swarm mission RL agent."
    )

    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train",  action="store_true", help="Train a new agent")
    mode.add_argument("--eval",   action="store_true", help="Evaluate a saved agent")
    mode.add_argument("--render", action="store_true", help="Render one episode")

    # ── Shared ───────────────────────────────────────────────────────────────
    p.add_argument("--num-cubes",   type=int,   default=27)
    p.add_argument("--max-steps",   type=int,   default=400)
    p.add_argument("--formation",   type=str,   default="cube",
                   choices=["cube", "plane", "line", "random"])

    # ── Train ────────────────────────────────────────────────────────────────
    p.add_argument("--steps",       type=int,   default=1_000_000)
    p.add_argument("--n-envs",      type=int,   default=8)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--ent-coef",    type=float, default=0.01)
    p.add_argument("--save-path",   type=str,   default="models/ppo_swarm_mission")
    p.add_argument("--no-tb",       action="store_true",
                   help="Disable TensorBoard logging")

    # ── Eval / Render ────────────────────────────────────────────────────────
    p.add_argument("--load-path",   type=str,   default=None)
    p.add_argument("--episodes",    type=int,   default=20)
    p.add_argument("--scenario",    type=str,   default=None,
                   help="Filter scenarios by name substring (render mode)")
    p.add_argument("--stochastic",  action="store_true",
                   help="Use stochastic policy during eval / render")

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    cfg = AgentConfig(
        num_cubes         = args.num_cubes,
        max_steps         = args.max_steps,
        initial_formation = args.formation,
        total_timesteps   = args.steps,
        n_envs            = args.n_envs,
        learning_rate     = args.lr,
        ent_coef          = args.ent_coef,
        save_path         = args.save_path,
        tensorboard_log   = None if args.no_tb else "runs/swarm_ppo",
    )

    if args.train:
        train(cfg)

    elif args.eval:
        if not args.load_path:
            raise ValueError("--load-path required for --eval")
        agent = MaskablePPO.load(args.load_path)
        evaluate(
            agent, cfg,
            n_episodes=args.episodes,
            deterministic=not args.stochastic,
        )

    elif args.render:
        if not args.load_path:
            raise ValueError("--load-path required for --render")
        agent = MaskablePPO.load(args.load_path)
        render_episode(agent, cfg, scenario_name=args.scenario)


if __name__ == "__main__":
    main()