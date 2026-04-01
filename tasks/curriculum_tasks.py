"""
tasks/curriculum_tasks.py
========================
Adaptive curriculum learning for the constellation RL agent.

Design goals
------------
1. **Single agent, many tasks** — the same PPO model trains on all task types;
   task context is already part of the observation (mission_mode index + env_features).
2. **Curriculum by difficulty** — each task type has multiple difficulty tiers
   (easy → medium → hard).  The sampler advances tiers based on rolling success rate.
3. **Adaptive sampling** — tasks where the agent is struggling get more weight;
   tasks the agent has mastered get less (to avoid forgetting, they are never
   dropped entirely).
4. **Randomised context** — sun/Earth directions, distances, and cube counts are
   sampled per episode so the agent generalises.

Quick start
-----------
    from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler

    curriculum = TaskCurriculum(num_cubes_range=(8, 64))
    sampler = CurriculumSampler(curriculum)

    # In the training loop, replace the fixed task with:
    task, num_cubes = sampler.sample()
    env = ConstellationTrainingEnv(num_cubes=num_cubes, task=task)

    # After each episode, report outcome:
    sampler.record_outcome(task_key, success=info['task_complete'],
                           progress=info['task_progress'])
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from collections import deque

# Lazy imports to avoid circular deps at module load time
# All task classes are imported inside factory functions below.


# =============================================================================
# Difficulty tiers
# =============================================================================

@dataclass
class DifficultyTier:
    """Parameters for one difficulty level of a task."""
    name: str                               # 'easy' | 'medium' | 'hard'
    kwargs: Dict                            # kwargs forwarded to task constructor
    num_cubes_range: Tuple[int, int]        # (min, max) cubes for this tier
    promotion_threshold: float = 0.70       # rolling success rate to advance
    demotion_threshold: float = 0.30        # rolling success rate to fall back
    window: int = 20                        # episodes in rolling window


# =============================================================================
# Task registry
# =============================================================================

def _make_random_direction() -> Tuple[float, float, float]:
    """Uniform random unit vector on S²."""
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arccos(np.random.uniform(-1, 1))
    return (
        float(np.sin(phi) * np.cos(theta)),
        float(np.sin(phi) * np.sin(theta)),
        float(np.cos(phi)),
    )


def _build_earth_downlink(tier_kwargs: Dict):
    from tasks.constellation_tasks import EarthDownlinkTask
    earth_dir = _make_random_direction()
    return EarthDownlinkTask(earth_direction=earth_dir, **tier_kwargs)


def _build_local_relay(tier_kwargs: Dict):
    from tasks.constellation_tasks import LocalRelayTask
    return LocalRelayTask(**tier_kwargs)


def _build_sparse_aperture(tier_kwargs: Dict):
    from tasks.constellation_tasks import SparseApertureTask
    return SparseApertureTask(**tier_kwargs)


def _build_occultation_array(tier_kwargs: Dict):
    from tasks.constellation_tasks import OccultationArrayTask
    shadow_dir = _make_random_direction()
    return OccultationArrayTask(shadow_direction=shadow_dir, **tier_kwargs)


def _build_in_situ_field(tier_kwargs: Dict):
    from tasks.constellation_tasks import InSituFieldTask
    sun_dir = _make_random_direction()
    return InSituFieldTask(sun_direction=sun_dir, **tier_kwargs)


def _build_cruise_mode(tier_kwargs: Dict):
    from tasks.constellation_tasks import CruiseModeTask
    sun_dir = _make_random_direction()
    sun_dist = float(np.random.uniform(5.0, 30.0))
    return CruiseModeTask(sun_direction=sun_dir, sun_distance_au=sun_dist, **tier_kwargs)


def _build_solar_collection(tier_kwargs: Dict):
    from tasks.constellation_tasks import SolarCollectionTask
    sun_dir = _make_random_direction()
    sun_dist = float(np.random.uniform(5.0, 30.0))
    return SolarCollectionTask(sun_direction=sun_dir, sun_distance_au=sun_dist, **tier_kwargs)


def _build_thermal_shield(tier_kwargs: Dict):
    from tasks.constellation_tasks import ThermalShieldTask
    threat_dir = _make_random_direction()
    return ThermalShieldTask(threat_direction=threat_dir, **tier_kwargs)


def _build_damaged_reconfig(tier_kwargs: Dict):
    from tasks.constellation_tasks import DamagedReconfigTask, SparseApertureTask
    # Wrap a sparse aperture task (can generalise to other tasks)
    baseline = tier_kwargs.pop('target_baseline_m', 200.0)
    primary = SparseApertureTask(target_baseline_m=baseline)
    return DamagedReconfigTask(primary_task=primary, **tier_kwargs)


def _build_form_constellation(tier_kwargs: Dict):
    from tasks.constellation_tasks import FormConstellationTask
    return FormConstellationTask(**tier_kwargs)


# =============================================================================
# TaskCurriculum — defines tiers per task type
# =============================================================================

class TaskCurriculum:
    """
    Defines the full curriculum: which task types exist and how difficulty
    scales within each.

    Attributes
    ----------
    registry : dict
        Maps task_key -> list[DifficultyTier] (ordered easy → hard).
    builders : dict
        Maps task_key -> callable(tier_kwargs) -> ConstellationTask
    """

    def __init__(self, num_cubes_range: Tuple[int, int] = (8, 64)):
        self.min_cubes, self.max_cubes = num_cubes_range

        # -------------------------------------------------------------------
        # Define tiers for every task type.
        # num_cubes_range per tier can override the global range if desired.
        # -------------------------------------------------------------------
        self.registry: Dict[str, List[DifficultyTier]] = {

            'earth_downlink': [
                DifficultyTier('easy',   {'min_aperture_m2': 0.05}, (8, 16)),
                DifficultyTier('medium', {'min_aperture_m2': 0.15}, (16, 32)),
                DifficultyTier('hard',   {'min_aperture_m2': 0.30}, (32, 64)),
            ],

            'local_relay': [
                DifficultyTier('easy',   {'target_relay_bandwidth': 1e4}, (8, 16)),
                DifficultyTier('medium', {'target_relay_bandwidth': 1e5}, (16, 32)),
                DifficultyTier('hard',   {'target_relay_bandwidth': 1e6}, (32, 64)),
            ],

            'sparse_aperture': [
                DifficultyTier('easy',   {'target_baseline_m': 100.0,  'min_group_size': 2}, (8, 16)),
                DifficultyTier('medium', {'target_baseline_m': 500.0,  'min_group_size': 4}, (16, 32)),
                DifficultyTier('hard',   {'target_baseline_m': 2000.0, 'min_group_size': 8}, (32, 64)),
            ],

            'occultation_array': [
                DifficultyTier('easy',   {'target_perp_baseline_m': 50.0},  (8, 16)),
                DifficultyTier('medium', {'target_perp_baseline_m': 200.0}, (16, 32)),
                DifficultyTier('hard',   {'target_perp_baseline_m': 800.0}, (32, 64)),
            ],

            'in_situ_field': [
                DifficultyTier('easy',   {'target_volume_m3': 1e6,  'target_num_groups': 2}, (8, 16)),
                DifficultyTier('medium', {'target_volume_m3': 1e7,  'target_num_groups': 3}, (16, 32)),
                DifficultyTier('hard',   {'target_volume_m3': 1e9,  'target_num_groups': 4}, (32, 64)),
            ],

            'cruise_mode': [
                DifficultyTier('easy',   {}, (8, 16)),
                DifficultyTier('medium', {}, (16, 32)),
                DifficultyTier('hard',   {}, (32, 64)),
            ],

            'solar_collection': [
                DifficultyTier('easy',   {}, (8, 16)),
                DifficultyTier('medium', {}, (16, 32)),
                DifficultyTier('hard',   {}, (32, 64)),
            ],

            'thermal_shield': [
                DifficultyTier('easy',   {'target_shielded_fraction': 0.5, 'target_shield_depth': 1.0}, (8, 16)),
                DifficultyTier('medium', {'target_shielded_fraction': 0.7, 'target_shield_depth': 2.0}, (16, 32)),
                DifficultyTier('hard',   {'target_shielded_fraction': 0.9, 'target_shield_depth': 3.0}, (32, 64)),
            ],

            'damaged_reconfig': [
                DifficultyTier('easy',   {'damage_fraction': 0.10, 'graceful_threshold': 0.6, 'target_baseline_m': 100.0},  (8, 16)),
                DifficultyTier('medium', {'damage_fraction': 0.25, 'graceful_threshold': 0.6, 'target_baseline_m': 300.0},  (16, 32)),
                DifficultyTier('hard',   {'damage_fraction': 0.40, 'graceful_threshold': 0.5, 'target_baseline_m': 1000.0}, (32, 64)),
            ],

            'form_constellation': [
                DifficultyTier('easy',   {'target_num_groups': 2, 'target_baseline': 100.0},   (8, 16)),
                DifficultyTier('medium', {'target_num_groups': 3, 'target_baseline': 500.0},   (16, 32)),
                DifficultyTier('hard',   {'target_num_groups': 4, 'target_baseline': 2000.0},  (32, 64)),
            ],
        }

        self.builders: Dict[str, Callable] = {
            'earth_downlink':    _build_earth_downlink,
            'local_relay':       _build_local_relay,
            'sparse_aperture':   _build_sparse_aperture,
            'occultation_array': _build_occultation_array,
            'in_situ_field':     _build_in_situ_field,
            'cruise_mode':       _build_cruise_mode,
            'solar_collection':  _build_solar_collection,
            'thermal_shield':    _build_thermal_shield,
            'damaged_reconfig':  _build_damaged_reconfig,
            'form_constellation':_build_form_constellation,
        }

    def get_tier(self, task_key: str, tier_idx: int) -> DifficultyTier:
        return self.registry[task_key][tier_idx]

    def num_tiers(self, task_key: str) -> int:
        return len(self.registry[task_key])

    def build_task(self, task_key: str, tier_idx: int):
        """Instantiate a task object for the given key and tier."""
        tier = self.get_tier(task_key, tier_idx)
        return self.builders[task_key](dict(tier.kwargs))  # copy to allow mutation

    def sample_num_cubes(self, task_key: str, tier_idx: int) -> int:
        """Sample a valid cube count for this tier."""
        tier = self.get_tier(task_key, tier_idx)
        lo, hi = tier.num_cubes_range
        # Round to nearest power of 2 for cleaner formations
        n = np.random.randint(lo, hi + 1)
        return n


# =============================================================================
# CurriculumSampler — adaptive task weighting + tier progression
# =============================================================================

@dataclass
class _TaskState:
    tier_idx: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    # Track weighted sampling probability
    weight: float = 1.0


class CurriculumSampler:
    """
    Maintains per-task difficulty state and samples tasks for each episode.

    Sampling strategy
    -----------------
    * Tasks where the agent is struggling (low success) get *more* weight →
      more practice on difficult areas.
    * Tasks the agent has mastered are sampled less frequently but never dropped.
    * Difficulty tier advances when rolling success rate > promotion_threshold,
      and falls back when < demotion_threshold (min tier = 0).

    Usage
    -----
        sampler = CurriculumSampler(curriculum)

        # Each episode:
        task_key, task, num_cubes = sampler.sample()
        env = ConstellationTrainingEnv(num_cubes=num_cubes, task=task)
        ...
        sampler.record_outcome(task_key, success, progress)

        # Periodically inspect:
        sampler.print_status()
    """

    def __init__(
        self,
        curriculum: TaskCurriculum,
        mastery_floor_weight: float = 0.1,  # min relative weight for mastered tasks
    ):
        self.curriculum = curriculum
        self.mastery_floor_weight = mastery_floor_weight

        self._states: Dict[str, _TaskState] = {
            key: _TaskState() for key in curriculum.registry
        }

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sample(self) -> Tuple[str, object, int]:
        """
        Sample a (task_key, task_instance, num_cubes) triple for one episode.
        """
        keys = list(self._states.keys())
        weights = np.array([self._states[k].weight for k in keys], dtype=float)
        weights /= weights.sum()

        task_key = keys[np.random.choice(len(keys), p=weights)]
        state = self._states[task_key]

        task = self.curriculum.build_task(task_key, state.tier_idx)
        num_cubes = self.curriculum.sample_num_cubes(task_key, state.tier_idx)

        return task_key, task, num_cubes

    def record_outcome(
        self,
        task_key: str,
        success: bool,
        progress: float = 0.0,
    ) -> None:
        """
        Update the sampler after an episode completes.

        Args:
            task_key: Key of the task that was just run.
            success: Whether is_complete() was True at episode end.
            progress: get_progress() at episode end (0–1).
        """
        state = self._states[task_key]
        # Record as 1 for full success, partial credit for near-completion
        score = 1.0 if success else min(0.9, progress)
        state.history.append(score)

        self._maybe_advance_tier(task_key, state)
        self._update_weight(task_key, state)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rolling_success(self, state: _TaskState) -> float:
        if not state.history:
            return 0.5  # neutral prior
        return float(np.mean(state.history))

    def _maybe_advance_tier(self, task_key: str, state: _TaskState) -> None:
        if len(state.history) < 5:
            return  # wait for enough data

        max_tier = self.curriculum.num_tiers(task_key) - 1
        tier = self.curriculum.get_tier(task_key, state.tier_idx)
        success_rate = self._rolling_success(state)

        if success_rate >= tier.promotion_threshold and state.tier_idx < max_tier:
            state.tier_idx += 1
            state.history.clear()  # fresh window for new tier
            print(f"[Curriculum] ↑ {task_key} promoted to tier {state.tier_idx} "
                  f"(success rate {success_rate:.2f})")

        elif success_rate <= tier.demotion_threshold and state.tier_idx > 0:
            state.tier_idx -= 1
            state.history.clear()
            print(f"[Curriculum] ↓ {task_key} demoted to tier {state.tier_idx} "
                  f"(success rate {success_rate:.2f})")

    def _update_weight(self, task_key: str, state: _TaskState) -> None:
        """
        Weight is inversely proportional to rolling success.
        Mastered tasks (success > 0.85) are clamped to a floor weight.
        Struggling tasks (success < 0.4) get extra emphasis.
        """
        sr = self._rolling_success(state)

        if sr > 0.85:
            # Mastered — keep some exposure to prevent forgetting
            state.weight = self.mastery_floor_weight
        elif sr < 0.4:
            # Struggling — increase weight
            state.weight = 1.0 + (0.4 - sr) * 2.0  # up to 1.8×
        else:
            state.weight = 1.0

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def print_status(self) -> None:
        """Print a summary table of current tier and success rates."""
        print("\n--- Curriculum Status ---")
        print(f"{'Task':<22} {'Tier':<8} {'Success':>8} {'Weight':>8}")
        print("-" * 50)
        for key, state in self._states.items():
            tier = self.curriculum.get_tier(key, state.tier_idx)
            sr = self._rolling_success(state)
            print(f"{key:<22} {tier.name:<8} {sr:>8.2f} {state.weight:>8.2f}")
        print()

    def get_status_dict(self) -> Dict[str, Dict]:
        """Return status as a dict (for logging)."""
        out = {}
        for key, state in self._states.items():
            tier = self.curriculum.get_tier(key, state.tier_idx)
            out[key] = {
                'tier': state.tier_idx,
                'tier_name': tier.name,
                'rolling_success': self._rolling_success(state),
                'weight': state.weight,
            }
        return out