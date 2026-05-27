"""
rl/env_wrapper.py
=================
Standalone training environment wrapper for constellation control.

Extracted from train.py [2] to allow imports from multiple modules
(train.py, train_fast.py, parallel_env.py) without circular dependencies.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from core.swarm import Swarm
from core.constellation import (
    Constellation, SeparationRequirements, DockingRequirements, CommunicationRequirements
)
from configs.formations import create_cube_formation
from mechanics.moves import MovementSystem
from mechanics.constellation_moves import ConstellationController
from tasks.constellation_tasks import ConstellationTask, FormConstellationTask
from rl.observation_builder import ConstellationObservationBuilder, ActionMaskBuilder


class ConstellationTrainingEnv:
    """
    Training environment wrapper for constellation control.

    Handles the full environment loop including observation building,
    action decoding, and reward computation.

    Originally defined in train.py [2]. Extracted here so that parallel_env.py
    and train_fast.py can import it without pulling in the entire training loop.
    """

    def __init__(
        self,
        num_cubes: int = 64,
        task: Optional[ConstellationTask] = None,
        max_steps: int = 500,
        time_step: float = 10.0,
    ):
        self.num_cubes = num_cubes
        self.task = task or FormConstellationTask(
            target_num_groups=2,
            target_baseline=5000.0,
        )
        self.max_steps = max_steps
        self.time_step = time_step

        # Will be initialized on reset
        self.swarm: Optional[Swarm] = None
        self.constellation: Optional[Constellation] = None
        self.movement: Optional[MovementSystem] = None
        self.controller: Optional[ConstellationController] = None

        self.obs_builder = ConstellationObservationBuilder()
        self.mask_builder = ActionMaskBuilder()

        self.current_step = 0
        self.episode_reward = 0.0

        # Mission context (randomized each episode)
        self.sun_direction = (0.0, 0.0, -1.0)
        self.earth_direction = (1.0, 0.0, 0.0)
        self.target_direction = (0.0, 1.0, 0.0)
        self.sun_distance_au = 10.0
        self.mission_mode = 0  # Index into mode list

        # Reward shaping
        self.gamma = 0.99
        self._prev_potential = 0.0

        # Noop abuse tracking [2]
        self.consecutive_noops = 0
        self.max_consecutive_noops = 10
        self.total_noops = 0
        self.max_total_noops = 50

    # ------------------------------------------------------------------
    # Potential-based reward shaping
    # ------------------------------------------------------------------

    def _compute_potential(self) -> float:
        """Potential function for reward shaping [2]."""
        progress = self.task.get_progress(self.constellation)

        num_groups = self.constellation.get_num_groups()
        target_groups = getattr(self.task, "target_num_groups", 2)
        group_score = np.exp(-abs(num_groups - target_groups))

        baseline = self.constellation.get_max_baseline()
        target_baseline = getattr(self.task, "target_baseline", 5000.0)
        baseline_score = min(1.0, baseline / target_baseline)

        potential = 0.5 * progress + 0.25 * group_score + 0.25 * baseline_score
        return potential * 10.0

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None) -> Tuple:
        """
        Reset environment and return initial observation.

        Returns:
            (graph_data, mode_idx, env_features, action_masks)
        """
        if seed is not None:
            np.random.seed(seed)

        # Create swarm
        self.swarm = Swarm(self.num_cubes)
        size = int(np.ceil(self.num_cubes ** (1 / 3)))
        create_cube_formation(self.swarm, size=size)

        # Create constellation with generous propulsion [2]
        sep_reqs = SeparationRequirements(
            min_separation_delta_v=0.1,
            default_separation_velocity=1.0,
            min_group_size=1,
            max_groups=8,
            allow_single_cube_groups=True,
        )

        self.constellation = Constellation(self.swarm, sep_reqs)
        for ps in self.constellation._cube_propulsion.values():
            ps.max_delta_v = 100.0
            ps.remaining_delta_v = 100.0

        # Set propulsion budget
        for ps in self.constellation._cube_propulsion.values():
            ps.max_delta_v = 100.0
            ps.remaining_delta_v = 100.0

        for grp in self.constellation._groups.values():
            total = sum(
                self.constellation._cube_propulsion[cid].remaining_delta_v
                for cid in grp.cube_ids
            )
            grp.propulsion.max_delta_v = total
            grp.propulsion.remaining_delta_v = total

        # Create controllers
        self.movement = MovementSystem(self.swarm, require_connectivity=False)
        self.controller = ConstellationController(self.constellation)

        # Reset tracking state
        self.consecutive_noops = 0
        self.total_noops = 0
        self.current_step = 0
        self.episode_reward = 0.0

        # Randomize mission context
        self._randomize_mission_context()

        # Build observation
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            self.constellation,
            self.mission_mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au,
        )

        # Build action masks
        action_masks = self.mask_builder.build_action_masks(
            self.constellation, self.controller, self.movement
        )

        # Store initial potential for shaping
        self._prev_potential = self._compute_potential()

        return graph_data, mode_idx, env_features, action_masks

    # ------------------------------------------------------------------
    # Mission context randomization
    # ------------------------------------------------------------------

    def _randomize_mission_context(self) -> None:
        """Randomize mission context for domain randomization [2]."""
        # Random sun direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.sun_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        )

        # Random earth direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.earth_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        )

        # Random target direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.target_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi),
        )

        # Random distance and mode
        self.sun_distance_au = np.random.uniform(1.0, 30.0)
        self.mission_mode = np.random.randint(0, 6)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action_type: int, sub_action: int, action_masks: Dict) -> Tuple:
        """
        Execute action and return new observation, reward, done, info.

        Args:
            action_type: High-level action type (0-4) [3]
                0 = cube move, 1 = separation, 2 = docking,
                3 = maneuver, 4 = noop
            sub_action: Sub-action index within chosen type
            action_masks: Current action masks dict

        Returns:
            (graph_data, mode_idx, env_features, new_action_masks, reward, done, info)
        """
        self.current_step += 1

        # Track noop behaviour [2]
        if action_type == 4:
            self.consecutive_noops += 1
            self.total_noops += 1
        else:
            self.consecutive_noops = 0

        # Track state before action
        prev_progress = self.task.get_progress(self.constellation)
        prev_baseline = self.constellation.get_max_baseline()
        prev_num_groups = self.constellation.get_num_groups()

        # Decode and execute action
        action = self.mask_builder.decode_action(
            action_type, sub_action, self.constellation, action_masks
        )

        success = False
        delta_v_used = 0.0
        reason = ""

        if action_type == 0 and action is not None:  # Cube move
            result = self.movement.execute_move(action)
            success = result.success
            reason = result.reason if not success else "Move executed"

        elif action_type == 1 and action is not None:  # Separation [8]
            result = self.controller.execute_separation(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used

        elif action_type == 2 and action is not None:  # Docking [8]
            result = self.controller.execute_docking(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used

        elif action_type == 3 and action is not None:  # Maneuver [8]
            result = self.controller.execute_maneuver(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used

        elif action_type == 4:  # Noop
            success = True
            reason = "No operation"

        # Propagate time if multiple groups exist
        if self.constellation.get_num_groups() > 1:
            self.constellation.propagate(self.time_step)

        # ------------------------------------------------------------------
        # Hybrid reward [2]
        # ------------------------------------------------------------------

        # 1. Potential-based shaping (dense)
        curr_potential = self._compute_potential()
        shaping_reward = self.gamma * curr_potential - self._prev_potential
        self._prev_potential = curr_potential

        # 2. Small action costs
        if action_type == 4:
            action_cost = -0.02
        elif not success:
            action_cost = -0.05
        else:
            action_cost = -0.001 - 0.01 * delta_v_used

        # 3. Sparse terminal reward
        terminated = self.task.is_complete(self.constellation)
        truncated = self.current_step >= self.max_steps
        done = terminated or truncated

        terminal_reward = 0.0
        if done:
            final_progress = self.task.get_progress(self.constellation)
            if terminated:
                terminal_reward = 20.0
            else:
                terminal_reward = 10.0 * final_progress - 5.0

        reward = shaping_reward + action_cost + terminal_reward

        # ------------------------------------------------------------------
        # Build new observation
        # ------------------------------------------------------------------
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            self.constellation,
            self.mission_mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au,
        )

        new_action_masks = self.mask_builder.build_action_masks(
            self.constellation, self.controller, self.movement
        )

        info = {
            "action_success": success,
            "action_reason": reason,
            "delta_v_used": delta_v_used,
            "task_progress": self.task.get_progress(self.constellation),
            "task_complete": self.task.is_complete(self.constellation),
            "num_groups": self.constellation.get_num_groups(),
            "max_baseline": self.constellation.get_max_baseline(),
            "episode_reward": self.episode_reward,
            "mode_idx": self.mission_mode,
            "env_features": env_features.numpy().flatten()
            if hasattr(env_features, "numpy")
            else np.zeros(12),
        }

        # Noop abuse check [2]
        noop_abuse = (
            self.consecutive_noops >= self.max_consecutive_noops
            or self.total_noops >= self.max_total_noops
        )

        if noop_abuse:
            done = True
            reward = -5.0
            info["termination_reason"] = "noop_abuse"

        self.episode_reward += reward

        return graph_data, mode_idx, env_features, new_action_masks, reward, done, info

    # ------------------------------------------------------------------
    # Convenience accessors (used by parallel_env worker & agent)
    # ------------------------------------------------------------------

    def get_env_state(self) -> Dict:
        """
        Return references needed by the agent's get_action() method.

        This avoids pickling full environment state across process boundaries
        when the agent needs to compute observations directly.
        """
        return {
            "constellation": self.constellation,
            "controller": self.controller,
            "movement": self.movement,
            "mission_mode": self.mission_mode,
            "sun_direction": self.sun_direction,
            "earth_direction": self.earth_direction,
            "target_direction": self.target_direction,
            "sun_distance_au": self.sun_distance_au,
        }
