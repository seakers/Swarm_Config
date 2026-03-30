import numpy as np
from typing import Dict, Tuple, Optional, List, Union, Any
from dataclasses import dataclass
from enum import Enum, auto
import gymnasium as gym
from gymnasium import spaces

from core.swarm import Swarm
from core.cube import Edge
from core.constellation import (
    Constellation, GroupState, 
    SeparationRequirements, DockingRequirements, CommunicationRequirements,
    PropulsionSubsystem
)
from mechanics.constellation_moves import (
    SeparationAction, DockingAction, ManeuverAction,
    ConstellationController,
)
from rewards.constellation_metrics import ConstellationMetrics
from tasks.constellation_tasks import ConstellationTask, FormConstellationTask
from mechanics.moves import HingeMove, MoveResult, MovementSystem
from rewards.metrics import SwarmMetrics, SwarmFaceAnalyzer
from configs.formations import create_cube_formation, create_plane_formation
from visualization.renderer import SwarmVisualizer


class ActionLevel(Enum):
    """The level at which an action operates."""
    CUBE = auto()       # Individual cube hinge moves (within a group)
    GROUP = auto()       # Group-level actions (separation, docking, maneuvers)


@dataclass
class ConstellationActionResult:
    """Result of any action in the constellation environment."""
    success: bool
    reason: str = ""
    action_level: ActionLevel = ActionLevel.CUBE
    delta_v_used: float = 0.0
    new_group_id: Optional[int] = None


class ConstellationEnv(gym.Env):
    """
    Gymnasium environment for constellation-level spacecraft swarm control.
    
    This environment extends the basic swarm reconfiguration to include:
    - Separation of cubes into multiple groups
    - Orbital mechanics (simplified) for group propagation
    - Docking to rejoin groups
    - Delta-v management and propulsion constraints
    - Inter-group communication constraints
    
    The agent can perform two types of actions:
    1. Cube-level: Hinge moves within a group (same as SwarmReconfigurationEnv)
    2. Group-level: Separation, docking, and orbital maneuvers
    
    Observation Space:
    - Per-cube information (position, orientation)
    - Per-group information (position, velocity, delta-v)
    - Communication topology
    - Task-specific information
    
    Action Space:
    - Hybrid discrete space with action type selection
    - Or: Separate action spaces for each level
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self,
                 num_cubes: int = 64,
                 task: Optional[ConstellationTask] = None,
                 max_steps: int = 1000,
                 time_step: float = 10.0,  # Seconds per step for propagation
                 step_penalty: float = 0.001,
                 invalid_action_penalty: float = 0.1,
                 delta_v_penalty_weight: float = 0.01,
                 initial_formation: str = 'cube',
                 separation_reqs: Optional[SeparationRequirements] = None,
                 docking_reqs: Optional[DockingRequirements] = None,
                 comm_reqs: Optional[CommunicationRequirements] = None,
                 enable_group_actions: bool = True,
                 enable_cube_actions: bool = True,
                 render_mode: Optional[str] = None):
        """
        Args:
            num_cubes: Number of cubes in the swarm
            task: The constellation task to complete
            max_steps: Maximum steps per episode
            time_step: Time propagation per step (seconds)
            step_penalty: Small penalty per step
            invalid_action_penalty: Penalty for invalid actions
            delta_v_penalty_weight: Weight for delta-v usage penalty
            initial_formation: Starting formation
            separation_reqs: Separation requirements
            docking_reqs: Docking requirements
            comm_reqs: Communication requirements
            enable_group_actions: Allow separation/docking/maneuvers
            enable_cube_actions: Allow individual cube moves
            render_mode: Visualization mode
        """
        super().__init__()
        
        self.num_cubes = num_cubes
        self.task = task or FormConstellationTask(target_num_groups=2, target_baseline=5000.0)
        self.max_steps = max_steps
        self.time_step = time_step
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.delta_v_penalty_weight = delta_v_penalty_weight
        self.initial_formation = initial_formation
        self.enable_group_actions = enable_group_actions
        self.enable_cube_actions = enable_cube_actions
        self.render_mode = render_mode
        
        # Requirements (use defaults if not specified)
        self.separation_reqs = separation_reqs or SeparationRequirements()
        self.docking_reqs = docking_reqs or DockingRequirements()
        self.comm_reqs = comm_reqs or CommunicationRequirements()
        
        # Will be initialized in reset()
        self.swarm: Optional[Swarm] = None
        self.constellation: Optional[Constellation] = None
        self.movement: Optional[MovementSystem] = None
        self.controller: Optional[ConstellationController] = None
        self.current_step = 0
        
        # Visualization
        self._visualizer: Optional[SwarmVisualizer] = None
        
        # Define action space
        self._setup_action_space()
        
        # Define observation space
        self._setup_observation_space()
    
    def _setup_action_space(self):
        """
        Set up the action space.
        
        We use a Dict space with:
        - action_type: Which type of action (cube move, separate, dock, maneuver)
        - action_params: Parameters for the specific action
        """
        # Maximum dimensions for each action type
        self.max_cube_actions = self.num_cubes * 12 * 2  # cube × edge × direction
        self.max_separation_actions = 100  # Reasonable limit on separation options
        self.max_docking_actions = self.separation_reqs.max_groups ** 2
        self.max_maneuver_directions = 7  # 6 cardinal + zero
        self.max_groups = self.separation_reqs.max_groups
        
        # For simplicity, use a flat discrete action space
        # Action encoding:
        # [0, max_cube_actions): Cube hinge moves
        # [max_cube_actions, max_cube_actions + max_separation_actions): Separations
        # [next, next + max_docking_actions): Docking
        # [next, next + max_groups * max_maneuver_directions): Maneuvers
        # [last]: No-op
        
        self.cube_action_offset = 0
        self.separation_action_offset = self.max_cube_actions
        self.docking_action_offset = self.separation_action_offset + self.max_separation_actions
        self.maneuver_action_offset = self.docking_action_offset + self.max_docking_actions
        self.noop_action = self.maneuver_action_offset + self.max_groups * self.max_maneuver_directions
        
        self.total_actions = self.noop_action + 1
        
        self.action_space = spaces.Discrete(self.total_actions)
        
        # Build cube action mapping (same as base environment)
        self._cube_action_to_move: Dict[int, HingeMove] = {}
        self._build_cube_action_mapping()
    
    def _build_cube_action_mapping(self):
        """Build mapping from action indices to HingeMove objects."""
        action_idx = 0
        edges = list(Edge)
        
        for cube_id in range(self.num_cubes):
            for edge in edges:
                for direction in [+1, -1]:
                    self._cube_action_to_move[action_idx] = HingeMove(cube_id, edge, direction)
                    action_idx += 1
    
    def _setup_observation_space(self):
        """
        Set up the observation space.
        
        Includes:
        - Per-cube: position (3) + orientation (9) = 12
        - Per-group: position (3) + velocity (3) + num_cubes (1) + delta_v (1) = 8
        - Communication matrix (flattened)
        - Task info
        """
        # Cube observations
        self.cube_obs_size = 12  # position (3) + orientation (9)
        cube_total = self.num_cubes * self.cube_obs_size
        
        # Group observations
        self.group_obs_size = 8  # position (3) + velocity (3) + num_cubes (1) + delta_v (1)
        group_total = self.max_groups * self.group_obs_size
        
        # Communication (which groups can talk to each other)
        comm_matrix_size = self.max_groups * self.max_groups
        
        # Global state
        global_size = 10  # num_groups, time, total_delta_v, is_connected, task info, etc.
        
        # Task-specific
        task_size = 16
        
        total_obs_size = cube_total + group_total + comm_matrix_size + global_size + task_size
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_obs_size,),
            dtype=np.float32
        )
        
        # Store sizes for observation construction
        self._obs_sizes = {
            'cube': cube_total,
            'group': group_total,
            'comm': comm_matrix_size,
            'global': global_size,
            'task': task_size,
        }
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        idx = 0
        
        # --- Cube observations ---
        # Get centroid for relative positions
        centroid = self.swarm.get_center_of_mass()
        
        for cube_id in range(self.num_cubes):
            cube = self.swarm.get_cube(cube_id)
            if cube is not None:
                # Relative position (normalized by swarm extent)
                rel_pos = np.array(cube.position) - np.array(centroid)
                obs[idx:idx+3] = rel_pos / max(1.0, np.linalg.norm(rel_pos) + 1.0)
                idx += 3
                
                # Flattened orientation
                obs[idx:idx+9] = cube.orientation.matrix.flatten()
                idx += 9
            else:
                idx += self.cube_obs_size
        
        # --- Group observations ---
        groups = self.constellation.get_all_groups()
        
        for i in range(self.max_groups):
            if i < len(groups):
                group = groups[i]
                
                # Position (normalized to km)
                obs[idx:idx+3] = group.position / 1000.0
                idx += 3
                
                # Velocity (m/s)
                obs[idx:idx+3] = group.velocity
                idx += 3
                
                # Number of cubes (normalized)
                obs[idx] = len(group.cube_ids) / self.num_cubes
                idx += 1
                
                # Delta-v remaining (normalized)
                obs[idx] = group.propulsion.remaining_delta_v / 50.0
                idx += 1
            else:
                idx += self.group_obs_size
        
        # --- Communication matrix ---
        comm_graph = self.constellation.get_communication_graph()
        
        for i in range(self.max_groups):
            for j in range(self.max_groups):
                if i < len(groups) and j < len(groups):
                    gi_id = groups[i].group_id
                    gj_id = groups[j].group_id
                    
                    if gj_id in comm_graph.get(gi_id, set()):
                        obs[idx] = 1.0
                    else:
                        obs[idx] = 0.0
                idx += 1
        
        # --- Global state ---
        obs[idx] = len(groups) / self.max_groups  # Normalized num groups
        idx += 1
        
        obs[idx] = self.constellation.get_time() / 3600.0  # Time in hours
        idx += 1
        
        obs[idx] = self.constellation.get_total_delta_v_remaining() / (50.0 * self.num_cubes)
        idx += 1
        
        obs[idx] = 1.0 if self.constellation.is_constellation_connected() else 0.0
        idx += 1
        
        obs[idx] = self.constellation.get_max_baseline() / 100000.0  # Normalized to 100km
        idx += 1
        
        # Progress
        obs[idx] = self.task.get_progress(self.constellation)
        idx += 1
        
        # Padding for global
        idx += 4
        
        # --- Task info ---
        task_info = self.task.get_task_info()
        
        task_type_map = {
            'form_constellation': 0,
            'rendezvous_and_dock': 1,
            'stereo_imaging': 2,
            'multi_point_sensing': 3,
        }
        obs[idx] = task_type_map.get(task_info.get('task_type', ''), -1)
        idx += 1
        
        if 'target_num_groups' in task_info:
            obs[idx] = task_info['target_num_groups'] / self.max_groups
        idx += 1
        
        if 'target_baseline' in task_info:
            obs[idx] = task_info['target_baseline'] / 100000.0
        idx += 1
        
        # Fill remaining task space
        idx += (self._obs_sizes['task'] - 3)
        
        return obs
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions.
        """
        mask = np.zeros(self.total_actions, dtype=bool)
        
        # --- Cube actions ---
        if self.enable_cube_actions:
            # Get valid moves for each group
            for group in self.constellation.get_all_groups():
                # Create movement system for this group's cubes
                for cube_id in group.cube_ids:
                    valid_moves = self.movement.get_valid_moves(cube_id)
                    
                    for move in valid_moves:
                        # Find action index
                        for action_idx, mapped_move in self._cube_action_to_move.items():
                            if (mapped_move.cube_id == move.cube_id and
                                mapped_move.pivot_edge == move.pivot_edge and
                                mapped_move.direction == move.direction):
                                mask[self.cube_action_offset + action_idx] = True
                                break
        
        # --- Separation actions ---
        if self.enable_group_actions:
            valid_separations = self.controller.get_valid_separation_actions()
            
            for i, action in enumerate(valid_separations[:self.max_separation_actions]):
                mask[self.separation_action_offset + i] = True
            
            # Store for later retrieval
            self._valid_separations = valid_separations
        
        # --- Docking actions ---
        if self.enable_group_actions:
            valid_dockings = self.controller.get_valid_docking_actions()
            
            for i, action in enumerate(valid_dockings[:self.max_docking_actions]):
                mask[self.docking_action_offset + i] = True
            
            self._valid_dockings = valid_dockings
        
        # --- Maneuver actions ---
        if self.enable_group_actions:
            groups = self.constellation.get_all_groups()
            
            for g_idx, group in enumerate(groups):
                if g_idx >= self.max_groups:
                    break
                
                # Check if group has propulsion
                if group.propulsion.remaining_delta_v > 0.1:
                    # Allow all maneuver directions
                    for d_idx in range(self.max_maneuver_directions):
                        action_idx = self.maneuver_action_offset + g_idx * self.max_maneuver_directions + d_idx
                        mask[action_idx] = True
        
        # --- No-op is always valid ---
        mask[self.noop_action] = True
        
        return mask
    
    def _decode_action(self, action: Tuple[int, int]) -> Tuple[str, Any]:
        """
        Decode action index into action type and parameters.
        
        Returns:
            (action_type, action_data)
        """
        action_type_idx = action[0]
        action_data = action[1]

        if action_type_idx == 4:
            return ('noop', None)
        
        if action_type_idx == 0:
            # Cube action
            move = self._cube_action_to_move.get(action_data)
            return ('cube_move', move)
        
        elif action_type_idx == 1:
            # Separation action
            if hasattr(self, '_valid_separations') and action_data < len(self._valid_separations):
                return ('separation', self._valid_separations[action_data])
            return ('invalid', None)
        
        elif action_type_idx == 2:
            # Docking action
            if hasattr(self, '_valid_dockings') and action_data < len(self._valid_dockings):
                return ('docking', self._valid_dockings[action_data])
            return ('invalid', None)
        
        elif action_type_idx == 3:
            # Maneuver action
            group_idx = action_data // self.max_maneuver_directions
            direction_idx = action_data % self.max_maneuver_directions
            
            # Get group
            groups = self.constellation.get_all_groups()
            if group_idx < len(groups):
                group = groups[group_idx]
                
                # Decode direction
                directions = [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                    (0, 0, 0),  # No maneuver
                ]
                
                if direction_idx < len(directions):
                    direction = directions[direction_idx]
                    delta_v = tuple(d * 1.0 for d in direction)  # 1 m/s maneuver
                    maneuver = ManeuverAction(group.group_id, delta_v)
                    return ('maneuver', maneuver)
            
            return ('invalid', None)
        
        return ('invalid', None)
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Create new swarm
        self.swarm = Swarm(self.num_cubes)
        
        # Initialize formation
        if self.initial_formation == 'cube':
            size = round(self.num_cubes ** (1/3))
            create_cube_formation(self.swarm, size=size)
        elif self.initial_formation == 'plane':
            side = round(self.num_cubes ** 0.5)
            create_plane_formation(self.swarm, width=side, height=side)
        else:
            # Default to cube
            size = round(self.num_cubes ** (1/3))
            create_cube_formation(self.swarm, size=size)
        
        # Create constellation
        self.constellation = Constellation(
            self.swarm,
            self.separation_reqs,
            self.docking_reqs,
            self.comm_reqs
        )
        
        # Create controllers
        self.movement = MovementSystem(self.swarm, require_connectivity=False)
        self.controller = ConstellationController(self.constellation)
        
        # Reset step counter
        self.current_step = 0
        
        # Clear cached valid actions
        self._valid_separations = []
        self._valid_dockings = []
        
        # Get observation
        obs = self._get_observation()
        
        # Get action mask
        action_mask = self._get_action_mask()
        
        # Build info dict
        info = {
            'action_mask': action_mask,
            'num_groups': self.constellation.get_num_groups(),
            'max_baseline': self.constellation.get_max_baseline(),
            'total_delta_v': self.constellation.get_total_delta_v_remaining(),
            'is_connected': self.constellation.is_constellation_connected(),
            'task_progress': self.task.get_progress(self.constellation),
        }
        
        return obs, info
    
    def step(self, action: Tuple[int, int]) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step in the environment."""
        self.current_step += 1
        
        # Decode action
        action_type, action_data = self._decode_action(action)
        
        # Track delta-v usage
        delta_v_used = 0.0
        action_success = False
        action_reason = ""
        
        # Execute action based on type
        if action_type == 'noop':
            action_success = True
            action_reason = "No operation"
        
        elif action_type == 'cube_move':
            if action_data is not None:
                move = action_data
                
                # Check which group this cube belongs to
                group = self.constellation.get_group_for_cube(move.cube_id)
                
                if group is not None:
                    # Execute the move
                    result = self.movement.execute_move(move)
                    action_success = result.success
                    action_reason = result.reason if not result.success else "Move executed"
                else:
                    action_success = False
                    action_reason = "Cube not in any group"
            else:
                action_success = False
                action_reason = "Invalid cube move"
        
        elif action_type == 'separation':
            if action_data is not None:
                result = self.controller.execute_separation(action_data)
                action_success = result.success
                action_reason = result.reason
                delta_v_used = result.delta_v_used
            else:
                action_success = False
                action_reason = "Invalid separation action"
        
        elif action_type == 'docking':
            if action_data is not None:
                result = self.controller.execute_docking(action_data)
                action_success = result.success
                action_reason = result.reason
                delta_v_used = result.delta_v_used
            else:
                action_success = False
                action_reason = "Invalid docking action"
        
        elif action_type == 'maneuver':
            if action_data is not None:
                result = self.controller.execute_maneuver(action_data)
                action_success = result.success
                action_reason = result.reason
                delta_v_used = result.delta_v_used
            else:
                action_success = False
                action_reason = "Invalid maneuver action"
        
        else:
            action_success = False
            action_reason = "Unknown action type"
        
        # Propagate time (groups move according to their velocities)
        if self.constellation.get_num_groups() > 1:
            self.constellation.propagate(self.time_step)
        
        # Compute reward
        if action_success:
            task_reward = self.task.compute_reward(self.constellation)
            delta_v_penalty = self.delta_v_penalty_weight * delta_v_used
            reward = task_reward - self.step_penalty - delta_v_penalty
        else:
            reward = -self.invalid_action_penalty
        
        # Check termination
        terminated = self.task.is_complete(self.constellation) or self.task.is_failed(self.constellation)
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        obs = self._get_observation()
        
        # Get new action mask
        action_mask = self._get_action_mask()
        
        # Build info dict
        info = {
            'action_mask': action_mask,
            'action_type': action_type,
            'action_success': action_success,
            'action_reason': action_reason,
            'delta_v_used': delta_v_used,
            'num_groups': self.constellation.get_num_groups(),
            'max_baseline': self.constellation.get_max_baseline(),
            'total_delta_v': self.constellation.get_total_delta_v_remaining(),
            'is_connected': self.constellation.is_constellation_connected(),
            'task_progress': self.task.get_progress(self.constellation),
            'task_complete': self.task.is_complete(self.constellation),
            'task_failed': self.task.is_failed(self.constellation),
            'simulation_time': self.constellation.get_time(),
        }
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if self.render_mode is None:
            return None
        
        if self._visualizer is None:
            self._visualizer = SwarmVisualizer(self.swarm)
        
        # Update visualizer's swarm reference
        self._visualizer.swarm = self.swarm
        
        # Create title with constellation info
        title = (f"Step {self.current_step} | "
                f"Groups: {self.constellation.get_num_groups()} | "
                f"Baseline: {self.constellation.get_max_baseline():.0f}m | "
                f"Progress: {self.task.get_progress(self.constellation):.2%}")
        
        if self.render_mode == 'human':
            self._visualizer.render(title=title, show_connections=True)
            return None
        
        elif self.render_mode == 'rgb_array':
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            self._visualizer.render(title=title, show_connections=True)
            
            canvas = FigureCanvasAgg(self._visualizer._fig)
            canvas.draw()
            
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            
            return image[:, :, :3]
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
    
    def get_constellation_state(self) -> Dict:
        """Get detailed constellation state for debugging."""
        return self.constellation.get_constellation_state()


# =============================================================================
# Simplified constellation environment with separate action spaces
# =============================================================================

class HierarchicalConstellationEnv(gym.Env):
    """
    A hierarchical environment where the agent first chooses action type,
    then chooses specific action within that type.
    
    This can be easier to learn than the flat action space.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self,
                 num_cubes: int = 64,
                 task: Optional[ConstellationTask] = None,
                 max_steps: int = 1000,
                 time_step: float = 10.0,
                 render_mode: Optional[str] = None,
                 **kwargs):
        super().__init__()
        
        self.num_cubes = num_cubes
        self.task = task or FormConstellationTask()
        self.max_steps = max_steps
        self.time_step = time_step
        self.render_mode = render_mode
        self.kwargs = kwargs
        
        # Action type space
        # 0: cube_move, 1: separate, 2: dock, 3: maneuver, 4: noop
        self.num_action_types = 5
        
        # Sub-action spaces (will select from valid actions at runtime)
        self.max_sub_actions = 256
        
        # Combined action space: (action_type, sub_action_index)
        self.action_space = spaces.MultiDiscrete([self.num_action_types, self.max_sub_actions])
        
        # Observation space (same as ConstellationEnv)
        # Simplified for this example
        obs_size = num_cubes * 12 + 8 * 8 + 64 + 10 + 16
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )
        
        # Initialize on first reset
        self.constellation: Optional[Constellation] = None
        self.swarm: Optional[Swarm] = None
        self.movement: Optional[MovementSystem] = None
        self.controller: Optional[ConstellationController] = None
        self.current_step = 0
        
        # Cache valid actions per type
        self._valid_actions_cache: Dict[int, List] = {}
    
    def _update_valid_actions_cache(self):
        """Update the cache of valid actions for each type."""
        self._valid_actions_cache = {
            0: [],  # Cube moves
            1: [],  # Separations
            2: [],  # Dockings
            3: [],  # Maneuvers
            4: [None],  # Noop (always one valid)
        }
        
        # Cube moves
        for cube_id in range(self.num_cubes):
            moves = self.movement.get_valid_moves(cube_id)
            self._valid_actions_cache[0].extend(moves)
        
        # Separations
        self._valid_actions_cache[1] = self.controller.get_valid_separation_actions()
        
        # Dockings
        self._valid_actions_cache[2] = self.controller.get_valid_docking_actions()
        
        # Maneuvers
        maneuvers = self.controller.get_valid_maneuver_actions(max_delta_v=1.0)
        self._valid_actions_cache[3] = maneuvers
    
    def reset(self, seed: Optional[int] = None, 
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Create swarm and constellation
        self.swarm = Swarm(self.num_cubes)
        size = round(self.num_cubes ** (1/3))
        create_cube_formation(self.swarm, size=size)
        
        self.constellation = Constellation(self.swarm)
        self.movement = MovementSystem(self.swarm, require_connectivity=False)
        self.controller = ConstellationController(self.constellation)
        
        self.current_step = 0
        
        # Update valid actions
        self._update_valid_actions_cache()
        
        obs = self._get_observation()
        info = self._get_info()
        
        return obs, info
    
    def _get_observation(self) -> np.ndarray:
        """Get observation vector."""
        # Simplified observation
        obs = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        idx = 0
        centroid = self.swarm.get_center_of_mass()
        
        # Cube positions and orientations
        for cube_id in range(self.num_cubes):
            cube = self.swarm.get_cube(cube_id)
            if cube:
                rel_pos = np.array(cube.position) - np.array(centroid)
                obs[idx:idx+3] = rel_pos / 10.0
                idx += 3
                obs[idx:idx+9] = cube.orientation.matrix.flatten()
                idx += 9
            else:
                idx += 12
        
        # Group info
        groups = self.constellation.get_all_groups()
        for i in range(8):
            if i < len(groups):
                g = groups[i]
                obs[idx:idx+3] = g.position / 1000.0
                obs[idx+3:idx+6] = g.velocity
                obs[idx+6] = len(g.cube_ids) / self.num_cubes
                obs[idx+7] = g.propulsion.remaining_delta_v / 50.0
            idx += 8
        
        # Communication matrix (8x8 = 64)
        idx += 64
        
        # Global state
        obs[idx] = len(groups) / 8.0
        obs[idx+1] = self.constellation.get_time() / 3600.0
        obs[idx+2] = self.constellation.get_total_delta_v_remaining() / (50 * self.num_cubes)
        obs[idx+3] = float(self.constellation.is_constellation_connected())
        obs[idx+4] = self.constellation.get_max_baseline() / 100000.0
        obs[idx+5] = self.task.get_progress(self.constellation)
        idx += 10
        
        # Task info
        idx += 16
        
        return obs
    
    def _get_info(self) -> Dict:
        """Get info dict including action masks."""
        # Create action mask per type
        action_type_mask = np.zeros(self.num_action_types, dtype=bool)
        sub_action_counts = []
        
        for action_type in range(self.num_action_types):
            valid_count = len(self._valid_actions_cache.get(action_type, []))
            action_type_mask[action_type] = valid_count > 0
            sub_action_counts.append(valid_count)
        
        return {
            'action_type_mask': action_type_mask,
            'sub_action_counts': sub_action_counts,
            'num_groups': self.constellation.get_num_groups(),
            'max_baseline': self.constellation.get_max_baseline(),
            'task_progress': self.task.get_progress(self.constellation),
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action."""
        self.current_step += 1
        
        action_type = int(action[0])
        sub_action_idx = int(action[1])
        
        # Get valid actions for this type
        valid_actions = self._valid_actions_cache.get(action_type, [])
        
        success = False
        reason = ""
        delta_v_used = 0.0
        
        if sub_action_idx < len(valid_actions):
            action_data = valid_actions[sub_action_idx]
            
            if action_type == 0:  # Cube move
                result = self.movement.execute_move(action_data)
                success = result.success
                reason = result.reason
            
            elif action_type == 1:  # Separation
                result = self.controller.execute_separation(action_data)
                success = result.success
                reason = result.reason
                delta_v_used = result.delta_v_used
            
            elif action_type == 2:  # Docking
                result = self.controller.execute_docking(action_data)
                success = result.success
                reason = result.reason
                delta_v_used = result.delta_v_used
            
            elif action_type == 3:  # Maneuver
                result = self.controller.execute_maneuver(action_data)
                success = result.success
                reason = result.reason
                delta_v_used = result.delta_v_used
            
            elif action_type == 4:  # Noop
                success = True
                reason = "No operation"
        
        else:
            success = False
            reason = f"Invalid sub-action index {sub_action_idx} for action type {action_type}"
        
        # Propagate time
        if self.constellation.get_num_groups() > 1:
            self.constellation.propagate(self.time_step)
        
        # Update valid actions cache
        self._update_valid_actions_cache()
        
        # Compute reward
        if success:
            task_reward = self.task.compute_reward(self.constellation)
            reward = task_reward - 0.001 - 0.01 * delta_v_used
        else:
            reward = -0.1
        
        # Check termination
        terminated = self.task.is_complete(self.constellation) or self.task.is_failed(self.constellation)
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        info = self._get_info()
        info.update({
            'action_type': action_type,
            'action_success': success,
            'action_reason': reason,
            'delta_v_used': delta_v_used,
            'task_complete': self.task.is_complete(self.constellation),
            'task_failed': self.task.is_failed(self.constellation),
        })
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render current state."""
        # Use same rendering as ConstellationEnv
        return None
    
    def close(self) -> None:
        """Clean up."""
        pass