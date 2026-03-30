import numpy as np
from typing import Tuple, Optional, Dict
import gymnasium as gym
from gymnasium import spaces

from core.cube import Edge
from core.swarm import Swarm
from tasks.tasks import Task, FormPlaneTask
from mechanics.moves import MovementSystem, HingeMove
from rewards.metrics import SwarmMetrics
from visualization.renderer import SwarmVisualizer
from configs.formations import create_cube_formation, create_plane_formation, create_line_formation


class SwarmReconfigurationEnv(gym.Env):
    """
    Gymnasium environment for spacecraft swarm reconfiguration.
    
    The agent controls a swarm of modular cubesats and must reconfigure
    them to achieve various mission objectives.
    
    Observation Space:
        - Position of each cube (relative to centroid)
        - Orientation of each cube (flattened rotation matrix)
        - Connection state
        - Task-specific information
    
    Action Space:
        - Discrete: index into list of valid moves
        - Or: MultiDiscrete for (cube_id, edge, direction)
    
    Rewards:
        - Task-specific reward function
        - Small penalty for each step (encourages efficiency)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self,
                 num_cubes: int = 64,
                 task: Optional[Task] = None,
                 max_steps: int = 1000,
                 step_penalty: float = 0.001,
                 invalid_action_penalty: float = 0.1,
                 require_connectivity: bool = False,
                 initial_formation: str = 'cube',
                 render_mode: Optional[str] = None):
        """
        Args:
            num_cubes: Number of cubes in the swarm
            task: The task to complete (default: FormPlaneTask)
            max_steps: Maximum steps per episode
            step_penalty: Small penalty per step to encourage efficiency
            invalid_action_penalty: Penalty for attempting invalid actions
            require_connectivity: If True, moves that disconnect swarm are invalid
            initial_formation: Starting formation ('cube', 'plane', 'line', 'random')
            render_mode: 'human' for interactive, 'rgb_array' for image
        """
        super().__init__()
        
        self.num_cubes = num_cubes
        self.task = task or FormPlaneTask()
        self.max_steps = max_steps
        self.step_penalty = step_penalty
        self.invalid_action_penalty = invalid_action_penalty
        self.require_connectivity = require_connectivity
        self.initial_formation = initial_formation
        self.render_mode = render_mode
        
        # Will be initialized in reset()
        self.swarm: Optional[Swarm] = None
        self.movement: Optional[MovementSystem] = None
        self.current_step = 0
        self._valid_moves: list[HingeMove] = []
        
        # Visualization
        self._visualizer: Optional[SwarmVisualizer] = None
        
        # Define observation space
        # For each cube: position (3) + flattened orientation (9) = 12 values
        # Plus task info encoded as fixed-size vector
        self.cube_obs_size = 12  # 3 position + 9 orientation
        self.task_obs_size = 16  # Fixed size for task encoding
        
        obs_size = num_cubes * self.cube_obs_size + self.task_obs_size
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Action space: discrete index into valid moves
        # Maximum possible moves: 64 cubes * 12 edges * 2 directions = 1536
        # But we'll use action masking for invalid moves
        self.max_actions = num_cubes * 12 * 2
        self.action_space = spaces.Discrete(self.max_actions)
        
        # For action mapping
        self._action_to_move: Dict[int, HingeMove] = {}
        self._build_action_mapping()
    
    def _build_action_mapping(self) -> None:
        """Build mapping from action indices to HingeMove objects."""
        action_idx = 0
        edges = list(Edge)
        
        for cube_id in range(self.num_cubes):
            for edge in edges:
                for direction in [+1, -1]:
                    self._action_to_move[action_idx] = HingeMove(cube_id, edge, direction)
                    action_idx += 1
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector from current state."""
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # Get centroid for relative positions
        centroid = self.swarm.get_center_of_mass()
        
        idx = 0
        for cube_id in range(self.num_cubes):
            cube = self.swarm.get_cube(cube_id)
            if cube is not None:
                # Relative position
                obs[idx:idx+3] = np.array(cube.position) - np.array(centroid)
                idx += 3
                
                # Flattened orientation matrix
                obs[idx:idx+9] = cube.orientation.matrix.flatten()
                idx += 9
            else:
                idx += self.cube_obs_size
        
        # Task info encoding
        task_info = self.task.get_task_info()
        task_type_encoding = {
            'form_cube': 0,
            'form_plane': 1,
            'form_line': 2,
            'maximize_spread': 3,
            'minimize_surface': 4,
            'split_groups': 5,
            'target_configuration': 6,
        }
        
        task_type = task_info.get('task_type', 'unknown')
        obs[idx] = task_type_encoding.get(task_type, -1)
        idx += 1
        
        # Task-specific parameters
        if 'normal' in task_info:
            obs[idx:idx+3] = task_info['normal']
        idx += 3
        
        if 'axis' in task_info:
            obs[idx:idx+3] = task_info['axis']
        idx += 3
        
        if 'target_size' in task_info:
            obs[idx] = task_info['target_size']
        idx += 1
        
        if 'num_groups' in task_info:
            obs[idx] = task_info['num_groups']
        idx += 1
        
        # Progress toward goal
        obs[idx] = self.task.get_progress(self.swarm)
        idx += 1
        
        # Current metrics
        metrics = SwarmMetrics(self.swarm)
        obs[idx] = metrics.compactness()
        idx += 1
        obs[idx] = metrics.connectivity_ratio()
        idx += 1
        obs[idx] = metrics.maximum_baseline() / self.num_cubes  # Normalized
        idx += 1
        
        return obs
    
    def _get_action_mask(self) -> np.ndarray:
        """
        Get mask of valid actions.
        
        Returns:
            Boolean array where True means action is valid
        """
        mask = np.zeros(self.max_actions, dtype=bool)
        
        # Get all valid moves
        self._valid_moves = self.movement.get_all_valid_moves()
        
        # Mark valid moves in mask
        for move in self._valid_moves:
            # Find action index for this move
            for action_idx, mapped_move in self._action_to_move.items():
                if (mapped_move.cube_id == move.cube_id and 
                    mapped_move.pivot_edge == move.pivot_edge and
                    mapped_move.direction == move.direction):
                    mask[action_idx] = True
                    break
        
        return mask
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
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
        elif self.initial_formation == 'line':
            create_line_formation(self.swarm, length=self.num_cubes)
        elif self.initial_formation == 'random':
            # Start as cube, then do random moves
            size = round(self.num_cubes ** (1/3))
            create_cube_formation(self.swarm, size=size)
            # Do some random moves to scramble
            self.movement = MovementSystem(self.swarm, self.require_connectivity)
            for _ in range(50):
                valid = self.movement.get_all_valid_moves()
                if valid:
                    move = valid[self.np_random.integers(len(valid))]
                    self.movement.execute_move(move)
        else:
            raise ValueError(f"Unknown formation: {self.initial_formation}")
        
        # Create movement system
        self.movement = MovementSystem(self.swarm, self.require_connectivity)
        
        # Reset step counter
        self.current_step = 0
        
        # Get observation
        obs = self._get_observation()
        
        # Info dict
        info = {
            'action_mask': self._get_action_mask(),
            'valid_move_count': len(self._valid_moves),
            'task_progress': self.task.get_progress(self.swarm),
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Index of the action to take
            
        Returns:
            observation, reward, terminated, terminated, info
        """
        self.current_step += 1
        
        # Get the move for this action
        move = self._action_to_move.get(action)
        
        if move is None:
            # Invalid action index
            reward = -self.invalid_action_penalty
            obs = self._get_observation()
            info = {
                'action_mask': self._get_action_mask(),
                'valid_move_count': len(self._valid_moves),
                'task_progress': self.task.get_progress(self.swarm),
                'move_success': False,
                'move_reason': 'Invalid action index',
            }
            terminated = False
            truncated = self.current_step >= self.max_steps
            return obs, reward, terminated, truncated, info
        
        # Try to execute the move
        result = self.movement.execute_move(move)
        
        if result.success:
            # Compute task reward
            task_reward = self.task.compute_reward(self.swarm, move)
            
            # Apply step penalty to encourage efficiency
            reward = task_reward - self.step_penalty
            
            move_success = True
            move_reason = 'Success'
        else:
            # Move was invalid
            reward = -self.invalid_action_penalty
            move_success = False
            move_reason = result.reason
        
        # Check termination conditions
        terminated = self.task.is_complete(self.swarm) or self.task.is_failed(self.swarm)
        truncated = self.current_step >= self.max_steps
        
        # Get new observation
        obs = self._get_observation()
        
        # Build info dict
        info = {
            'action_mask': self._get_action_mask(),
            'valid_move_count': len(self._valid_moves),
            'task_progress': self.task.get_progress(self.swarm),
            'move_success': move_success,
            'move_reason': move_reason,
            'task_complete': self.task.is_complete(self.swarm),
            'task_failed': self.task.is_failed(self.swarm),
            'connections_broken': len(result.broken_connections) if result.success else 0,
            'connections_formed': len(result.new_connections) if result.success else 0,
        }
        
        # Add metrics to info
        metrics = SwarmMetrics(self.swarm)
        info['metrics'] = metrics.get_all_metrics()
        
        return obs, reward, terminated, truncated, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the current state."""
        if self.render_mode is None:
            return None
        
        if self._visualizer is None:
            self._visualizer = SwarmVisualizer(self.swarm)
        
        # Update visualizer's swarm reference
        self._visualizer.swarm = self.swarm
        
        title = (f"Step {self.current_step} | "
                f"Progress: {self.task.get_progress(self.swarm):.2%} | "
                f"Components: {len(self.swarm.get_connected_components())}")
        
        if self.render_mode == 'human':
            self._visualizer.render(title=title, show_connections=True)
            return None
        
        elif self.render_mode == 'rgb_array':
            # Render to buffer and return as numpy array
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            
            self._visualizer.render(title=title, show_connections=True)
            
            canvas = FigureCanvasAgg(self._visualizer._fig)
            canvas.draw()
            
            # Get RGB buffer
            buf = canvas.buffer_rgba()
            width, height = canvas.get_width_height()
            image = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)
            
            # Convert RGBA to RGB
            return image[:, :, :3]
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        if self._visualizer is not None:
            self._visualizer.close()
            self._visualizer = None
    
    def get_valid_actions(self) -> list[int]:
        """
        Get list of valid action indices.
        
        Useful for agents that need to know valid actions directly.
        """
        mask = self._get_action_mask()
        return [i for i, valid in enumerate(mask) if valid]
    
    def action_to_move(self, action: int) -> Optional[HingeMove]:
        """Convert action index to HingeMove object."""
        return self._action_to_move.get(action)
    
    def move_to_action(self, move: HingeMove) -> Optional[int]:
        """Convert HingeMove to action index."""
        for action_idx, mapped_move in self._action_to_move.items():
            if (mapped_move.cube_id == move.cube_id and
                mapped_move.pivot_edge == move.pivot_edge and
                mapped_move.direction == move.direction):
                return action_idx
        return None


class MaskedSwarmEnv(SwarmReconfigurationEnv):
    """
    Variant of SwarmReconfigurationEnv that uses a reduced action space.
    
    Instead of a large discrete space with masking, this uses a smaller
    space where action N means "take the Nth valid move."
    
    This is simpler for some RL algorithms but changes the action
    semantics between steps.
    """
    
    def __init__(self, *args, max_valid_actions: int = 256, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.max_valid_actions = max_valid_actions
        
        # Override action space to be smaller
        self.action_space = spaces.Discrete(max_valid_actions)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute step where action is index into current valid moves.
        """
        self.current_step += 1
        
        # Get current valid moves
        valid_moves = self.movement.get_all_valid_moves()
        
        if action < 0 or action >= len(valid_moves):
            # Invalid action index
            reward = -self.invalid_action_penalty
            obs = self._get_observation()
            info = {
                'valid_move_count': len(valid_moves),
                'task_progress': self.task.get_progress(self.swarm),
                'move_success': False,
                'move_reason': f'Action {action} out of range (max {len(valid_moves)})',
            }
            terminated = False
            truncated = self.current_step >= self.max_steps
            return obs, reward, terminated, truncated, info
        
        # Get the move
        move = valid_moves[action]
        
        # Execute the move
        result = self.movement.execute_move(move)
        
        if result.success:
            task_reward = self.task.compute_reward(self.swarm, move)
            reward = task_reward - self.step_penalty
            move_success = True
            move_reason = 'Success'
        else:
            # This shouldn't happen since we got move from valid_moves
            reward = -self.invalid_action_penalty
            move_success = False
            move_reason = result.reason
        
        terminated = self.task.is_complete(self.swarm) or self.task.is_failed(self.swarm)
        truncated = self.current_step >= self.max_steps
        
        obs = self._get_observation()
        
        # Update valid moves for next step
        new_valid_moves = self.movement.get_all_valid_moves()
        
        info = {
            'valid_move_count': len(new_valid_moves),
            'task_progress': self.task.get_progress(self.swarm),
            'move_success': move_success,
            'move_reason': move_reason,
            'task_complete': self.task.is_complete(self.swarm),
            'task_failed': self.task.is_failed(self.swarm),
        }
        
        metrics = SwarmMetrics(self.swarm)
        info['metrics'] = metrics.get_all_metrics()
        
        return obs, reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """Override to include valid move count in observation."""
        base_obs = super()._get_observation()
        
        # Could extend observation with valid move info if needed
        return base_obs