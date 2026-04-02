"""
visualize_agent.py
==================
Visualization script for trained constellation control agents.

Loads a trained model and visualizes its behavior on various tasks,
showing both global constellation view and per-group cube configurations.

Usage:
    python visualize_agent.py --checkpoint ./checkpoints/best_model.pt --task sparse_aperture
    python visualize_agent.py --checkpoint ./checkpoints/final_model.pt --task form_constellation --num-cubes 32
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# Project imports - adjust paths as needed
from rl.ppo_agent import ConstellationPPOAgent, PPOConfig
from rl.train import ConstellationTrainingEnv, TrainingConfig
from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler
from core.swarm import Swarm
from core.constellation import Constellation
from core.cube import Face
from core.cube_faces import FaceFunction


# =============================================================================
# Face Colors Configuration
# =============================================================================

# Color mapping for each face function type
FACE_FUNCTION_COLORS = {
    FaceFunction.SOLAR_ARRAY: '#FFD700',      # Gold - solar panels
    FaceFunction.ANTENNA_HIGH_GAIN: '#00CED1',           # Dark Cyan - communication
    FaceFunction.CAMERA: '#4169E1',            # Royal Blue - imaging
    FaceFunction.RADIATOR: '#FF6347',          # Tomato Red - thermal
    FaceFunction.ANTENNA_INTER_SAT: '#32CD32',      # Lime Green - inter-satellite communication
    FaceFunction.SCIENCE_INSTRUMENTS: "#DE10C6",        # magenta - science
}

# Direction vector colors
DIRECTION_COLORS = {
    'sun': '#FFA500',       # Orange
    'earth': '#1E90FF',     # Dodger Blue
    'target': '#FF1493',    # Deep Pink
    'velocity': '#00FF00',  # Green
}


# =============================================================================
# 3D Arrow for Direction Vectors
# =============================================================================

class Arrow3D(FancyArrowPatch):
    """3D arrow for showing direction vectors."""
    
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


# =============================================================================
# Cube Renderer
# =============================================================================

class CubeRenderer:
    """Renders individual cubes with colored faces."""
    
    def __init__(self, cube_size: float = 1.0):
        self.cube_size = cube_size
        
    def get_cube_vertices(self, position: Tuple[float, float, float]) -> np.ndarray:
        """Get the 8 vertices of a unit cube at given position."""
        x, y, z = position
        s = self.cube_size / 2
        
        vertices = np.array([
            [x - s, y - s, z - s],  # 0
            [x + s, y - s, z - s],  # 1
            [x + s, y + s, z - s],  # 2
            [x - s, y + s, z - s],  # 3
            [x - s, y - s, z + s],  # 4
            [x + s, y - s, z + s],  # 5
            [x + s, y + s, z + s],  # 6
            [x - s, y + s, z + s],  # 7
        ])
        return vertices
    
    def get_face_polygons(self, vertices: np.ndarray) -> List[np.ndarray]:
        """Get the 6 face polygons from vertices."""
        # Face indices: +X, -X, +Y, -Y, +Z, -Z
        face_indices = [
            [1, 2, 6, 5],  # +X face
            [0, 3, 7, 4],  # -X face
            [2, 3, 7, 6],  # +Y face
            [0, 1, 5, 4],  # -Y face
            [4, 5, 6, 7],  # +Z face
            [0, 1, 2, 3],  # -Z face
        ]
        
        return [vertices[indices] for indices in face_indices]
    
    def render_cube(self, ax, position: Tuple[float, float, float],
                    face_colors: Optional[List[str]] = None,
                    alpha: float = 0.8,
                    edge_color: str = 'black',
                    edge_width: float = 0.5) -> None:
        """Render a single cube with colored faces."""
        
        vertices = self.get_cube_vertices(position)
        faces = self.get_face_polygons(vertices)
        
        for i, face in enumerate(faces):
            color = face_colors[i % len(face_colors)]
            poly = Poly3DCollection([face], alpha=alpha)
            poly.set_facecolor(color)
            poly.set_edgecolor(edge_color)
            poly.set_linewidth(edge_width)
            ax.add_collection3d(poly)


# =============================================================================
# Constellation Visualizer
# =============================================================================

class ConstellationVisualizer:
    """
    Visualizes constellation state with multiple views.
    
    Features:
    - Global view showing all groups and their positions
    - Per-group detail views showing cube configurations
    - Direction vectors for sun, earth, target
    - Animation support for real-time visualization
    """
    
    def __init__(self, figsize: Tuple[int, int] = (16, 10)):
        self.figsize = figsize
        self.cube_renderer = CubeRenderer()
        self.fig = None
        self.axes = {}
        
    def setup_figure(self, num_groups: int = 1) -> None:
        """Set up the figure with appropriate subplots."""
        plt.close('all')
        
        # Layout: Global view on left, group views on right
        if num_groups <= 2:
            self.fig = plt.figure(figsize=self.figsize)
            gs = self.fig.add_gridspec(2, 2, width_ratios=[1.5, 1])
            
            # Global view spans left column
            self.axes['global'] = self.fig.add_subplot(gs[:, 0], projection='3d')
            
            # Group views on right
            for i in range(min(num_groups, 2)):
                self.axes[f'group_{i}'] = self.fig.add_subplot(gs[i, 1], projection='3d')
        else:
            # More groups: 2x3 grid
            self.fig = plt.figure(figsize=(18, 12))
            gs = self.fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
            
            self.axes['global'] = self.fig.add_subplot(gs[:, 0], projection='3d')
            
            group_positions = [(0, 1), (0, 2), (1, 1), (1, 2)]
            for i in range(min(num_groups, 4)):
                row, col = group_positions[i]
                self.axes[f'group_{i}'] = self.fig.add_subplot(gs[row, col], projection='3d')
        
        plt.tight_layout()
    
    def draw_direction_vector(self, ax, origin: np.ndarray, direction: Tuple[float, float, float],
                              label: str, color: str, scale: float = 10.0) -> None:
        """Draw a direction vector as an arrow."""
        direction = np.array(direction)
        direction = direction / (np.linalg.norm(direction) + 1e-8)  # Normalize
        
        end = origin + direction * scale
        
        arrow = Arrow3D(
            [origin[0], end[0]],
            [origin[1], end[1]],
            [origin[2], end[2]],
            mutation_scale=15,
            lw=2,
            arrowstyle="-|>",
            color=color
        )
        ax.add_artist(arrow)
        
        # Add label
        ax.text(end[0], end[1], end[2], f'  {label}', color=color, fontsize=9, fontweight='bold')
    
    def render_global_view(self, constellation: Constellation,
                           sun_direction: Tuple[float, float, float],
                           earth_direction: Tuple[float, float, float],
                           target_direction: Tuple[float, float, float],
                           title: str = "Global Constellation View") -> None:
        """Render the global view showing all groups."""
        ax = self.axes['global']
        ax.clear()
        
        groups = constellation.get_all_groups()
        
        # Collect all positions for axis scaling
        all_positions = []
        group_colors = plt.cm.Set1(np.linspace(0, 1, max(len(groups), 1)))
        
        for i, group in enumerate(groups):
            pos = group.position
            all_positions.append(pos)
            
            # Draw group as a sphere/marker
            ax.scatter(*pos, s=200, c=[group_colors[i]], marker='o',
                       label=f'Group {group.group_id} ({len(group.cube_ids)} cubes)',
                       edgecolors='black', linewidth=2)
            
            # Draw velocity vector if moving
            vel = group.velocity
            if np.linalg.norm(vel) > 0.1:
                vel_scale = min(np.linalg.norm(vel) * 10, 500)
                vel_norm = vel / (np.linalg.norm(vel) + 1e-8)
                ax.quiver(*pos, *vel_norm, length=vel_scale, color='green',
                          arrow_length_ratio=0.1, alpha=0.7)
        
        # Calculate center and scale for direction vectors
        if all_positions:
            center = np.mean(all_positions, axis=0)
            max_dist = max(np.max(np.abs(np.array(all_positions) - center)), 100)
        else:
            center = np.zeros(3)
            max_dist = 100
        
        vector_scale = max_dist * 0.6
        
        # Draw direction vectors from center
        self.draw_direction_vector(ax, center, sun_direction, 'Sun',
                                   DIRECTION_COLORS['sun'], vector_scale)
        self.draw_direction_vector(ax, center, earth_direction, 'Earth',
                                   DIRECTION_COLORS['earth'], vector_scale)
        self.draw_direction_vector(ax, center, target_direction, 'Target',
                                   DIRECTION_COLORS['target'], vector_scale)
        
        # Set axis properties
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        
        # Equal aspect ratio
        if all_positions:
            positions = np.array(all_positions)
            max_range = np.max(np.ptp(positions, axis=0)) / 2 + max_dist * 0.5
            mid = np.mean(positions, axis=0)
            ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
            ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
            ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    def render_group_view(self, swarm: Swarm, group_cube_ids: set,
                          group_idx: int,
                          sun_direction: Tuple[float, float, float],
                          earth_direction: Tuple[float, float, float],
                          target_direction: Tuple[float, float, float],
                          title: str = None) -> None:
        """Render detailed view of a single group's cube configuration."""
        
        ax_key = f'group_{group_idx}'
        if ax_key not in self.axes:
            return
            
        ax = self.axes[ax_key]
        ax.clear()
        
        # Get positions of cubes in this group
        positions = []
        for cube_id in group_cube_ids:
            cube = swarm.get_cube(cube_id)
            if cube is not None:
                positions.append(cube.position)
                
                # Get face colors based on cube's face functions
                face_colors = self._get_cube_face_colors(cube)
                
                # Render the cube
                self.cube_renderer.render_cube(ax, cube.position, face_colors)
        
        if not positions:
            ax.set_title(f"Group {group_idx} (empty)")
            return
        
        positions = np.array(positions)
        center = np.mean(positions, axis=0)
        
        # Draw direction vectors
        vector_scale = max(np.max(np.ptp(positions, axis=0)), 2) * 2.0
        self.draw_direction_vector(ax, center, sun_direction, 'S',
                                   DIRECTION_COLORS['sun'], vector_scale)
        self.draw_direction_vector(ax, center, earth_direction, 'E',
                                   DIRECTION_COLORS['earth'], vector_scale)
        self.draw_direction_vector(ax, center, target_direction, 'T',
                                   DIRECTION_COLORS['target'], vector_scale)
        
        # Set axis properties
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        if title is None:
            title = f"Group {group_idx} ({len(group_cube_ids)} cubes)"
        ax.set_title(title, fontsize=10)
        
        # Equal aspect ratio
        max_range = np.max(np.ptp(positions, axis=0)) / 2 + 2
        mid = center
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
    
    def _get_cube_face_colors(self, cube) -> List[str]:
        """Get face colors based on cube's face functions."""
        colors = [color for color in FACE_FUNCTION_COLORS.values()]
        return colors
    
    # Add this method to ConstellationVisualizer class
    def add_face_legend(self) -> None:
        """Add a legend showing face function colors."""
        legend_elements = [
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FFD700', 
                markersize=12, label='Solar Array'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#00CED1', 
                markersize=12, label='Antenna High Gain'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#4169E1', 
                markersize=12, label='Camera'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#FF6347', 
                markersize=12, label='Radiator'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#32CD32', 
                markersize=12, label='Antenna Inter-Sat'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#DE10C6', 
                markersize=12, label='Science Instruments'),
        ]
        self.fig.legend(handles=legend_elements, loc='lower right', 
                        fontsize=8, title='Face Functions', ncol=2)
    
    def render_full(self, env: ConstellationTrainingEnv,
                    step: int = 0,
                    episode_reward: float = 0.0,
                    task_progress: float = 0.0,
                    action_info: str = "") -> None:
        """Render complete visualization with all views."""
        
        constellation = env.constellation
        groups = constellation.get_all_groups()
        num_groups = len(groups)
        
        # Setup figure if needed
        if self.fig is None or len([k for k in self.axes if k.startswith('group_')]) != min(num_groups, 4):
            self.setup_figure(num_groups)
        
        # Render global view
        global_title = (f"Constellation Overview | Step: {step} | "
                       f"Reward: {episode_reward:.2f} | Progress: {task_progress:.1%}")
        self.render_global_view(
            constellation,
            env.sun_direction,
            env.earth_direction,
            env.target_direction,
            title=global_title
        )
        
        # Render each group view
        for i, group in enumerate(groups[:4]):  # Max 4 group views
            self.render_group_view(
                env.swarm,
                group.cube_ids,
                i,
                env.sun_direction,
                env.earth_direction,
                env.target_direction,
                title=f"Group {group.group_id} ({len(group.cube_ids)} cubes)"
            )
        
        self.add_face_legend()

        # Add info text
        if action_info:
            self.fig.suptitle(f"Last Action: {action_info}", fontsize=10, y=0.02)
        
        plt.tight_layout()
    
    def update(self, pause: float = 0.1) -> None:
        """Update the display."""
        plt.draw()
        plt.pause(pause)
    
    def show(self) -> None:
        """Display the visualization."""
        plt.show()
    
    def save(self, filepath: str, dpi: int = 150) -> None:
        """Save the current figure to file."""
        if self.fig is not None:
            self.fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            print(f"Saved visualization to {filepath}")
    
    def close(self) -> None:
        """Close the figure."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.axes = {}


# =============================================================================
# Agent Visualization Runner
# =============================================================================

class AgentVisualizer:
    """
    Runs a trained agent and visualizes its behavior in real-time.
    
    Loads a checkpoint, creates an environment with a specified task,
    and renders the agent's actions step by step.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.agent = self._load_agent(checkpoint_path)
        self.visualizer = ConstellationVisualizer()
        
    def _load_agent(self, checkpoint_path: str) -> ConstellationPPOAgent:
        """Load a trained agent from checkpoint."""
        print(f"Loading agent from {checkpoint_path}...")
        
        # Create agent with default config (will be overwritten by checkpoint)
        config = PPOConfig()
        agent = ConstellationPPOAgent(config, self.device)
        agent.load(checkpoint_path)
        
        print(f"Agent loaded successfully on {self.device}")
        return agent
    
    def create_task_from_curriculum(self, task_key: str, tier: str = 'medium') -> tuple:
        """Create a task instance from the curriculum."""
        curriculum = TaskCurriculum()
        
        tier_map = {'easy': 0, 'medium': 1, 'hard': 2}
        tier_idx = tier_map.get(tier, 1)
        
        # Clamp to valid tier range
        max_tier = curriculum.num_tiers(task_key) - 1
        tier_idx = min(tier_idx, max_tier)
        
        task = curriculum.build_task(task_key, tier_idx)
        num_cubes = curriculum.sample_num_cubes(task_key, tier_idx)
        
        return task, num_cubes

    # Add this method to the AgentVisualizer class
    def plot_episode_metrics(self, action_history: List[Dict], save_path: Optional[str] = None) -> None:
        """Plot reward and progress over time."""
        steps = [a['step'] for a in action_history]
        rewards = [a['reward'] for a in action_history]
        cumulative_reward = np.cumsum(rewards)
        
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        
        # Cumulative reward
        axes[0].plot(steps, cumulative_reward, 'b-', linewidth=2)
        axes[0].set_ylabel('Cumulative Reward')
        axes[0].set_title('Episode Metrics Over Time')
        axes[0].grid(True, alpha=0.3)
        
        # Per-step reward
        axes[1].bar(steps, rewards, color=['green' if r > 0 else 'red' for r in rewards], alpha=0.7)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Step Reward')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved metrics plot to {save_path}")
        plt.show()
    
    def run_episode(self,
                    task_key: str = 'sparse_aperture',
                    tier: str = 'medium',
                    num_cubes: Optional[int] = None,
                    max_steps: int = 500,
                    render_interval: int = 1,
                    pause_time: float = 0.1,
                    deterministic: bool = True,
                    save_video: bool = False,
                    video_path: str = './episode.mp4') -> Dict:
        """
        Run a single episode with visualization.
        
        Args:
            task_key: Task type from curriculum (e.g., 'sparse_aperture', 'earth_downlink')
            tier: Difficulty tier ('easy', 'medium', 'hard')
            num_cubes: Override number of cubes (None = use curriculum default)
            max_steps: Maximum episode steps
            render_interval: Render every N steps
            pause_time: Pause between renders (seconds)
            deterministic: Use deterministic actions
            save_video: Whether to save as video
            video_path: Path for video output
            
        Returns:
            Episode statistics dictionary
        """
        # Create task and environment
        task, default_num_cubes = self.create_task_from_curriculum(task_key, tier)
        if num_cubes is None:
            num_cubes = default_num_cubes
        
        env = ConstellationTrainingEnv(
            num_cubes=num_cubes,
            task=task,
            max_steps=max_steps
        )
        
        print(f"\n{'='*60}")
        print(f"Running Episode: {task_key} ({tier})")
        print(f"{'='*60}")
        print(f"  Num cubes: {num_cubes}")
        print(f"  Max steps: {max_steps}")
        print(f"  Task info: {task.get_task_info()}")
        
        # Reset environment
        graph_data, mode_idx, env_features, action_masks = env.reset()
        
        # Setup visualization
        num_groups = env.constellation.get_num_groups()
        self.visualizer.setup_figure(num_groups)
        
        # Video recording setup
        frames = [] if save_video else None
        
        # Episode loop
        episode_reward = 0.0
        step = 0
        done = False
        
        action_history = []
        
        while not done and step < max_steps:
            # Get action from agent
            action_type, sub_action, log_prob, value, masks = self.agent.get_action(
                env.constellation,
                env.controller,
                env.movement,
                env.mission_mode,
                env.sun_direction,
                env.earth_direction,
                env.target_direction,
                env.sun_distance_au,
                deterministic=deterministic
            )
            
            # Execute action
            (new_graph_data, new_mode_idx, new_env_features,
             new_action_masks, reward, done, info) = env.step(
                action_type, sub_action, masks
            )
            
            episode_reward += reward
            step += 1
            
            # Record action
            action_info = self._format_action(action_type, sub_action, info)
            action_history.append({
                'step': step,
                'action_type': action_type,
                'sub_action': sub_action,
                'reward': reward,
                'success': info.get('action_success', False),
                'reason': info.get('action_reason', '')
            })
            
            # Render visualization
            if step % render_interval == 0:
                self.visualizer.render_full(
                    env,
                    step=step,
                    episode_reward=episode_reward,
                    task_progress=info.get('task_progress', 0.0),
                    action_info=action_info
                )
                self.visualizer.update(pause_time)
                
                # Save frame for video
                if save_video:
                    frames.append(self._capture_frame())
            
            # Print progress
            if step % 50 == 0 or done:
                print(f"  Step {step:4d} | Reward: {episode_reward:8.3f} | "
                      f"Progress: {info.get('task_progress', 0.0):6.2%} | "
                      f"Groups: {info.get('num_groups', 1)} | "
                      f"Baseline: {info.get('max_baseline', 0.0):8.1f}m")
            
            # Update state
            graph_data = new_graph_data
            action_masks = new_action_masks
        
        # Episode summary
        print(f"\n{'='*60}")
        print(f"Episode Complete")
        print(f"{'='*60}")
        print(f"  Total steps: {step}")
        print(f"  Total reward: {episode_reward:.4f}")
        print(f"  Task complete: {info.get('task_complete', False)}")
        print(f"  Final progress: {info.get('task_progress', 0.0):.2%}")
        print(f"  Final groups: {info.get('num_groups', 1)}")
        print(f"  Final baseline: {info.get('max_baseline', 0.0):.1f}m")
        
        # Save video if requested
        if save_video and frames:
            self._save_video(frames, video_path)

        metrics_path = './episode_metrics.png' if save_video else None
        self.plot_episode_metrics(action_history, save_path=metrics_path)
        
        return {
            'steps': step,
            'reward': episode_reward,
            'task_complete': info.get('task_complete', False),
            'task_progress': info.get('task_progress', 0.0),
            'num_groups': info.get('num_groups', 1),
            'max_baseline': info.get('max_baseline', 0.0),
            'action_history': action_history
        }
    
    def _format_action(self, action_type: int, sub_action: int, info: Dict) -> str:
        """Format action information for display."""
        action_names = ['CubeMove', 'Separate', 'Dock', 'Maneuver', 'Noop']
        action_name = action_names[action_type] if action_type < len(action_names) else f'Unknown({action_type})'
        
        success = "✓" if info.get('action_success', False) else "✗"
        reason = info.get('action_reason', '')[:30]
        
        return f"{action_name}[{sub_action}] {success} {reason}"
    
    def _capture_frame(self) -> np.ndarray:
        """Capture current figure as numpy array."""
        self.visualizer.fig.canvas.draw()
        data = np.frombuffer(self.visualizer.fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(self.visualizer.fig.canvas.get_width_height()[::-1] + (3,))
        return data
    
    def _save_video(self, frames: List[np.ndarray], video_path: str, fps: int = 10) -> None:
        """Save frames as video using imageio or matplotlib animation."""
        try:
            import imageio
            print(f"Saving video to {video_path}...")
            imageio.mimsave(video_path, frames, fps=fps)
            print(f"Video saved successfully")
        except ImportError:
            print("imageio not available, saving as GIF instead...")
            gif_path = video_path.replace('.mp4', '.gif')
            try:
                import imageio
                imageio.mimsave(gif_path, frames[::2], fps=fps//2)  # Skip frames for smaller GIF
            except:
                print("Could not save video/GIF. Install imageio: pip install imageio[ffmpeg]")
    
    def run_task_comparison(self,
                            task_keys: List[str] = None,
                            tier: str = 'medium',
                            episodes_per_task: int = 1,
                            max_steps: int = 300) -> Dict:
        """
        Run the agent on multiple tasks for comparison.
        
        Args:
            task_keys: List of task keys to test (None = all tasks)
            tier: Difficulty tier for all tasks
            episodes_per_task: Number of episodes per task
            max_steps: Maximum steps per episode
            
        Returns:
            Dictionary of results per task
        """
        if task_keys is None:
            curriculum = TaskCurriculum()
            task_keys = list(curriculum.registry.keys())
        
        results = {}
        
        for task_key in task_keys:
            print(f"\n{'#'*60}")
            print(f"# Testing Task: {task_key}")
            print(f"{'#'*60}")
            
            task_results = []
            for ep in range(episodes_per_task):
                result = self.run_episode(
                    task_key=task_key,
                    tier=tier,
                    max_steps=max_steps,
                    render_interval=5,
                    pause_time=0.05
                )
                task_results.append(result)
            
            # Aggregate results
            results[task_key] = {
                'mean_reward': np.mean([r['reward'] for r in task_results]),
                'mean_progress': np.mean([r['task_progress'] for r in task_results]),
                'success_rate': np.mean([r['task_complete'] for r in task_results]),
                'episodes': task_results
            }
        
        # Print summary
        print(f"\n{'='*60}")
        print("Task Comparison Summary")
        print(f"{'='*60}")
        print(f"{'Task':<25} {'Reward':>10} {'Progress':>10} {'Success':>10}")
        print("-" * 60)
        for task_key, res in results.items():
            print(f"{task_key:<25} {res['mean_reward']:>10.2f} "
                  f"{res['mean_progress']:>10.2%} {res['success_rate']:>10.2%}")
        
        return results


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained constellation control agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single episode on sparse_aperture task
  python visualize_agent.py --checkpoint ./checkpoints/final_model.pt --task sparse_aperture
  
  # Run on hard difficulty with more cubes
  python visualize_agent.py --checkpoint ./checkpoints/final_model.pt --task form_constellation --tier hard --num-cubes 48
  
  # Compare performance across all tasks
  python visualize_agent.py --checkpoint ./checkpoints/final_model.pt --compare-all
  
  # Save episode as video
  python visualize_agent.py --checkpoint ./checkpoints/final_model.pt --task sparse_aperture --save-video
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--task', type=str, default='sparse_aperture',
                        choices=['earth_downlink', 'local_relay', 'sparse_aperture',
                                 'occultation_array', 'in_situ_field', 'cruise_mode',
                                 'solar_collection', 'thermal_shield', 'damaged_reconfig',
                                 'form_constellation'],
                        help='Task type to visualize')
    parser.add_argument('--tier', type=str, default='medium',
                        choices=['easy', 'medium', 'hard'],
                        help='Difficulty tier')
    parser.add_argument('--num-cubes', type=int, default=None,
                        help='Number of cubes (overrides curriculum default)')
    parser.add_argument('--max-steps', type=int, default=50,
                        help='Maximum episode steps')
    parser.add_argument('--render-interval', type=int, default=1,
                        help='Render every N steps')
    parser.add_argument('--pause', type=float, default=0.1,
                        help='Pause time between renders (seconds)')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions (default: deterministic)')
    parser.add_argument('--save-video', action='store_true',
                        help='Save episode as video')
    parser.add_argument('--video-path', type=str, default='./episode.mp4',
                        help='Path for video output')
    parser.add_argument('--compare-all', action='store_true',
                        help='Run comparison across all tasks')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Create visualizer
    viz = AgentVisualizer(args.checkpoint, device=args.device)
    
    if args.compare_all:
        # Run comparison across all tasks
        viz.run_task_comparison(
            task_keys=None,
            tier=args.tier,
            episodes_per_task=1,
            max_steps=args.max_steps
        )
    else:
        # Run single episode
        viz.run_episode(
            task_key=args.task,
            tier=args.tier,
            num_cubes=args.num_cubes,
            max_steps=args.max_steps,
            render_interval=args.render_interval,
            pause_time=args.pause,
            deterministic=not args.stochastic,
            save_video=args.save_video,
            video_path=args.video_path
        )
    
    # Keep visualization open
    print("\nClose the visualization window to exit...")
    viz.visualizer.show()


if __name__ == "__main__":
    main()