"""
episode_recorder.py
===================
Records episodes and creates animations of agent behavior.

Uses the ConstellationVisualizer to render frames and creates
MP4/GIF animations of the best performing episodes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import json

from visualization.constellation_renderer import ConstellationVisualizer
from core.constellation import Constellation


@dataclass
class EpisodeFrame:
    """Single frame of an episode recording."""
    step: int
    action_type: int
    sub_action: int
    reward: float
    cumulative_reward: float
    task_progress: float
    num_groups: int
    max_baseline: float
    delta_v_remaining: float
    action_success: bool
    action_reason: str
    
    # Constellation state for rendering
    group_positions: List[np.ndarray] = field(default_factory=list)
    group_velocities: List[np.ndarray] = field(default_factory=list)
    group_sizes: List[int] = field(default_factory=list)
    cube_positions: List[Tuple[int, int, int]] = field(default_factory=list)
    cube_group_assignments: List[int] = field(default_factory=list)


@dataclass
class EpisodeRecording:
    """Complete recording of an episode."""
    episode_id: int
    total_reward: float
    total_steps: int
    task_complete: bool
    frames: List[EpisodeFrame] = field(default_factory=list)
    
    # Metadata
    task_type: str = ""
    mission_mode: int = 0
    sun_direction: Tuple[float, float, float] = (0, 0, -1)
    earth_direction: Tuple[float, float, float] = (1, 0, 0)


class EpisodeRecorder:
    """
    Records episodes for visualization and analysis.
    
    Captures constellation state at each step and can generate
    animations showing agent behavior.
    """
    
    def __init__(self, 
                 save_dir: str,
                 max_recordings: int = 10,
                 record_frequency: int = 100):
        """
        Args:
            save_dir: Directory to save recordings and animations
            max_recordings: Maximum number of recordings to keep
            record_frequency: Record every N episodes (0 = only record best)
        """
        self.save_dir = save_dir
        self.max_recordings = max_recordings
        self.record_frequency = record_frequency
        
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'animations'), exist_ok=True)
        
        self.recordings: List[EpisodeRecording] = []
        self.best_recording: Optional[EpisodeRecording] = None
        self.best_reward: float = -float('inf')
        
        self.current_recording: Optional[EpisodeRecording] = None
        self.episode_count: int = 0
    
    def start_episode(self, episode_id: int, task_info: Dict) -> None:
        """Start recording a new episode."""
        self.current_recording = EpisodeRecording(
            episode_id=episode_id,
            total_reward=0.0,
            total_steps=0,
            task_complete=False,
            task_type=task_info.get('task_type', 'unknown'),
        )
    
    def record_step(self,
                    constellation: Constellation,
                    action_type: int,
                    sub_action: int,
                    reward: float,
                    info: Dict) -> None:
        """Record a single step."""
        if self.current_recording is None:
            return
        
        self.current_recording.total_reward += reward
        self.current_recording.total_steps += 1
        
        # Capture constellation state
        groups = constellation.get_all_groups()
        
        frame = EpisodeFrame(
            step=self.current_recording.total_steps,
            action_type=action_type,
            sub_action=sub_action,
            reward=reward,
            cumulative_reward=self.current_recording.total_reward,
            task_progress=info.get('task_progress', 0.0),
            num_groups=len(groups),
            max_baseline=constellation.get_max_baseline(),
            delta_v_remaining=constellation.get_total_delta_v_remaining(),
            action_success=info.get('action_success', True),
            action_reason=info.get('action_reason', ''),
            group_positions=[g.position.copy() for g in groups],
            group_velocities=[g.velocity.copy() for g in groups],
            group_sizes=[len(g.cube_ids) for g in groups],
        )
        
        # Capture cube positions
        for group in groups:
            for cube_id in group.cube_ids:
                cube = constellation.swarm.get_cube(cube_id)
                if cube:
                    frame.cube_positions.append(cube.position)
                    frame.cube_group_assignments.append(group.group_id)
        
        self.current_recording.frames.append(frame)
    
    def end_episode(self, task_complete: bool) -> None:
        """End the current episode recording."""
        if self.current_recording is None:
            return
        
        self.current_recording.task_complete = task_complete
        self.episode_count += 1
        
        # Check if this is the best episode
        if self.current_recording.total_reward > self.best_reward:
            self.best_reward = self.current_recording.total_reward
            self.best_recording = self.current_recording
            print(f"  New best episode! Reward: {self.best_reward:.4f}")
        
        # Decide whether to keep this recording
        keep_recording = False
        
        if self.record_frequency > 0 and self.episode_count % self.record_frequency == 0:
            keep_recording = True
        
        if self.current_recording.total_reward == self.best_reward:
            keep_recording = True
        
        if keep_recording:
            self.recordings.append(self.current_recording)
            
            # Trim to max recordings
            if len(self.recordings) > self.max_recordings:
                # Keep best and most recent
                self.recordings.sort(key=lambda r: r.total_reward, reverse=True)
                self.recordings = self.recordings[:self.max_recordings]
        
        self.current_recording = None
    
    def create_animation(self, 
                         recording: EpisodeRecording,
                         filename: str,
                         fps: int = 10,
                         dpi: int = 100) -> str:
        """
        Create an animation from a recording.
        
        Args:
            recording: The episode recording to animate
            filename: Output filename (without extension)
            fps: Frames per second
            dpi: Resolution
            
        Returns:
            Path to saved animation
        """
        if not recording.frames:
            print("No frames to animate")
            return ""
        
        # Create figure with multiple panels
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main 3D view (local)
        ax_local = fig.add_subplot(gs[0, 0], projection='3d')
        
        # Global view
        ax_global = fig.add_subplot(gs[0, 1], projection='3d')
        
        # Metrics panel
        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.axis('off')
        
        # Reward curve
        ax_reward = fig.add_subplot(gs[1, 0])
        
        # Progress curve
        ax_progress = fig.add_subplot(gs[1, 1])
        
        # Groups/Baseline curve
        ax_groups = fig.add_subplot(gs[1, 2])
        
        # Group colors
        group_colors = [
            'steelblue', 'coral', 'seagreen', 'orchid',
            'goldenrod', 'turquoise', 'salmon', 'mediumpurple'
        ]
        
        # Precompute data for plots
        steps = [f.step for f in recording.frames]
        rewards = [f.cumulative_reward for f in recording.frames]
        progress = [f.task_progress for f in recording.frames]
        num_groups = [f.num_groups for f in recording.frames]
        baselines = [f.max_baseline for f in recording.frames]
        
        def update(frame_idx):
            """Update function for animation."""
            frame = recording.frames[frame_idx]
            
            # Clear axes
            ax_local.clear()
            ax_global.clear()
            ax_metrics.clear()
            ax_metrics.axis('off')
            
            # --- Local view (cube positions) ---
            if frame.cube_positions:
                # Group cubes by their group assignment
                group_cubes = {}
                for i, (pos, group_id) in enumerate(zip(frame.cube_positions, 
                                                         frame.cube_group_assignments)):
                    if group_id not in group_cubes:
                        group_cubes[group_id] = []
                    group_cubes[group_id].append(pos)
                
                # Plot each group with different color
                for g_idx, (group_id, positions) in enumerate(group_cubes.items()):
                    color = group_colors[g_idx % len(group_colors)]
                    xs = [p[0] for p in positions]
                    ys = [p[1] for p in positions]
                    zs = [p[2] for p in positions]
                    ax_local.scatter(xs, ys, zs, c=color, s=50, alpha=0.8,
                                    label=f'Group {group_id}')
            
            ax_local.set_xlabel('X')
            ax_local.set_ylabel('Y')
            ax_local.set_zlabel('Z')
            ax_local.set_title('Cube Positions (Local)', fontsize=10)
            if frame.cube_positions:
                ax_local.legend(loc='upper left', fontsize=8)
            
            # --- Global view (group positions) ---
            if frame.group_positions:
                for g_idx, (pos, vel, size) in enumerate(zip(frame.group_positions,
                                                              frame.group_velocities,
                                                              frame.group_sizes)):
                    color = group_colors[g_idx % len(group_colors)]
                    marker_size = 100 + size * 5
                    
                    ax_global.scatter(pos[0], pos[1], pos[2], 
                                     c=color, s=marker_size, alpha=0.8,
                                     label=f'G{g_idx} ({size} cubes)')
                    
                    # Velocity vector
                    vel_scale = 50
                    if np.linalg.norm(vel) > 0.01:
                        ax_global.quiver(pos[0], pos[1], pos[2],
                                        vel[0]*vel_scale, vel[1]*vel_scale, vel[2]*vel_scale,
                                        color=color, alpha=0.5, arrow_length_ratio=0.1)
            
            ax_global.set_xlabel('X (m)')
            ax_global.set_ylabel('Y (m)')
            ax_global.set_zlabel('Z (m)')
            ax_global.set_title('Group Positions (Global)', fontsize=10)
            if frame.group_positions:
                ax_global.legend(loc='upper left', fontsize=8)
            
            # --- Metrics panel ---
            action_names = ['Cube Move', 'Separation', 'Docking', 'Maneuver', 'No-op']
            action_name = action_names[frame.action_type] if frame.action_type < len(action_names) else 'Unknown'
            
            metrics_text = (
                f"{'='*30}\n"
                f"Step: {frame.step}\n"
                f"{'='*30}\n\n"
                f"Action: {action_name}\n"
                f"Success: {'✓' if frame.action_success else '✗'}\n"
                f"Reason: {frame.action_reason[:30]}\n\n"
                f"Reward: {frame.reward:+.4f}\n"
                f"Cumulative: {frame.cumulative_reward:.4f}\n\n"
                f"Progress: {frame.task_progress:.2%}\n"
                f"Groups: {frame.num_groups}\n"
                f"Baseline: {frame.max_baseline:.1f}m\n"
                f"Δv Left: {frame.delta_v_remaining:.1f} m/s\n"
            )
            
            ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes,
                           fontsize=9, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            ax_metrics.set_title('Current State', fontsize=10)
            
            # --- Reward curve ---
            ax_reward.clear()
            ax_reward.plot(steps[:frame_idx+1], rewards[:frame_idx+1], 
                          'b-', linewidth=2)
            ax_reward.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax_reward.set_xlabel('Step')
            ax_reward.set_ylabel('Cumulative Reward')
            ax_reward.set_title('Reward Over Time', fontsize=10)
            ax_reward.set_xlim(0, max(steps))
            if rewards:
                ax_reward.set_ylim(min(min(rewards), 0) - 0.1, max(max(rewards), 0) + 0.1)
            ax_reward.grid(True, alpha=0.3)
            
            # --- Progress curve ---
            ax_progress.clear()
            ax_progress.fill_between(steps[:frame_idx+1], progress[:frame_idx+1],
                                    alpha=0.3, color='green')
            ax_progress.plot(steps[:frame_idx+1], progress[:frame_idx+1],
                           'g-', linewidth=2)
            ax_progress.set_xlabel('Step')
            ax_progress.set_ylabel('Task Progress')
            ax_progress.set_title('Task Progress', fontsize=10)
            ax_progress.set_xlim(0, max(steps))
            ax_progress.set_ylim(0, 1.05)
            ax_progress.grid(True, alpha=0.3)
            
            # --- Groups and baseline ---
            ax_groups.clear()
            ax_groups_twin = ax_groups.twinx()
            
            line1, = ax_groups.plot(steps[:frame_idx+1], num_groups[:frame_idx+1],
                                   'b-', linewidth=2, label='Groups')
            line2, = ax_groups_twin.plot(steps[:frame_idx+1], baselines[:frame_idx+1],
                                        'r-', linewidth=2, label='Baseline')
            
            ax_groups.set_xlabel('Step')
            ax_groups.set_ylabel('Number of Groups', color='blue')
            ax_groups_twin.set_ylabel('Max Baseline (m)', color='red')
            ax_groups.set_title('Groups & Baseline', fontsize=10)
            ax_groups.set_xlim(0, max(steps))
            ax_groups.set_ylim(0, max(max(num_groups), 1) + 1)
            ax_groups.tick_params(axis='y', labelcolor='blue')
            ax_groups_twin.tick_params(axis='y', labelcolor='red')
            ax_groups.grid(True, alpha=0.3)
            
            # Add legend
            lines = [line1, line2]
            labels = [l.get_label() for l in lines]
            ax_groups.legend(lines, labels, loc='upper left', fontsize=8)
            
            fig.suptitle(f"Episode {recording.episode_id} - "
                        f"{'Complete!' if recording.task_complete else 'In Progress'} - "
                        f"Total Reward: {recording.total_reward:.2f}",
                        fontsize=12, fontweight='bold')
            
            return []
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, update, frames=len(recording.frames),
            interval=1000//fps, blit=False
        )
        
        # Save animation
        output_path = os.path.join(self.save_dir, 'animations', f'{filename}.mp4')
        
        try:
            writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
            anim.save(output_path, writer=writer, dpi=dpi)
            print(f"  Saved animation: {output_path}")
        except Exception as e:
            print(f"  Could not save MP4 (ffmpeg not found?): {e}")
            # Try saving as GIF instead
            try:
                gif_path = os.path.join(self.save_dir, 'animations', f'{filename}.gif')
                writer = animation.PillowWriter(fps=fps)
                anim.save(gif_path, writer=writer, dpi=dpi//2)
                output_path = gif_path
                print(f"  Saved GIF instead: {gif_path}")
            except Exception as e2:
                print(f"  Could not save animation: {e2}")
                output_path = ""
        
        plt.close(fig)
        return output_path
    
    def create_best_episode_animation(self, filename: str = "best_episode") -> str:
        """Create animation of the best episode."""
        if self.best_recording is None:
            print("No best episode recorded yet")
            return ""
        
        print(f"\nCreating animation for best episode (reward: {self.best_reward:.4f})...")
        return self.create_animation(self.best_recording, filename)
    
    def save_recording_data(self, recording: EpisodeRecording, filename: str) -> None:
        """Save recording data to JSON for later analysis."""
        data = {
            'episode_id': recording.episode_id,
            'total_reward': recording.total_reward,
            'total_steps': recording.total_steps,
            'task_complete': recording.task_complete,
            'task_type': recording.task_type,
            'frames': [
                {
                    'step': f.step,
                    'action_type': f.action_type,
                    'sub_action': f.sub_action,
                    'reward': f.reward,
                    'cumulative_reward': f.cumulative_reward,
                    'task_progress': f.task_progress,
                    'num_groups': f.num_groups,
                    'max_baseline': f.max_baseline,
                    'delta_v_remaining': f.delta_v_remaining,
                    'action_success': f.action_success,
                    'action_reason': f.action_reason,
                }
                for f in recording.frames
            ]
        }
        
        path = os.path.join(self.save_dir, f'{filename}.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)