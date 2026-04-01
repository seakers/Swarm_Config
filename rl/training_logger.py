"""
training_logger.py
==================
Logging and visualization for training metrics in academic RL evaluation.

Tracks and plots:
- Episode rewards (mean, min, max, std)
- Episode lengths
- Policy/value losses
- Entropy
- Learning rate
- Task-specific metrics (progress, completion rate)
- Action distribution statistics
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass, field
import time


@dataclass
class TrainingMetrics:
    """Container for all training metrics."""
    # Episode metrics
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    episode_successes: List[bool] = field(default_factory=list)
    
    # Per-update metrics
    policy_losses: List[float] = field(default_factory=list)
    value_losses: List[float] = field(default_factory=list)
    entropies: List[float] = field(default_factory=list)
    approx_kls: List[float] = field(default_factory=list)
    clip_fractions: List[float] = field(default_factory=list)
    
    # Learning rate
    learning_rates: List[float] = field(default_factory=list)
    
    # Timesteps
    timesteps: List[int] = field(default_factory=list)
    update_timesteps: List[int] = field(default_factory=list)
    
    # Task-specific
    task_progress: List[float] = field(default_factory=list)
    num_groups: List[int] = field(default_factory=list)
    max_baselines: List[float] = field(default_factory=list)
    delta_v_used: List[float] = field(default_factory=list)
    
    # Timing
    fps: List[float] = field(default_factory=list)
    
    # Best episode tracking
    best_reward: float = -float('inf')
    best_episode_idx: int = -1


class TrainingLogger:
    """
    Comprehensive logger for deep RL training.
    
    Provides:
    - Real-time metric tracking
    - Periodic plot generation
    - JSON export for reproducibility
    - Rolling statistics computation
    """
    
    def __init__(self, 
                 log_dir: str,
                 experiment_name: str = "constellation_ppo",
                 window_size: int = 100,
                 save_frequency: int = 10):
        """
        Args:
            log_dir: Directory to save logs and plots
            experiment_name: Name for this experiment
            window_size: Window for rolling statistics
            save_frequency: Save plots every N updates
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.window_size = window_size
        self.save_frequency = save_frequency
        
        # Create directories
        self.plots_dir = os.path.join(log_dir, "plots")
        self.data_dir = os.path.join(log_dir, "data")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        
        # Rolling windows for statistics
        self.reward_window = deque(maxlen=window_size)
        self.length_window = deque(maxlen=window_size)
        self.success_window = deque(maxlen=window_size)
        
        # Timing
        self.start_time = time.time()
        self.last_log_time = self.start_time
        
        # Update counter
        self.num_updates = 0
        self.total_timesteps = 0
    
    def log_episode(self, 
                    reward: float, 
                    length: int, 
                    success: bool,
                    task_progress: float = 0.0,
                    num_groups: int = 1,
                    max_baseline: float = 0.0,
                    delta_v_used: float = 0.0) -> None:
        """Log metrics for a completed episode."""
        self.metrics.episode_rewards.append(reward)
        self.metrics.episode_lengths.append(length)
        self.metrics.episode_successes.append(success)
        self.metrics.task_progress.append(task_progress)
        self.metrics.num_groups.append(num_groups)
        self.metrics.max_baselines.append(max_baseline)
        self.metrics.delta_v_used.append(delta_v_used)
        self.metrics.timesteps.append(self.total_timesteps)
        
        # Update rolling windows
        self.reward_window.append(reward)
        self.length_window.append(length)
        self.success_window.append(success)
        
        # Track best episode
        if reward > self.metrics.best_reward:
            self.metrics.best_reward = reward
            self.metrics.best_episode_idx = len(self.metrics.episode_rewards) - 1
    
    def log_update(self,
                   policy_loss: float,
                   value_loss: float,
                   entropy: float,
                   approx_kl: float,
                   clip_fraction: float,
                   learning_rate: float,
                   timesteps: int) -> None:
        """Log metrics for a policy update."""
        self.metrics.policy_losses.append(policy_loss)
        self.metrics.value_losses.append(value_loss)
        self.metrics.entropies.append(entropy)
        self.metrics.approx_kls.append(approx_kl)
        self.metrics.clip_fractions.append(clip_fraction)
        self.metrics.learning_rates.append(learning_rate)
        self.metrics.update_timesteps.append(timesteps)
        
        self.total_timesteps = timesteps
        self.num_updates += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_log_time
        if elapsed > 0:
            fps = (timesteps - (self.metrics.update_timesteps[-2] if len(self.metrics.update_timesteps) > 1 else 0)) / elapsed
            self.metrics.fps.append(fps)
        self.last_log_time = current_time
        
        # Periodic saving
        if self.num_updates % self.save_frequency == 0:
            self.save_plots()
            self.save_data()
    
    def get_rolling_stats(self) -> Dict[str, float]:
        """Get rolling statistics for recent episodes."""
        stats = {}
        
        if self.reward_window:
            rewards = list(self.reward_window)
            stats['mean_reward'] = np.mean(rewards)
            stats['std_reward'] = np.std(rewards)
            stats['min_reward'] = np.min(rewards)
            stats['max_reward'] = np.max(rewards)
        
        if self.length_window:
            stats['mean_length'] = np.mean(list(self.length_window))
        
        if self.success_window:
            stats['success_rate'] = np.mean(list(self.success_window))
        
        return stats
    
    def save_plots(self) -> None:
        """Generate and save all training plots."""
        self._plot_training_curves()
        self._plot_loss_curves()
        self._plot_task_metrics()
        self._plot_summary_dashboard()
    
    def _plot_training_curves(self) -> None:
        """Plot episode reward and length curves."""
        if len(self.metrics.episode_rewards) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{self.experiment_name} - Training Curves', fontsize=14, fontweight='bold')
        
        episodes = range(len(self.metrics.episode_rewards))
        
        # Episode Rewards
        ax = axes[0, 0]
        ax.plot(episodes, self.metrics.episode_rewards, alpha=0.3, color='blue', label='Episode')
        
        # Rolling mean
        if len(self.metrics.episode_rewards) >= self.window_size:
            rolling_mean = np.convolve(
                self.metrics.episode_rewards, 
                np.ones(self.window_size)/self.window_size, 
                mode='valid'
            )
            ax.plot(range(self.window_size-1, len(self.metrics.episode_rewards)), 
                   rolling_mean, color='red', linewidth=2, label=f'Rolling Mean ({self.window_size})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Episode Lengths
        ax = axes[0, 1]
        ax.plot(episodes, self.metrics.episode_lengths, alpha=0.3, color='green')
        
        if len(self.metrics.episode_lengths) >= self.window_size:
            rolling_mean = np.convolve(
                self.metrics.episode_lengths,
                np.ones(self.window_size)/self.window_size,
                mode='valid'
            )
            ax.plot(range(self.window_size-1, len(self.metrics.episode_lengths)),
                   rolling_mean, color='darkgreen', linewidth=2)
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.grid(True, alpha=0.3)
        
        # Success Rate (rolling)
        ax = axes[1, 0]
        if self.metrics.episode_successes:
            successes = [float(s) for s in self.metrics.episode_successes]
            if len(successes) >= self.window_size:
                rolling_success = np.convolve(
                    successes,
                    np.ones(self.window_size)/self.window_size,
                    mode='valid'
                )
                ax.plot(range(self.window_size-1, len(successes)),
                       rolling_success, color='purple', linewidth=2)
                ax.fill_between(range(self.window_size-1, len(successes)),
                               rolling_success, alpha=0.3, color='purple')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title(f'Rolling Success Rate ({self.window_size} episodes)')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Reward Distribution
        ax = axes[1, 1]
        ax.hist(self.metrics.episode_rewards, bins=50, color='steelblue', 
                edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(self.metrics.episode_rewards), color='red', 
                   linestyle='--', linewidth=2, label=f'Mean: {np.mean(self.metrics.episode_rewards):.2f}')
        ax.axvline(self.metrics.best_reward, color='gold',
                   linestyle='--', linewidth=2, label=f'Best: {self.metrics.best_reward:.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_loss_curves(self) -> None:
        """Plot policy and value loss curves."""
        if len(self.metrics.policy_losses) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle(f'{self.experiment_name} - Loss Curves', fontsize=14, fontweight='bold')
        
        updates = range(len(self.metrics.policy_losses))
        
        # Policy Loss
        ax = axes[0, 0]
        ax.plot(updates, self.metrics.policy_losses, color='blue')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
        
        # Value Loss
        ax = axes[0, 1]
        ax.plot(updates, self.metrics.value_losses, color='orange')
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
        
        # Entropy
        ax = axes[0, 2]
        ax.plot(updates, self.metrics.entropies, color='green')
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
        
        # KL Divergence
        ax = axes[1, 0]
        ax.plot(updates, self.metrics.approx_kls, color='red')
        ax.axhline(0.01, color='gray', linestyle='--', alpha=0.5, label='Target KL')
        ax.set_xlabel('Update')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Approximate KL')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Clip Fraction
        ax = axes[1, 1]
        ax.plot(updates, self.metrics.clip_fractions, color='purple')
        ax.set_xlabel('Update')
        ax.set_ylabel('Fraction')
        ax.set_title('Clip Fraction')
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 2]
        ax.plot(updates, self.metrics.learning_rates, color='brown')
        ax.set_xlabel('Update')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'loss_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_task_metrics(self) -> None:
        """Plot task-specific metrics."""
        if len(self.metrics.task_progress) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{self.experiment_name} - Task Metrics', fontsize=14, fontweight='bold')
        
        episodes = range(len(self.metrics.task_progress))
        
        # Task Progress
        ax = axes[0, 0]
        ax.plot(episodes, self.metrics.task_progress, alpha=0.3, color='teal')
        if len(self.metrics.task_progress) >= self.window_size:
            rolling = np.convolve(
                self.metrics.task_progress,
                np.ones(self.window_size)/self.window_size,
                mode='valid'
            )
            ax.plot(range(self.window_size-1, len(self.metrics.task_progress)),
                   rolling, color='darkcyan', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Progress')
        ax.set_title('Task Progress')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
        
        # Number of Groups
        ax = axes[0, 1]
        ax.plot(episodes, self.metrics.num_groups, alpha=0.5, color='coral')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Groups')
        ax.set_title('Number of Groups at Episode End')
        ax.grid(True, alpha=0.3)
        
        # Max Baseline
        ax = axes[1, 0]
        ax.plot(episodes, self.metrics.max_baselines, alpha=0.3, color='navy')
        if len(self.metrics.max_baselines) >= self.window_size:
            rolling = np.convolve(
                self.metrics.max_baselines,
                np.ones(self.window_size)/self.window_size,
                mode='valid'
            )
            ax.plot(range(self.window_size-1, len(self.metrics.max_baselines)),
                   rolling, color='blue', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Baseline (m)')
        ax.set_title('Maximum Baseline')
        ax.grid(True, alpha=0.3)
        
        # Delta-V Usage
        ax = axes[1, 1]
        ax.plot(episodes, self.metrics.delta_v_used, alpha=0.3, color='darkred')
        if len(self.metrics.delta_v_used) >= self.window_size:
            rolling = np.convolve(
                self.metrics.delta_v_used,
                np.ones(self.window_size)/self.window_size,
                mode='valid'
            )
            ax.plot(range(self.window_size-1, len(self.metrics.delta_v_used)),
                   rolling, color='red', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delta-V (m/s)')
        ax.set_title('Total Delta-V Used per Episode')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'task_metrics.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_summary_dashboard(self) -> None:
        """Create a comprehensive summary dashboard."""
        if len(self.metrics.episode_rewards) < 10:
            return
        
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'{self.experiment_name} - Training Summary Dashboard\n'
                    f'Total Timesteps: {self.total_timesteps:,} | '
                    f'Episodes: {len(self.metrics.episode_rewards)} | '
                    f'Updates: {self.num_updates}',
                    fontsize=14, fontweight='bold')
        
        # Main reward curve (large)
        ax_main = fig.add_subplot(gs[0, :2])
        episodes = range(len(self.metrics.episode_rewards))
        ax_main.fill_between(episodes, self.metrics.episode_rewards, alpha=0.3, color='blue')
        ax_main.plot(episodes, self.metrics.episode_rewards, alpha=0.5, color='blue', linewidth=0.5)
        
        if len(self.metrics.episode_rewards) >= self.window_size:
            rolling = np.convolve(
                self.metrics.episode_rewards,
                np.ones(self.window_size)/self.window_size,
                mode='valid'
            )
            ax_main.plot(range(self.window_size-1, len(self.metrics.episode_rewards)),
                        rolling, color='red', linewidth=2, label=f'Rolling Mean ({self.window_size})')
        
        ax_main.axhline(self.metrics.best_reward, color='gold', linestyle='--', 
                       label=f'Best: {self.metrics.best_reward:.2f}')
        ax_main.set_xlabel('Episode')
        ax_main.set_ylabel('Reward')
        ax_main.set_title('Episode Rewards')
        ax_main.legend(loc='lower right')
        ax_main.grid(True, alpha=0.3)
        
        # Loss curves (top right)
        ax_loss = fig.add_subplot(gs[0, 2:])
        if self.metrics.policy_losses:
            updates = range(len(self.metrics.policy_losses))
            ax_loss.plot(updates, self.metrics.policy_losses, label='Policy', alpha=0.8)
            ax_loss.plot(updates, self.metrics.value_losses, label='Value', alpha=0.8)
        ax_loss.set_xlabel('Update')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training Losses')
        ax_loss.legend()
        ax_loss.grid(True, alpha=0.3)
        
        # Task progress (middle left)
        ax_progress = fig.add_subplot(gs[1, :2])
        if self.metrics.task_progress:
            ax_progress.fill_between(range(len(self.metrics.task_progress)), 
                                    self.metrics.task_progress, alpha=0.3, color='teal')
            if len(self.metrics.task_progress) >= self.window_size:
                rolling = np.convolve(
                    self.metrics.task_progress,
                    np.ones(self.window_size)/self.window_size,
                    mode='valid'
                )
                ax_progress.plot(range(self.window_size-1, len(self.metrics.task_progress)),
                               rolling, color='darkcyan', linewidth=2)
        ax_progress.set_xlabel('Episode')
        ax_progress.set_ylabel('Progress')
        ax_progress.set_title('Task Progress')
        ax_progress.set_ylim([0, 1.05])
        ax_progress.grid(True, alpha=0.3)
        
        # Entropy and KL (middle right)
        ax_ent = fig.add_subplot(gs[1, 2])
        if self.metrics.entropies:
            ax_ent.plot(self.metrics.entropies, color='green')
        ax_ent.set_xlabel('Update')
        ax_ent.set_ylabel('Entropy')
        ax_ent.set_title('Policy Entropy')
        ax_ent.grid(True, alpha=0.3)
        
        ax_kl = fig.add_subplot(gs[1, 3])
        if self.metrics.approx_kls:
            ax_kl.plot(self.metrics.approx_kls, color='red')
            ax_kl.axhline(0.01, color='gray', linestyle='--', alpha=0.5)
        ax_kl.set_xlabel('Update')
        ax_kl.set_ylabel('KL')
        ax_kl.set_title('Approx KL Divergence')
        ax_kl.grid(True, alpha=0.3)
        
        # Statistics table (bottom left)
        ax_stats = fig.add_subplot(gs[2, :2])
        ax_stats.axis('off')
        
        stats = self.get_rolling_stats()
        stats_text = (
            f"{'=' * 40}\n"
            f"TRAINING STATISTICS (last {self.window_size} episodes)\n"
            f"{'=' * 40}\n\n"
            f"Mean Reward:     {stats.get('mean_reward', 0):.4f}\n"
            f"Std Reward:      {stats.get('std_reward', 0):.4f}\n"
            f"Min Reward:      {stats.get('min_reward', 0):.4f}\n"
            f"Max Reward:      {stats.get('max_reward', 0):.4f}\n"
            f"Success Rate:    {stats.get('success_rate', 0):.2%}\n"
            f"Mean Length:     {stats.get('mean_length', 0):.1f}\n\n"
            f"Best Episode:    #{self.metrics.best_episode_idx}\n"
            f"Best Reward:     {self.metrics.best_reward:.4f}\n"
            f"Total Episodes:  {len(self.metrics.episode_rewards)}\n"
            f"Total Updates:   {self.num_updates}\n"
        )
        
        ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # FPS and timing (bottom right)
        ax_fps = fig.add_subplot(gs[2, 2:])
        if self.metrics.fps:
            ax_fps.plot(self.metrics.fps, color='purple', alpha=0.7)
            if len(self.metrics.fps) >= 10:
                rolling_fps = np.convolve(self.metrics.fps, np.ones(10)/10, mode='valid')
                ax_fps.plot(range(9, len(self.metrics.fps)), rolling_fps, 
                           color='darkviolet', linewidth=2)
        ax_fps.set_xlabel('Update')
        ax_fps.set_ylabel('FPS')
        ax_fps.set_title('Training Speed (Frames per Second)')
        ax_fps.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'summary_dashboard.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_data(self) -> None:
        """Save all metrics to JSON."""
        data = {
            'experiment_name': self.experiment_name,
            'total_timesteps': self.total_timesteps,
            'num_updates': self.num_updates,
            'episode_rewards': self.metrics.episode_rewards,
            'episode_lengths': self.metrics.episode_lengths,
            'episode_successes': [bool(s) for s in self.metrics.episode_successes],
            'policy_losses': self.metrics.policy_losses,
            'value_losses': self.metrics.value_losses,
            'entropies': self.metrics.entropies,
            'approx_kls': self.metrics.approx_kls,
            'clip_fractions': self.metrics.clip_fractions,
            'learning_rates': self.metrics.learning_rates,
            'task_progress': self.metrics.task_progress,
            'num_groups': self.metrics.num_groups,
            'max_baselines': self.metrics.max_baselines,
            'delta_v_used': self.metrics.delta_v_used,
            'fps': self.metrics.fps,
            'best_reward': self.metrics.best_reward,
            'best_episode_idx': self.metrics.best_episode_idx,
        }
        
        path = os.path.join(self.data_dir, 'training_metrics.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def print_summary(self) -> None:
        """Print training summary to console."""
        stats = self.get_rolling_stats()
        
        elapsed = time.time() - self.start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        
        print(f"\n{'=' * 60}")
        print(f"TRAINING SUMMARY - {self.experiment_name}")
        print(f"{'=' * 60}")
        print(f"Duration: {hours}h {minutes}m")
        print(f"Timesteps: {self.total_timesteps:,}")
        print(f"Episodes: {len(self.metrics.episode_rewards)}")
        print(f"Updates: {self.num_updates}")
        print(f"\nLast {self.window_size} episodes:")
        print(f"  Mean Reward: {stats.get('mean_reward', 0):.4f}")
        print(f"  Success Rate: {stats.get('success_rate', 0):.2%}")
        print(f"\nBest Performance:")
        print(f"  Episode: #{self.metrics.best_episode_idx}")
        print(f"  Reward: {self.metrics.best_reward:.4f}")
        print(f"{'=' * 60}\n")