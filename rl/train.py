"""
train.py
========
Training script for the constellation control PPO agent.
"""

import os
import time
import argparse
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque

from rl.ppo_agent import ConstellationPPOAgent, PPOConfig, RolloutBuffer
from rl.observation_builder import (
    ConstellationObservationBuilder,
    ActionMaskBuilder,
    ObservationConfig
)
from rl.training_logger import TrainingLogger
from rl.episode_recorder import EpisodeRecorder

from core.swarm import Swarm
from core.constellation import (
    Constellation, SeparationRequirements, DockingRequirements, CommunicationRequirements
)
from configs.formations import create_cube_formation
from mechanics.moves import MovementSystem
from mechanics.constellation_moves import ConstellationController
from tasks.constellation_tasks import (
    ConstellationTask, FormConstellationTask, MultiPointSensingTask
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    num_cubes: int = 64
    max_episode_steps: int = 500
    time_step: float = 10.0
    
    # Training
    total_timesteps: int = 1_000_000
    rollout_steps: int = 2048
    num_envs: int = 8  # Parallel environments
    
    # Logging
    log_interval: int = 10
    save_interval: int = 50_000
    eval_interval: int = 10_000
    plot_interval: int = 10  # Save plots every N updates
    record_best: bool = True  # Record best episodes for animation
    
    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Task
    task_type: str = "form_constellation"


class ConstellationTrainingEnv:
    """
    Training environment wrapper for constellation control.
    
    Handles the full environment loop including observation building,
    action decoding, and reward computation.
    """
    
    def __init__(self,
                 num_cubes: int = 64,
                 task: Optional[ConstellationTask] = None,
                 max_steps: int = 500,
                 time_step: float = 10.0):
        self.num_cubes = num_cubes
        self.task = task or FormConstellationTask(
            target_num_groups=2,
            target_baseline=5000.0
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
        
        # Mission context (could be randomized)
        self.sun_direction = (0.0, 0.0, -1.0)
        self.earth_direction = (1.0, 0.0, 0.0)
        self.target_direction = (0.0, 1.0, 0.0)
        self.sun_distance_au = 10.0
        self.mission_mode = 0  # Index into mode list
    
    def reset(self, seed: Optional[int] = None) -> Tuple:
        """Reset environment and return initial observation."""
        if seed is not None:
            np.random.seed(seed)
        
        # Create swarm
        self.swarm = Swarm(self.num_cubes)
        size = round(self.num_cubes ** (1/3))
        create_cube_formation(self.swarm, size=size)
        
        # Create constellation with generous propulsion
        sep_reqs = SeparationRequirements(
            min_separation_delta_v=0.1,
            default_separation_velocity=1.0,
            min_group_size=1,
            max_groups=8,
            allow_single_cube_groups=True
        )
        
        self.constellation = Constellation(self.swarm, sep_reqs)
        
        # Set propulsion budget
        for ps in self.constellation._cube_propulsion.values():
            ps.max_delta_v = 100.0
            ps.remaining_delta_v = 100.0
        
        for grp in self.constellation._groups.values():
            total = sum(self.constellation._cube_propulsion[cid].remaining_delta_v 
                       for cid in grp.cube_ids)
            grp.propulsion.max_delta_v = total
            grp.propulsion.remaining_delta_v = total
        
        # Create controllers
        self.movement = MovementSystem(self.swarm, require_connectivity=False)
        self.controller = ConstellationController(self.constellation)
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Optionally randomize mission context
        self._randomize_mission_context()
        
        # Build observation
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            self.constellation,
            self.mission_mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au
        )
        
        # Build action masks
        action_masks = self.mask_builder.build_action_masks(
            self.constellation, self.controller, self.movement
        )
        
        return graph_data, mode_idx, env_features, action_masks
    
    def _randomize_mission_context(self) -> None:
        """Randomize mission context for domain randomization."""
        # Random sun direction
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.sun_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        )
        
        # Random earth direction (different from sun)
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.earth_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        )
        
        # Random target
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        self.target_direction = (
            np.sin(phi) * np.cos(theta),
            np.sin(phi) * np.sin(theta),
            np.cos(phi)
        )
        
        # Random distance
        self.sun_distance_au = np.random.uniform(1.0, 30.0)
        
        # Random mode
        self.mission_mode = np.random.randint(0, 6)
    
    def step(self, action_type: int, sub_action: int, 
             action_masks: Dict) -> Tuple:
        """
        Execute action and return new observation, reward, done, info.
        """
        self.current_step += 1
        
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
            
        elif action_type == 1 and action is not None:  # Separation
            result = self.controller.execute_separation(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used
            
        elif action_type == 2 and action is not None:  # Docking
            result = self.controller.execute_docking(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used
            
        elif action_type == 3 and action is not None:  # Maneuver
            result = self.controller.execute_maneuver(action)
            success = result.success
            reason = result.reason
            delta_v_used = result.delta_v_used
            
        elif action_type == 4:  # Noop
            success = True
            reason = "No operation"
        
        # Propagate time if multiple groups
        if self.constellation.get_num_groups() > 1:
            self.constellation.propagate(self.time_step)
        
        # Compute reward
        if success:
            task_reward = self.task.compute_reward(self.constellation)
            step_penalty = 0.001
            dv_penalty = 0.01 * delta_v_used
            reward = task_reward - step_penalty - dv_penalty
        else:
            reward = -0.1  # Invalid action penalty
        
        self.episode_reward += reward
        
        # Check termination
        terminated = (self.task.is_complete(self.constellation) or 
                      self.task.is_failed(self.constellation))
        truncated = self.current_step >= self.max_steps
        done = terminated or truncated
        
        # Build new observation
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            self.constellation,
            self.mission_mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au
        )
        
        # Build new action masks
        new_action_masks = self.mask_builder.build_action_masks(
            self.constellation, self.controller, self.movement
        )
        
        info = {
            'action_success': success,
            'action_reason': reason,
            'delta_v_used': delta_v_used,
            'task_progress': self.task.get_progress(self.constellation),
            'task_complete': self.task.is_complete(self.constellation),
            'num_groups': self.constellation.get_num_groups(),
            'max_baseline': self.constellation.get_max_baseline(),
            'episode_reward': self.episode_reward,
        }
        
        return graph_data, mode_idx, env_features, new_action_masks, reward, done, info


def train(config: TrainingConfig):
    """Main training loop."""
    print("=" * 60)
    print("CONSTELLATION PPO TRAINING")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    # Create task
    if config.task_type == "form_constellation":
        task = FormConstellationTask(
            target_num_groups=2,
            target_baseline=5000.0,
            min_group_size=8
        )
    elif config.task_type == "multi_point":
        task = MultiPointSensingTask(
            target_num_groups=4,
            target_volume=1e6,
            min_group_size=4
        )
    else:
        task = FormConstellationTask(
            target_num_groups=2,
            target_baseline=5000.0
        )
    
    # Create environment
    env = ConstellationTrainingEnv(
        num_cubes=config.num_cubes,
        task=task,
        max_steps=config.max_episode_steps,
        time_step=config.time_step
    )
    
    # Create agent
    ppo_config = PPOConfig(
        learning_rate=3e-4,
        hidden_dim=128,
        num_gnn_layers=3,
        max_cubes=config.num_cubes,
    )
    
    agent = ConstellationPPOAgent(ppo_config, device)
    
    # Create rollout buffer
    buffer = RolloutBuffer(config.rollout_steps, device)
    
    # Create logger
    logger = TrainingLogger(
        log_dir=os.path.join(config.log_dir, config.task_type),
        experiment_name=config.task_type,
        window_size=100,
        save_frequency=config.plot_interval
    )

    # Create recorder
    recorder = EpisodeRecorder(
        save_dir=os.path.join(config.log_dir, config.task_type),
        max_recordings=10,
        record_frequency=100  # Record every 100 episodes
    )

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_mean_reward = -float('inf')
    
    # Initialize environment
    graph_data, mode_idx, env_features, action_masks = env.reset()

    episode_count = 0
    recorder.start_episode(episode_count, {'task_type': config.task_type})
    
    total_steps = 0
    num_updates = 0
    episode_reward = 0.0
    episode_length = 0
    
    print(f"\nStarting training for {config.total_timesteps} timesteps...")
    print(f"Task: {config.task_type}")
    print(f"Num cubes: {config.num_cubes}")
    print(f"Rollout steps: {config.rollout_steps}")
    
    start_time = time.time()
    
    while total_steps < config.total_timesteps:
        # Collect rollout
        buffer.reset()
        
        for step in range(config.rollout_steps):
            total_steps += 1
            episode_length += 1
            
            # Get action from policy
            action_type, sub_action, log_prob, value, masks = agent.get_action(
                env.constellation,
                env.controller,
                env.movement,
                env.mission_mode,
                env.sun_direction,
                env.earth_direction,
                env.target_direction,
                env.sun_distance_au,
                deterministic=False
            )
            
            # Execute action
            (new_graph_data, new_mode_idx, new_env_features, 
             new_action_masks, reward, done, info) = env.step(
                action_type, sub_action, masks
            )

            recorder.record_step(
                env.constellation,
                action_type,
                sub_action,
                reward,
                info
            )
            
            episode_reward += reward
            
            # Store transition
            buffer.add(
                observation=graph_data,
                action_type=action_type,
                sub_action=sub_action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=done,
                action_mask=masks,
                mode_idx=env.mission_mode,
                env_features=env_features.numpy().flatten()
            )
            
            # Update state
            graph_data = new_graph_data
            mode_idx = new_mode_idx
            env_features = new_env_features
            action_masks = new_action_masks
            
            # Handle episode end
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Log episode metrics
                logger.log_episode(
                    reward=episode_reward,
                    length=episode_length,
                    success=info.get('task_complete', False),
                    task_progress=info.get('task_progress', 0.0),
                    num_groups=info.get('num_groups', 1),
                    max_baseline=info.get('max_baseline', 0.0),
                    delta_v_used=info.get('delta_v_used', 0.0)
                )

                # End recording
                recorder.end_episode(info.get('task_complete', False))

                # Start new recording
                episode_count += 1
                recorder.start_episode(episode_count, {'task_type': config.task_type})
                
                # Reset environment
                graph_data, mode_idx, env_features, action_masks = env.reset()
                episode_reward = 0.0
                episode_length = 0
            
            # Check if we should stop
            if total_steps >= config.total_timesteps:
                break
        
        # Compute returns and advantages
        with torch.no_grad():
            _, _, last_log_prob, last_value, _ = agent.get_action(
                env.constellation,
                env.controller,
                env.movement,
                env.mission_mode,
                env.sun_direction,
                env.earth_direction,
                env.target_direction,
                env.sun_distance_au,
                deterministic=False
            )
        
        buffer.compute_returns_and_advantages(
            last_value,
            ppo_config.gamma,
            ppo_config.gae_lambda
        )
        
        # Update policy
        update_metrics = agent.update(buffer)
        num_updates += 1
        
        logger.log_update(
            policy_loss=update_metrics['policy_loss'],
            value_loss=update_metrics['value_loss'],
            entropy=update_metrics['entropy'],
            approx_kl=update_metrics['approx_kl'],
            clip_fraction=update_metrics['clip_fraction'],
            learning_rate=ppo_config.learning_rate,
            timesteps=total_steps
        )
        
        # Save checkpoint
        if total_steps % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.save_dir, f"checkpoint_{total_steps}.pt"
            )
            agent.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")
            
            # Save best model
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_path = os.path.join(config.save_dir, "best_model.pt")
                agent.save(best_path)
                print(f"  New best model! Mean reward: {mean_reward:.4f}")
        
        # Evaluation
        if total_steps % config.eval_interval == 0:
            eval_reward = evaluate_agent(agent, env, num_episodes=5)
            print(f"  Evaluation reward (5 ep): {eval_reward:.4f}")

    # Create animation of best episode
    if recorder.best_recording is not None:
        recorder.create_best_episode_animation(filename="best_episode")
        recorder.save_recording_data(recorder.best_recording, "best_episode_data")

    # Print and save final summary
    logger.print_summary()
    logger.save_data()
    logger.save_plots()
    
    # Final save
    final_path = os.path.join(config.save_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    return agent


def evaluate_agent(agent: ConstellationPPOAgent,
                   env: ConstellationTrainingEnv,
                   num_episodes: int = 10) -> float:
    """Evaluate agent performance."""
    total_rewards = []
    
    for ep in range(num_episodes):
        graph_data, mode_idx, env_features, action_masks = env.reset(seed=ep)
        episode_reward = 0.0
        done = False
        
        while not done:
            action_type, sub_action, _, _, masks = agent.get_action(
                env.constellation,
                env.controller,
                env.movement,
                env.mission_mode,
                env.sun_direction,
                env.earth_direction,
                env.target_direction,
                env.sun_distance_au,
                deterministic=True  # Use deterministic actions for evaluation
            )
            
            (graph_data, mode_idx, env_features,
             action_masks, reward, done, info) = env.step(action_type, sub_action, masks)
            
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)


def main():
    """Entry point for training."""
    # import argparse
    
    # parser = argparse.ArgumentParser(description="Train constellation PPO agent")
    # parser.add_argument("--num-cubes", type=int, default=64,
    #                     help="Number of cubes in swarm")
    # parser.add_argument("--total-timesteps", type=int, default=1_000_000,
    #                     help="Total training timesteps")
    # parser.add_argument("--task", type=str, default="form_constellation",
    #                     choices=["form_constellation", "multi_point"],
    #                     help="Task type")
    # parser.add_argument("--save-dir", type=str, default="./checkpoints",
    #                     help="Directory for saving checkpoints")
    # parser.add_argument("--log-dir", type=str, default="./logs",
    #                     help="Directory for logs")
    # parser.add_argument("--rollout-steps", type=int, default=2048,
    #                     help="Steps per rollout")
    # parser.add_argument("--max-episode-steps", type=int, default=500,
    #                     help="Max steps per episode")
    
    # args = parser.parse_args()
    
    # config = TrainingConfig(
    #     num_cubes=args.num_cubes,
    #     total_timesteps=args.total_timesteps,
    #     task_type=args.task,
    #     save_dir=args.save_dir,
    #     log_dir=args.log_dir,
    #     rollout_steps=args.rollout_steps,
    #     max_episode_steps=args.max_episode_steps,
    # )
    params = {
        "num_cubes": 8, # 64
        "total_timesteps": 1_000, # 1_000_000
        "task_type": "form_constellation",
        "save_dir": "./checkpoints",
        "log_dir": "./logs",
        "rollout_steps": 2048,
        "max_episode_steps": 500,
    }
    config = TrainingConfig(**params)
    
    agent = train(config)
    
    return agent


if __name__ == "__main__":
    main()