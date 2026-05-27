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
from rl.env_wrapper import ConstellationTrainingEnv

from core.swarm import Swarm
from core.constellation import (
    Constellation, SeparationRequirements, DockingRequirements, CommunicationRequirements
)
from configs.formations import create_cube_formation
from mechanics.moves import MovementSystem
from mechanics.constellation_moves import ConstellationController
from tasks.constellation_tasks import ConstellationTask, FormConstellationTask
from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler


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
    
    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Task
    task_type: str = "form_constellation"
    curriculum_enabled: bool = False
    num_cubes_range: Tuple[int, int] = (8, 64)  # For curriculum


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

    # Create task and environment
    def _make_env_and_sampler(config):
        """Create environment and optional curriculum sampler."""
        if config.curriculum_enabled:
            curriculum = TaskCurriculum(num_cubes_range=config.num_cubes_range)
            sampler = CurriculumSampler(curriculum)
            task_key, task, num_cubes = sampler.sample()
        else:
            sampler = None
            task = FormConstellationTask(target_num_groups=2, target_baseline=5000.0)
            num_cubes = config.num_cubes
            task_key = 'form_constellation'
    
        env = ConstellationTrainingEnv(
            num_cubes=num_cubes,
            task=task,
            max_steps=config.max_episode_steps,
        )
        env._current_task_key = task_key

        return env, sampler
    
    def _episode_end_reset(env, sampler, config, info):
        """Reset env, optionally sampling a new task from the curriculum."""
        if sampler is not None:
            task_key = getattr(env, '_current_task_key', 'unknown')
            sampler.record_outcome(
                task_key,
                success=info.get('task_complete', False),
                progress=info.get('task_progress', 0.0),
            )
            new_task_key, new_task, num_cubes = sampler.sample()
            env.task = new_task
            env.num_cubes = num_cubes
            env._current_task_key = new_task_key
    
        return env.reset()

    env, sampler = _make_env_and_sampler(config)

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

    # Training metrics
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_mean_reward = -float('inf')
    
    # Initialize environment
    graph_data, mode_idx, env_features, action_masks = env.reset()

    episode_count = 0    
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

                episode_count += 1
                
                # Reset environment
                graph_data, mode_idx, env_features, action_masks = _episode_end_reset(
                    env, sampler, config, info
                )
                episode_reward = 0.0
                episode_length = 0

                # Print progress
                if episode_count % config.log_interval == 0:
                    elapsed = time.time() - start_time
                    mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
                    print(f"Episode {episode_count} | Total Steps: {total_steps} | "
                          f"Mean Reward: {mean_reward:.4f} | Total Elapsed Time: {elapsed:.2f}s")
            
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

        if num_updates % 10 == 0 and sampler is not None:
            sampler.print_status()
            # for key, stats in sampler.get_status_dict().items():
            #     logger.log_scalar(f"curriculum/{key}/tier", stats['tier'], total_steps)
            #     logger.log_scalar(f"curriculum/{key}/success", stats['rolling_success'], total_steps)
        
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
