"""
train_fast.py
=============
Updated training loop that uses:
- Multiple parallel environments (actually uses num_envs)
- The optimized PPO agent with AMP
- Larger model (256 hidden, 5 GNN layers)
- Vectorized PPO updates (no per-sample loop)
- Better logging and checkpointing
"""

import os
import time
import numpy as np
import torch
from dataclasses import dataclass
from collections import deque
from typing import Tuple, Optional

from rl.ppo_agent_fast import FastConstellationPPOAgent, PPOConfig, FastRolloutBuffer
from rl.observation_builder import ConstellationObservationBuilder, ActionMaskBuilder
from rl.training_logger import TrainingLogger
from rl.env_wrapper import ConstellationTrainingEnv
from parallel_env import ParallelEnvManager
from tasks.constellation_tasks import FormConstellationTask
from tasks.curriculum_tasks import TaskCurriculum, CurriculumSampler


@dataclass
class TrainingConfig:
    """Updated configuration with all new options."""
    # Environment
    num_cubes: int = 64
    max_episode_steps: int = 1000       # Was 500 [2] - longer episodes
    time_step: float = 10.0
    
    # Training
    total_timesteps: int = 50_000_000   # 10x more than original 100k default [1]
    rollout_steps: int = 4096           # Was 2048 [1] - larger rollouts
    num_envs: int = 16                  # ACTUALLY USED now (was defined but unused [2])
    num_workers: int = 16               # CPU workers for parallel envs
    
    # Model (LARGER)
    hidden_dim: int = 256               # Was 128 [4]
    num_gnn_layers: int = 5             # Was 3 [4]
    num_attention_heads: int = 8        # Was 4 [4]
    learning_rate: float = 1e-4         # Lower for larger model (was 3e-4 [2])
    
    # Performance
    use_amp: bool = True                # Mixed precision on A100
    compile_model: bool = False         # torch.compile (experimental)
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100_000
    eval_interval: int = 50_000
    plot_interval: int = 10
    
    # Paths
    save_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    
    # Task
    task_type: str = "form_constellation"
    curriculum_enabled: bool = True
    num_cubes_range: Tuple[int, int] = (8, 64)


def train(config: TrainingConfig):
    """
    Main training loop — optimized version.
    
    Key differences from original [2]:
    1. Runs num_envs environments in parallel (original only ran 1)
    2. Uses AMP for ~2x GPU throughput on A100
    3. Vectorized PPO update (no per-sample Python loop) [4]
    4. Larger model (256 hidden, 5 GNN layers vs 128/3) [4]
    5. More total timesteps (50M vs 100K default) [1]
    """
    print("=" * 60)
    print("CONSTELLATION PPO TRAINING (OPTIMIZED)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
    
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    # --- Create PPO Agent with LARGER model ---
    ppo_config = PPOConfig(
        learning_rate=config.learning_rate,
        hidden_dim=config.hidden_dim,           # 256 (was 128)
        num_gnn_layers=config.num_gnn_layers,   # 5 (was 3)
        num_attention_heads=config.num_attention_heads,  # 8 (was 4)
        max_cubes=config.num_cubes,
        use_amp=config.use_amp,
        entropy_coef=0.02,                      # Higher for more exploration
        num_minibatches=8,                      # More minibatches for larger buffer
    )
    
    agent = FastConstellationPPOAgent(ppo_config, device)
    
    # Setup LR schedule
    total_updates = (config.total_timesteps // (config.rollout_steps * config.num_envs)) * ppo_config.num_epochs * ppo_config.num_minibatches
    agent.setup_scheduler(total_updates)
    
    print(f"\nModel size: {sum(p.numel() for p in agent.policy.parameters()):,} parameters")
    print(f"Mixed precision: {config.use_amp}")
    print(f"Parallel environments: {config.num_envs}")
    print(f"Effective batch size per rollout: {config.rollout_steps * config.num_envs}")
    
    # --- Create Parallel Environments ---
    # Instead of a single env [2], we run num_envs in parallel
    
    env_manager = ParallelEnvManager(
        num_envs=config.num_envs,
        config=config,
    )
    
    # Also keep a single env for evaluation
    eval_env = ConstellationTrainingEnv(
        num_cubes=config.num_cubes,
        task=FormConstellationTask(target_num_groups=2, target_baseline=5000.0),
        max_steps=config.max_episode_steps,
    )
    
    # --- Create Rollout Buffer (sized for parallel envs) ---
    buffer = FastRolloutBuffer(config.rollout_steps, config.num_envs, device)
    
    # --- Create Logger ---
    logger = TrainingLogger(
        log_dir=os.path.join(config.log_dir, config.task_type),
        experiment_name=config.task_type,
        window_size=100,
        save_frequency=config.plot_interval,
    )

    # --- Training Metrics ---
    episode_rewards = deque(maxlen=100)
    episode_lengths = deque(maxlen=100)
    best_mean_reward = -float('inf')
    
    total_steps = 0
    num_updates = 0
    episode_count = 0
    
    # Per-env tracking
    env_episode_rewards = [0.0] * config.num_envs
    env_episode_lengths = [0] * config.num_envs
    
    print(f"\nStarting training for {config.total_timesteps:,} timesteps...")
    print(f"  Rollout size: {config.rollout_steps} steps × {config.num_envs} envs = "
          f"{config.rollout_steps * config.num_envs:,} transitions per update")
    
    start_time = time.time()
    
    while total_steps < config.total_timesteps:
        rollout_start = time.time()
        buffer.reset()
        
        # === COLLECT ROLLOUT FROM ALL PARALLEL ENVS ===
        for step in range(config.rollout_steps):
            # Get actions for ALL envs simultaneously
            actions_for_envs = []
            log_probs = []
            values = []
            graph_datas = []
            
            for env_idx in range(config.num_envs):
                obs = env_manager.current_obs[env_idx]
                graph_data, mode_idx, env_features, action_masks = obs
                
                # Get action from policy (batching across envs would be even better,
                # but requires restructuring observation building)
                graph_data, mode_idx, env_features, action_masks = env_manager.current_obs[env_idx]
                action_type, sub_action, log_prob, value, masks = agent.get_action_from_obs(
                    graph_data, mode_idx, env_features, action_masks
                )
                
                actions_for_envs.append((action_type, sub_action, masks))
                log_probs.append(log_prob)
                values.append(value)
                graph_datas.append(graph_data)
            
            # Step ALL envs in parallel (this is the big speedup)
            results = env_manager.step(actions_for_envs)
            
            # Store transitions for all envs
            for env_idx, (obs, reward, done, info) in enumerate(results):
                buffer.add(
                    env_idx=env_idx,
                    step=step,
                    observation=graph_datas[env_idx],
                    action_type=actions_for_envs[env_idx][0],
                    sub_action=actions_for_envs[env_idx][1],
                    log_prob=log_probs[env_idx],
                    reward=reward,
                    value=values[env_idx],
                    done=done,
                    action_mask=actions_for_envs[env_idx][2],
                    mode_idx=info.get('mode_idx', 0),
                    env_features=info.get('env_features', np.zeros(12)),
                )
                
                env_episode_rewards[env_idx] += reward
                env_episode_lengths[env_idx] += 1
                
                if done:
                    episode_rewards.append(env_episode_rewards[env_idx])
                    episode_lengths.append(env_episode_lengths[env_idx])
                    episode_count += 1
                    
                    logger.log_episode(
                        reward=env_episode_rewards[env_idx],
                        length=env_episode_lengths[env_idx],
                        success=info.get('task_complete', False),
                        task_progress=info.get('task_progress', 0.0),
                        num_groups=info.get('num_groups', 1),
                        max_baseline=info.get('max_baseline', 0.0),
                        delta_v_used=info.get('delta_v_used', 0.0),
                    )
                    
                    env_episode_rewards[env_idx] = 0.0
                    env_episode_lengths[env_idx] = 0
                
                # Update current obs
                env_manager.current_obs[env_idx] = obs
            
            total_steps += config.num_envs  # We took num_envs steps
        
        rollout_time = time.time() - rollout_start
        
        # === COMPUTE ADVANTAGES ===
        # Get bootstrap values for all envs
        with torch.no_grad():
            last_values = []
            for env_idx in range(config.num_envs):
                graph_data, mode_idx, env_features, action_masks = env_manager.current_obs[env_idx]
                _, _, _, value, _ = agent.get_action_from_obs(
                    graph_data, mode_idx, env_features, action_masks,
                    deterministic=False,
                )
                last_values.append(value)        
                
        buffer.compute_returns_and_advantages(
            last_values, ppo_config.gamma, ppo_config.gae_lambda
        )
        
        # === UPDATE POLICY (vectorized, with AMP) ===
        update_start = time.time()
        update_metrics = agent.update(buffer)
        update_time = time.time() - update_start
        num_updates += 1
        
        # === LOGGING ===
        logger.log_update(
            policy_loss=update_metrics['policy_loss'],
            value_loss=update_metrics['value_loss'],
            entropy=update_metrics['entropy'],
            approx_kl=update_metrics['approx_kl'],
            clip_fraction=update_metrics['clip_fraction'],
            learning_rate=agent.optimizer.param_groups[0]['lr'],
            timesteps=total_steps,
        )
        
        if num_updates % 5 == 0:
            elapsed = time.time() - start_time
            fps = total_steps / elapsed
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            mean_length = np.mean(episode_lengths) if episode_lengths else 0.0
            
            print(f"Update {num_updates:4d} | Steps: {total_steps:>10,} | "
                  f"FPS: {fps:.0f} | "
                  f"Mean Reward: {mean_reward:>8.3f} | "
                  f"Mean Ep Len: {mean_length:.0f} | "
                  f"Rollout: {rollout_time:.1f}s | "
                  f"Update: {update_time:.1f}s | "
                  f"KL: {update_metrics['approx_kl']:.4f}")
        
        # === CHECKPOINTING ===
        if total_steps % config.save_interval < config.rollout_steps * config.num_envs:
            checkpoint_path = os.path.join(config.save_dir, f"checkpoint_{total_steps}.pt")
            agent.save(checkpoint_path)
            
            mean_reward = np.mean(episode_rewards) if episode_rewards else 0.0
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_path = os.path.join(config.save_dir, "best_model.pt")
                agent.save(best_path)
                print(f"  ★ New best model! Mean reward: {mean_reward:.4f}")
        
        # === EVALUATION ===
        if total_steps % config.eval_interval < config.rollout_steps * config.num_envs:
            eval_reward = evaluate_agent(agent, eval_env, num_episodes=5)
            print(f"  Evaluation (5 ep): {eval_reward:.4f}")
    
    # Final save
    env_manager.close()
    logger.print_summary()
    logger.save_data()
    logger.save_plots()
    
    final_path = os.path.join(config.save_dir, "final_model.pt")
    agent.save(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    return agent


def evaluate_agent(agent, env, num_episodes=10):
    """Evaluate agent deterministically."""
    total_rewards = []
    
    for ep in range(num_episodes):
        graph_data, mode_idx, env_features, action_masks = env.reset(seed=ep)
        episode_reward = 0.0
        done = False
        
        while not done:
            action_type, sub_action, _, _, masks = agent.get_action(
                env.constellation, env.controller, env.movement,
                env.mission_mode, env.sun_direction, env.earth_direction,
                env.target_direction, env.sun_distance_au,
                deterministic=True,
            )
            (graph_data, mode_idx, env_features,
             action_masks, reward, done, info) = env.step(action_type, sub_action, masks)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)
