"""
ppo_agent.py
============
PPO agent implementation for constellation control using the GNN policy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

from core.constellation import Constellation
from mechanics.constellation_moves import ConstellationController
from mechanics.moves import MovementSystem
from rl.gnn_encoder import ConstellationObservationEncoder
from rl.policy_heads import HierarchicalPolicy
from rl.observation_builder import (
    ConstellationObservationBuilder, 
    ActionMaskBuilder,
    ObservationConfig
)


@dataclass
class PPOConfig:
    """Configuration for PPO training."""
    # Learning rates
    learning_rate: float = 3e-4
    lr_schedule: str = 'linear'  # 'linear', 'constant'
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_value: bool = True
    value_clip_epsilon: float = 0.2
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    num_epochs: int = 4
    num_minibatches: int = 4
    normalize_advantage: bool = True
    
    # Architecture
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    num_attention_heads: int = 4
    
    # Environment
    max_cubes: int = 256
    max_groups: int = 8
    num_mission_modes: int = 6


class RolloutBuffer:
    """Buffer for storing rollout data."""
    
    def __init__(self, buffer_size: int, device: torch.device):
        self.buffer_size = buffer_size
        self.device = device
        self.reset()
    
    def reset(self):
        """Clear the buffer."""
        self.observations = []
        self.actions = []
        self.action_types = []
        self.sub_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.action_masks = []
        self.mode_indices = []
        self.env_features = []
        
        self.ptr = 0
    
    def add(self,
            observation,
            action_type: int,
            sub_action: int,
            log_prob: float,
            reward: float,
            value: float,
            done: bool,
            action_mask: Dict,
            mode_idx: int,
            env_features: np.ndarray):
        """Add a transition to the buffer."""
        self.observations.append(observation)
        self.action_types.append(action_type)
        self.sub_actions.append(sub_action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.action_masks.append(action_mask)
        self.mode_indices.append(mode_idx)
        self.env_features.append(env_features)
        
        self.ptr += 1
    
    def compute_returns_and_advantages(self, 
                                        last_value: float,
                                        gamma: float,
                                        gae_lambda: float):
        """Compute GAE advantages and returns."""
        rewards = np.array(self.rewards)
        values = np.array(self.values + [last_value])
        dones = np.array(self.dones + [False])
        
        # GAE computation
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            next_non_terminal = 1.0 - dones[t + 1]
            delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        
        returns = advantages + values[:-1]
        
        self.advantages = advantages
        self.returns = returns
    
    def get_batches(self, num_minibatches: int):
        """Generate minibatches for training."""
        batch_size = len(self.observations)
        minibatch_size = batch_size // num_minibatches
        
        indices = np.random.permutation(batch_size)
        
        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            batch_indices = indices[start:end]
            
            yield {
                'observations': [self.observations[i] for i in batch_indices],
                'action_types': torch.tensor([self.action_types[i] for i in batch_indices]),
                'sub_actions': torch.tensor([self.sub_actions[i] for i in batch_indices]),
                'log_probs': torch.tensor([self.log_probs[i] for i in batch_indices]),
                'advantages': torch.tensor([self.advantages[i] for i in batch_indices], dtype=torch.float32),
                'returns': torch.tensor([self.returns[i] for i in batch_indices], dtype=torch.float32),
                'values': torch.tensor([self.values[i] for i in batch_indices], dtype=torch.float32),
                'action_masks': [self.action_masks[i] for i in batch_indices],
                'mode_indices': torch.tensor([self.mode_indices[i] for i in batch_indices]),
                'env_features': torch.tensor(np.array([self.env_features[i] for i in batch_indices]), dtype=torch.float32),
            }


class ConstellationPPOAgent:
    """
    PPO agent for constellation control.
    
    Uses a GNN-based policy with hierarchical action heads.
    """
    
    def __init__(self,
                 config: PPOConfig,
                 device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build encoder
        self.encoder = ConstellationObservationEncoder(
            cube_input_dim=25,
            group_input_dim=12,
            hidden_dim=config.hidden_dim,
            num_gnn_layers=config.num_gnn_layers,
            num_attention_heads=config.num_attention_heads,
            num_mission_modes=config.num_mission_modes,
            env_feature_dim=12,
        )
        
        # Build policy
        self.policy = HierarchicalPolicy(
            encoder=self.encoder,
            hidden_dim=config.hidden_dim,
            max_cubes=config.max_cubes,
            max_groups=config.max_groups,
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5
        )
        
        # Observation and action mask builders
        self.obs_builder = ConstellationObservationBuilder()
        self.mask_builder = ActionMaskBuilder(
            max_cubes=config.max_cubes,
            max_groups=config.max_groups,
        )
        
        # Training state
        self.total_steps = 0
        self.updates = 0
    
    def get_action(self,
                   constellation: 'Constellation',
                   controller: 'ConstellationController',
                   movement: 'MovementSystem',
                   mission_mode: int,
                   sun_direction: Tuple[float, float, float],
                   earth_direction: Tuple[float, float, float],
                   target_direction: Tuple[float, float, float],
                   sun_distance_au: float = 10.0,
                   deterministic: bool = False) -> Tuple[int, int, float, float]:
        """
        Get action from policy.
        
        Returns:
            (action_type, sub_action, log_prob, value)
        """
        # Build observation
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            constellation,
            mission_mode,
            sun_direction,
            earth_direction,
            target_direction,
            sun_distance_au
        )
        
        # Build action masks
        action_masks = self.mask_builder.build_action_masks(
            constellation, controller, movement
        )
        
        # Move to device
        graph_data = graph_data.to(self.device)
        mode_idx = mode_idx.to(self.device)
        env_features = env_features.to(self.device)
        
        # Move masks to device
        for key, mask in action_masks.items():
            if isinstance(mask, torch.Tensor):
                action_masks[key] = mask.to(self.device)
        
        # Get action from policy
        with torch.no_grad():
            result = self.policy.get_action_and_value(
                graph_data,
                mode_idx,
                env_features,
                action_masks,
                deterministic=deterministic
            )
        
        action_type = result['action_type'].item()
        sub_action = result['sub_action'].item()
        log_prob = result['log_prob'].item()
        value = result['value'].item()
        
        return action_type, sub_action, log_prob, value, action_masks
    
    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """
        Update policy using PPO.
        
        Returns:
            Dictionary of training metrics
        """
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
        }
        
        num_updates = 0
        
        for epoch in range(self.config.num_epochs):
            for batch in buffer.get_batches(self.config.num_minibatches):
                # Move batch to device
                action_types = batch['action_types'].to(self.device)
                sub_actions = batch['sub_actions'].to(self.device)
                old_log_probs = batch['log_probs'].to(self.device)
                advantages = batch['advantages'].to(self.device)
                returns = batch['returns'].to(self.device)
                old_values = batch['values'].to(self.device)
                mode_indices = batch['mode_indices'].to(self.device)
                env_features = batch['env_features'].to(self.device)
                
                # Normalize advantages
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Rebuild graph batch from observations
                from torch_geometric.data import Batch
                graph_batch = Batch.from_data_list(batch['observations']).to(self.device)
                
                # Collate action masks
                action_masks = self._collate_action_masks(batch['action_masks'])
                
                # Forward pass
                outputs = self.policy(graph_batch, mode_indices, env_features, action_masks)
                
                # Compute new log probs for the taken actions
                new_log_probs = torch.zeros_like(old_log_probs)
                entropies = torch.zeros_like(old_log_probs)
                
                batch_size = action_types.size(0)
                
                for b in range(batch_size):
                    at = action_types[b].item()
                    sa = sub_actions[b].item()
                    
                    # Action type distribution
                    type_dist = torch.distributions.Categorical(
                        logits=outputs['action_type_logits'][b]
                    )
                    type_log_prob = type_dist.log_prob(action_types[b])
                    type_entropy = type_dist.entropy()
                    
                    # Sub-action distribution
                    if at == 0:
                        logits = outputs['cube_move_logits'][b]
                    elif at == 1:
                        logits = outputs['separation_logits'][b]
                    elif at == 2:
                        logits = outputs['docking_logits'][b]
                    elif at == 3:
                        logits = outputs['maneuver_logits'][b]
                    else:  # noop
                        new_log_probs[b] = type_log_prob
                        entropies[b] = type_entropy
                        continue
                    
                    sub_dist = torch.distributions.Categorical(logits=logits)
                    sub_log_prob = sub_dist.log_prob(sub_actions[b])
                    sub_entropy = sub_dist.entropy()
                    
                    new_log_probs[b] = type_log_prob + sub_log_prob
                    entropies[b] = type_entropy + sub_entropy
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 
                                    1.0 - self.config.clip_epsilon,
                                    1.0 + self.config.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                new_values = outputs['value'].squeeze(-1)
                
                if self.config.clip_value:
                    value_clipped = old_values + torch.clamp(
                        new_values - old_values,
                        -self.config.value_clip_epsilon,
                        self.config.value_clip_epsilon
                    )
                    value_loss1 = (new_values - returns) ** 2
                    value_loss2 = (value_clipped - returns) ** 2
                    value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
                else:
                    value_loss = 0.5 * ((new_values - returns) ** 2).mean()
                
                # Entropy bonus
                entropy_loss = -entropies.mean()
                
                # Total loss
                loss = (policy_loss + 
                        self.config.value_loss_coef * value_loss +
                        self.config.entropy_coef * entropy_loss)
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), 
                        self.config.max_grad_norm
                    )
                
                self.optimizer.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_fraction = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropies.mean().item()
                metrics['approx_kl'] += approx_kl.item()
                metrics['clip_fraction'] += clip_fraction.item()
                num_updates += 1
        
        # Average metrics
        for key in metrics:
            metrics[key] /= max(num_updates, 1)
        
        self.updates += 1
        
        return metrics
    
    def _collate_action_masks(self, masks_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate action masks from multiple timesteps into batched tensors."""
        collated = {}
        
        # Get keys from first mask
        if not masks_list:
            return collated
        
        keys_to_collate = [k for k in masks_list[0].keys() if not k.startswith('_')]
        
        for key in keys_to_collate:
            tensors = [m[key] for m in masks_list if isinstance(m.get(key), torch.Tensor)]
            if tensors:
                collated[key] = torch.cat(tensors, dim=0).to(self.device)
        
        return collated
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'updates': self.updates,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)