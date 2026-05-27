"""
ppo_agent_fast.py
=============================
Replaces the inner loop in update() with vectorized operations,
and adds mixed-precision support.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from torch_geometric.data import Batch

from rl.gnn_encoder import ConstellationObservationEncoder
from rl.policy_heads import HierarchicalPolicy
from rl.observation_builder import ConstellationObservationBuilder, ActionMaskBuilder


@dataclass
class PPOConfig:
    """Updated configuration for PPO training."""
    # Learning rates
    learning_rate: float = 1e-4         # Lower LR for larger model
    lr_schedule: str = 'cosine'
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    
    # PPO hyperparameters
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    clip_value: bool = True
    value_clip_epsilon: float = 0.2
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.02          # Higher for more exploration
    max_grad_norm: float = 0.5
    
    # Training
    num_epochs: int = 4
    num_minibatches: int = 8            # More minibatches for larger buffer
    normalize_advantage: bool = True
    
    # Architecture (LARGER)
    hidden_dim: int = 256               # Was 128 [4]
    num_gnn_layers: int = 5             # Was 3 [4]
    num_attention_heads: int = 8        # Was 4 [4]
    
    # Environment
    max_cubes: int = 256
    max_groups: int = 8
    num_mission_modes: int = 6
    
    # Performance
    use_amp: bool = True                # Mixed precision
    compile_model: bool = False         # torch.compile (PyTorch 2.0+)


class FastRolloutBuffer:
    """
    Optimized rollout buffer that supports multiple parallel environments.

    Layout: transitions are stored at index = step * num_envs + env_idx,
    so all data from the same timestep across envs is contiguous.

    Key differences from original RolloutBuffer [4]:
    - Pre-allocates scalar arrays (no per-step list appends)
    - Supports num_envs parallel environments (original only supported 1) [2]
    - compute_returns_and_advantages handles per-env episode boundaries
    - get_batches returns structure expected by vectorized update
    """

    def __init__(self, buffer_size: int, num_envs: int, device: torch.device):
        """
        Args:
            buffer_size: Number of steps per environment per rollout
            num_envs: Number of parallel environments
            device: Torch device for tensor operations during batching
        """
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.total_size = buffer_size * num_envs
        self.reset()

    def reset(self):
        """Clear the buffer and pre-allocate storage."""
        # Pre-allocated scalar arrays (fixed size, no appends)
        self.rewards = np.zeros(self.total_size, dtype=np.float32)
        self.values = np.zeros(self.total_size, dtype=np.float32)
        self.log_probs = np.zeros(self.total_size, dtype=np.float32)
        self.dones = np.zeros(self.total_size, dtype=np.float32)
        self.action_types = np.zeros(self.total_size, dtype=np.int64)
        self.sub_actions = np.zeros(self.total_size, dtype=np.int64)
        self.mode_indices = np.zeros(self.total_size, dtype=np.int64)

        # Variable-size data (graph observations differ per constellation size)
        self.observations = [None] * self.total_size
        self.action_masks = [None] * self.total_size
        self.env_features = np.zeros((self.total_size, 12), dtype=np.float32)

        # Computed during finalization
        self.advantages = np.zeros(self.total_size, dtype=np.float32)
        self.returns = np.zeros(self.total_size, dtype=np.float32)

        self.ptr = 0
        self.full = False

    def add(
        self,
        env_idx: int,
        step: int,
        observation,
        action_type: int,
        sub_action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool,
        action_mask: Dict,
        mode_idx: int,
        env_features: np.ndarray,
    ):
        """
        Add a single transition.

        Storage index = step * num_envs + env_idx, which keeps all
        same-timestep data from different envs contiguous in memory.

        Args:
            env_idx: Which parallel environment this came from
            step: Which rollout step (0 to buffer_size-1)
            observation: PyG graph Data object (variable size)
            action_type: High-level action type index (0-4) [3]
            sub_action: Sub-action index within chosen type
            log_prob: Log probability of the action under current policy
            reward: Reward received after taking the action
            value: Value estimate at this state
            done: Whether episode ended after this step
            action_mask: Dict of action masks for this state
            mode_idx: Mission mode index
            env_features: Environmental feature vector (length 12)
        """
        idx = step * self.num_envs + env_idx

        self.observations[idx] = observation
        self.action_types[idx] = action_type
        self.sub_actions[idx] = sub_action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.action_masks[idx] = action_mask
        self.mode_indices[idx] = mode_idx

        if env_features is not None:
            feat = np.asarray(env_features, dtype=np.float32).flatten()
            self.env_features[idx, : len(feat)] = feat[: 12]

        self.ptr = max(self.ptr, idx + 1)

    def compute_returns_and_advantages(
        self,
        last_values: List[float],
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """
        Compute GAE advantages and discounted returns per environment.

        Unlike the original single-env implementation [4], this correctly
        handles episode boundaries independently for each parallel environment.

        Args:
            last_values: Bootstrap value for each env at the end of the rollout
                         (list of length num_envs)
            gamma: Discount factor
            gae_lambda: GAE lambda for bias-variance tradeoff
        """
        for env_idx in range(self.num_envs):
            last_gae = 0.0
            last_val = last_values[env_idx]

            # Walk backwards through this env's steps
            for step in reversed(range(self.buffer_size)):
                idx = step * self.num_envs + env_idx

                # Determine the next value
                if step == self.buffer_size - 1:
                    next_value = last_val
                    next_non_terminal = 1.0 - self.dones[idx]
                else:
                    next_idx = (step + 1) * self.num_envs + env_idx
                    next_value = self.values[next_idx]
                    next_non_terminal = 1.0 - self.dones[idx]

                # TD error
                delta = (
                    self.rewards[idx]
                    + gamma * next_value * next_non_terminal
                    - self.values[idx]
                )

                # GAE accumulation (reset on episode boundary)
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
                self.advantages[idx] = last_gae

            # Returns = advantages + values
            for step in range(self.buffer_size):
                idx = step * self.num_envs + env_idx
                self.returns[idx] = self.advantages[idx] + self.values[idx]

    def get_batches(self, num_minibatches: int) -> Generator[Dict, None, None]:
        """
        Generate randomized minibatches for PPO update epochs.

        Yields dicts matching the structure expected by
        FastConstellationPPOAgent.update() — specifically:
        - 'observations': list of PyG Data objects
        - 'action_types': LongTensor [minibatch_size]
        - 'sub_actions': LongTensor [minibatch_size]
        - 'log_probs': FloatTensor [minibatch_size]
        - 'advantages': FloatTensor [minibatch_size]
        - 'returns': FloatTensor [minibatch_size]
        - 'values': FloatTensor [minibatch_size]
        - 'action_masks': list of mask dicts
        - 'mode_indices': LongTensor [minibatch_size]
        - 'env_features': FloatTensor [minibatch_size, 12]

        Args:
            num_minibatches: Number of minibatches to split the buffer into

        Yields:
            Dict with batched data for one minibatch
        """
        batch_size = self.total_size
        minibatch_size = batch_size // num_minibatches

        # Ensure we don't lose samples due to integer division
        indices = np.random.permutation(batch_size)

        for start in range(0, batch_size, minibatch_size):
            end = min(start + minibatch_size, batch_size)
            batch_indices = indices[start:end]

            # Skip incomplete final minibatch if too small
            if len(batch_indices) < minibatch_size // 2:
                continue

            yield {
                "observations": [self.observations[i] for i in batch_indices],
                "action_types": torch.from_numpy(
                    self.action_types[batch_indices]
                ).long(),
                "sub_actions": torch.from_numpy(
                    self.sub_actions[batch_indices]
                ).long(),
                "log_probs": torch.from_numpy(
                    self.log_probs[batch_indices]
                ).float(),
                "advantages": torch.from_numpy(
                    self.advantages[batch_indices]
                ).float(),
                "returns": torch.from_numpy(
                    self.returns[batch_indices]
                ).float(),
                "values": torch.from_numpy(
                    self.values[batch_indices]
                ).float(),
                "action_masks": [self.action_masks[i] for i in batch_indices],
                "mode_indices": torch.from_numpy(
                    self.mode_indices[batch_indices]
                ).long(),
                "env_features": torch.from_numpy(
                    self.env_features[batch_indices]
                ).float(),
            }

    @property
    def size(self) -> int:
        """Number of transitions currently stored."""
        return self.ptr

    def get_statistics(self) -> Dict[str, float]:
        """
        Get buffer statistics for logging/debugging.

        Returns:
            Dict with reward stats, value stats, advantage stats, etc.
        """
        return {
            "buffer_size": self.total_size,
            "mean_reward": float(np.mean(self.rewards[: self.ptr])),
            "std_reward": float(np.std(self.rewards[: self.ptr])),
            "mean_value": float(np.mean(self.values[: self.ptr])),
            "mean_advantage": float(np.mean(self.advantages[: self.ptr])),
            "std_advantage": float(np.std(self.advantages[: self.ptr])),
            "mean_return": float(np.mean(self.returns[: self.ptr])),
            "done_fraction": float(np.mean(self.dones[: self.ptr])),
            "noop_fraction": float(
                np.mean(self.action_types[: self.ptr] == 4)
            ),
        }


class FastConstellationPPOAgent:
    """
    Optimized PPO agent with:
    - Mixed precision (AMP) for ~2x speedup on A100
    - Vectorized log-prob computation (eliminates per-sample loop)
    - Cosine LR schedule with warmup
    - Gradient accumulation support
    """
    
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build encoder (LARGER)
        self.encoder = ConstellationObservationEncoder(
            cube_input_dim=25,
            group_input_dim=12,
            hidden_dim=config.hidden_dim,          # 256
            num_gnn_layers=config.num_gnn_layers,  # 5
            num_attention_heads=config.num_attention_heads,  # 8
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
        
        # Optional: torch.compile for PyTorch 2.0+
        if config.compile_model and hasattr(torch, 'compile'):
            print("  Compiling model with torch.compile...")
            self.policy = torch.compile(self.policy, mode='reduce-overhead')
        
        # Count parameters
        total_params = sum(p.numel() for p in self.policy.parameters())
        trainable_params = sum(p.numel() for p in self.policy.parameters() if p.requires_grad)
        print(f"  Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Optimizer - AdamW often works better for larger models
        self.optimizer = optim.AdamW(
            self.policy.parameters(),
            lr=config.learning_rate,
            eps=1e-5,
            weight_decay=1e-4,
        )
        
        # Learning rate scheduler
        self.scheduler = None  # Will be created when total_timesteps is known
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=config.use_amp)
        
        # Observation and action mask builders
        self.obs_builder = ConstellationObservationBuilder()
        self.mask_builder = ActionMaskBuilder(
            max_cubes=config.max_cubes,
            max_groups=config.max_groups,
        )
        
        # Training state
        self.total_steps = 0
        self.updates = 0
    
    def setup_scheduler(self, total_updates: int):
        """Setup cosine LR schedule after knowing total training length."""
        warmup_updates = self.config.warmup_steps
        
        def lr_lambda(current_step):
            if current_step < warmup_updates:
                return float(current_step) / float(max(1, warmup_updates))
            progress = float(current_step - warmup_updates) / float(
                max(1, total_updates - warmup_updates)
            )
            return max(
                self.config.min_lr / self.config.learning_rate,
                0.5 * (1.0 + np.cos(np.pi * progress))
            )
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def get_action(self, constellation, controller, movement,
                   mission_mode, sun_direction, earth_direction,
                   target_direction, sun_distance_au=10.0,
                   deterministic=False):
        """Get action from policy (same interface as original)."""
        graph_data, mode_idx, env_features = self.obs_builder.build_observation(
            constellation, mission_mode, sun_direction,
            earth_direction, target_direction, sun_distance_au
        )
        
        action_masks = self.mask_builder.build_action_masks(
            constellation, controller, movement
        )
        
        graph_data = graph_data.to(self.device)
        mode_idx = mode_idx.to(self.device)
        env_features = env_features.to(self.device)
        
        for key, mask in action_masks.items():
            if isinstance(mask, torch.Tensor):
                action_masks[key] = mask.to(self.device)
        
        with torch.no_grad():
            # Use AMP for inference too
            with autocast(enabled=self.config.use_amp):
                result = self.policy.get_action_and_value(
                    graph_data, mode_idx, env_features,
                    action_masks, deterministic=deterministic
                )
        
        action_type = result['action_type'].item()
        sub_action = result['sub_action'].item()
        log_prob = result['log_prob'].item()
        value = result['value'].item()
        
        return action_type, sub_action, log_prob, value, action_masks


    def get_action_from_obs(self, graph_data, mode_idx, env_features, 
                             action_masks, deterministic=False):
        """Get action from pre-built observation tensors."""
        graph_data = graph_data.to(self.device)
        mode_idx = mode_idx.to(self.device)
        env_features = env_features.to(self.device)
        for key, mask in action_masks.items():
            if isinstance(mask, torch.Tensor):
                action_masks[key] = mask.to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.config.use_amp):
                result = self.policy.get_action_and_value(
                    graph_data, mode_idx, env_features,
                    action_masks, deterministic=deterministic
                )
        
        return (result['action_type'].item(), result['sub_action'].item(),
            result['log_prob'].item(), result['value'].item(), action_masks)

    def update(self, buffer) -> Dict[str, float]:
        """
        Optimized PPO update with AMP and vectorized computation.
        
        Key difference from original [4]: eliminates the per-sample Python loop
        by computing log probs for ALL action types at once and then selecting.
        """
        metrics = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'grad_norm': 0.0,
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
                
                if self.config.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                raw_obs = batch['observations']
                
                # Defensive extraction - handle both correctly stored Data objects
                # and accidentally stored tuples (graph_data, mode_idx, env_features, masks)
                extracted_graphs = []
                for obs in raw_obs:
                    if isinstance(obs, tuple):
                        # Fallback: observation was stored as full tuple, extract graph_data
                        graph_data = obs[0]
                    else:
                        graph_data = obs
                    
                    if not hasattr(graph_data, 'stores_as'):
                        raise TypeError(
                            f"Expected PyG Data object in observations buffer, "
                            f"got {type(graph_data).__name__}. "
                            f"Check that buffer.add() is called with graph_data, not the full obs tuple."
                        )
                    extracted_graphs.append(graph_data)
                
                graph_batch = Batch.from_data_list(extracted_graphs).to(self.device)
                action_masks = self._collate_action_masks(batch['action_masks'])
                
                # === MIXED PRECISION FORWARD PASS ===
                with autocast(enabled=self.config.use_amp):
                    outputs = self.policy(graph_batch, mode_indices, env_features, action_masks)
                    
                    # === VECTORIZED LOG PROB COMPUTATION ===
                    # Instead of looping per sample [4], compute all at once
                    new_log_probs, entropies = self._vectorized_log_probs(
                        outputs, action_types, sub_actions
                    )
                    
                    # PPO policy loss
                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon
                    ) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss with clipping
                    new_values = outputs['value'].squeeze(-1)
                    if self.config.clip_value:
                        value_clipped = old_values + torch.clamp(
                            new_values - old_values,
                            -self.config.value_clip_epsilon,
                            self.config.value_clip_epsilon
                        )
                        value_loss = 0.5 * torch.max(
                            (new_values - returns) ** 2,
                            (value_clipped - returns) ** 2
                        ).mean()
                    else:
                        value_loss = 0.5 * ((new_values - returns) ** 2).mean()
                    
                    entropy_loss = -entropies.mean()
                    
                    loss = (
                        policy_loss +
                        self.config.value_loss_coef * value_loss +
                        self.config.entropy_coef * entropy_loss
                    )
                
                # === MIXED PRECISION BACKWARD PASS ===
                self.optimizer.zero_grad(set_to_none=True)  # Slightly faster
                self.scaler.scale(loss).backward()
                
                # Unscale before clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.config.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                
                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean()
                    clip_frac = ((ratio - 1.0).abs() > self.config.clip_epsilon).float().mean()
                
                metrics['policy_loss'] += policy_loss.item()
                metrics['value_loss'] += value_loss.item()
                metrics['entropy'] += entropies.mean().item()
                metrics['approx_kl'] += approx_kl.item()
                metrics['clip_fraction'] += clip_frac.item()
                metrics['grad_norm'] += grad_norm.item()
                num_updates += 1
        
        for key in metrics:
            metrics[key] /= max(num_updates, 1)
        
        self.updates += 1
        return metrics

    def _vectorized_log_probs(self, outputs, action_types, sub_actions):
        """
        Compute log probs WITHOUT a per-sample Python loop.
        
        This is the key optimization over the original [4] which had:
            for b in range(batch_size):
                at = action_types[b].item()
                ...
        
        Instead, we compute all sub-action logits for all types, then
        gather the correct ones based on action_type.
        """
        batch_size = action_types.size(0)
        device = action_types.device
        
        # 1. Action type log probs (already vectorized)
        type_dist = torch.distributions.Categorical(logits=outputs['action_type_logits'])
        type_log_probs = type_dist.log_prob(action_types)
        type_entropy = type_dist.entropy()
        
        # 2. Sub-action log probs - compute for all types, then select
        # Stack all sub-action logit tensors, padding to same size
        all_logits = {
            0: outputs['cube_move_logits'],      # [B, num_cube_actions]
            1: outputs['separation_logits'],     # [B, num_separation_actions]
            2: outputs['docking_logits'],        # [B, num_docking_actions]
            3: outputs['maneuver_logits'],       # [B, num_maneuver_actions]
        }
        
        sub_log_probs = torch.zeros(batch_size, device=device)
        sub_entropy = torch.zeros(batch_size, device=device)
        
        # Process each action type as a masked batch (much faster than per-sample)
        for at_idx in range(4):
            mask = (action_types == at_idx)
            if not mask.any():
                continue
            
            logits = all_logits[at_idx][mask]  # [num_matching, action_dim]
            actions = sub_actions[mask]         # [num_matching]
            
            # Clamp actions to valid range
            max_valid = logits.size(-1) - 1
            if (actions > max_valid).any():
                # This means the buffer stored an action that's now out of range
                # (e.g. after curriculum changed num_cubes). Zero them out.
                actions = actions.clamp(0, max_valid)
            
            dist = torch.distributions.Categorical(logits=logits)
            sub_log_probs[mask] = dist.log_prob(actions)
            sub_entropy[mask] = dist.entropy()
        
        # Noop (action_type == 4) has log_prob = 0, entropy = 0
        total_log_probs = type_log_probs + sub_log_probs
        total_entropy = type_entropy + sub_entropy
        
        return total_log_probs, total_entropy
    
    def _collate_action_masks(self, masks_list: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate action masks from multiple timesteps into batched tensors.
        
        Each mask was built with a leading dim of 1 (batch_size=1 at collection time).
        We torch.cat along dim=0 to get [minibatch_size, ...].
        
        Non-tensor entries (e.g. _separation_actions, _docking_actions) are skipped.
        Missing tensor entries are replaced with an all-False mask of the correct shape
        so every sample is represented in the batch.
        """
        if not masks_list:
            return {}
        
        collated = {}
        
        # Determine which keys hold tensors (skip private/_-prefixed action lists)
        tensor_keys = [
            k for k, v in masks_list[0].items()
            if isinstance(v, torch.Tensor) and not k.startswith('_')
        ]
        
        for key in tensor_keys:
            tensors = []
            reference_shape = None  # shape excluding dim-0

            for m in masks_list:
                val = m.get(key)
                if isinstance(val, torch.Tensor):
                    tensors.append(val)
                    if reference_shape is None:
                        reference_shape = val.shape[1:]  # drop the leading-1 dim
                else:
                    # Missing entry: append a correctly-shaped all-False placeholder
                    if reference_shape is not None:
                        placeholder = torch.zeros(
                            (1, *reference_shape), dtype=torch.bool
                        )
                        tensors.append(placeholder)
                    # If reference_shape is still unknown we'll pick it up on a later iter

            if tensors:
                collated[key] = torch.cat(tensors, dim=0).to(self.device)

        return collated
        
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': self.config,
            'total_steps': self.total_steps,
            'updates': self.updates,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.updates = checkpoint.get('updates', 0)
