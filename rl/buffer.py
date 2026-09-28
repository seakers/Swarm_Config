# rl/buffer.py
"""
Rollout buffer with Generalized Advantage Estimation (GAE) for PPO.

Stores fixed-size transitions and computes advantages/returns after a rollout.
Handles the multi-discrete action layout (actions and masks are per-module).
"""

from __future__ import annotations
import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, size: int, obs_dim: int, n_modules: int,
                 num_actions: int, device: str = "cpu"):
        self.size = size
        self.n_modules = n_modules
        self.num_actions = num_actions
        self.device = device

        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.masks = np.zeros((size, n_modules, num_actions), dtype=np.float32)
        self.actions = np.zeros((size, n_modules), dtype=np.int64)
        self.log_probs = np.zeros(size, dtype=np.float32)
        self.rewards = np.zeros(size, dtype=np.float32)
        self.values = np.zeros(size, dtype=np.float32)
        self.dones = np.zeros(size, dtype=np.float32)

        self.advantages = np.zeros(size, dtype=np.float32)
        self.returns = np.zeros(size, dtype=np.float32)

        self.ptr = 0

    def add(self, obs, mask, action, log_prob, reward, value, done):
        i = self.ptr
        self.obs[i] = obs
        self.masks[i] = mask
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = done
        self.ptr += 1

    def full(self) -> bool:
        return self.ptr >= self.size

    def reset(self):
        self.ptr = 0

# rl/buffer.py  (continued)

    def compute_gae(self, last_value: float, gamma: float, lam: float):
        """Compute GAE advantages and bootstrapped returns in place."""
        adv = 0.0
        for t in reversed(range(self.ptr)):
            next_value = last_value if t == self.ptr - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + gamma * next_value * next_nonterminal
                     - self.values[t])
            adv = delta + gamma * lam * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns[: self.ptr] = (
            self.advantages[: self.ptr] + self.values[: self.ptr]
        )

    def get_tensors(self, normalize_adv: bool = True):
        n = self.ptr
        adv = self.advantages[:n].copy()
        if normalize_adv and n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        to_t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        return {
            "obs": to_t(self.obs[:n], torch.float32),
            "masks": to_t(self.masks[:n], torch.float32),
            "actions": to_t(self.actions[:n], torch.int64),
            "log_probs": to_t(self.log_probs[:n], torch.float32),
            "advantages": to_t(adv, torch.float32),
            "returns": to_t(self.returns[:n], torch.float32),
            "values": to_t(self.values[:n], torch.float32),
        }