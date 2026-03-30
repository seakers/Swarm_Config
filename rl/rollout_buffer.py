"""
rollout_buffer.py
=================
PPO rollout buffer for the hierarchical action space.

Stores transitions collected during env rollouts and computes GAE-lambda
advantages.  Designed for a single (non-vectorised) environment; extend with
a VecEnv wrapper for parallel collection.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional


@dataclass
class Transition:
    """Single environment transition."""
    obs             : np.ndarray
    action_type     : int
    sub_action      : int
    log_prob        : float
    value           : float
    reward          : float
    done            : bool
    # raw masks stored as numpy for compact storage
    masks           : Dict[str, np.ndarray]


class RolloutBuffer:
    """
    Fixed-size rollout buffer with GAE-lambda advantage estimation.

    Parameters
    ----------
    capacity    : max transitions to store per rollout
    gamma       : discount factor
    gae_lambda  : GAE smoothing parameter
    device      : torch device for returned tensors
    """

    def __init__(self,
                 capacity  : int   = 2048,
                 gamma     : float = 0.99,
                 gae_lambda: float = 0.95,
                 device    : str   = 'cpu'):
        self.capacity   = capacity
        self.gamma      = gamma
        self.gae_lambda = gae_lambda
        self.device     = torch.device(device)
        self._transitions: List[Transition] = []

    # ──────────────────────────────────────────────────────────────────────
    # Collection
    # ──────────────────────────────────────────────────────────────────────

    def reset(self):
        self._transitions.clear()

    def add(self, t: Transition):
        self._transitions.append(t)

    def is_full(self) -> bool:
        return len(self._transitions) >= self.capacity

    def __len__(self):
        return len(self._transitions)

    # ──────────────────────────────────────────────────────────────────────
    # Advantage computation
    # ──────────────────────────────────────────────────────────────────────

    def compute_advantages(self, last_value: float = 0.0) -> None:
        """
        Compute GAE-lambda advantages and returns in-place.
        Must be called once per rollout before iterating.
        """
        n = len(self._transitions)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            tr = self._transitions[t]
            if t == n - 1:
                next_non_terminal = 1.0 - float(tr.done)
                next_value        = last_value
            else:
                next_tr           = self._transitions[t + 1]
                next_non_terminal = 1.0 - float(tr.done)
                next_value        = next_tr.value

            delta    = tr.reward + self.gamma * next_value * next_non_terminal - tr.value
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        self._advantages = advantages
        self._returns    = advantages + np.array([t.value for t in self._transitions],
                                                  dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Mini-batch sampling
    # ──────────────────────────────────────────────────────────────────────

    def get_batches(self,
                    batch_size: int,
                    ) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yield random mini-batches as dicts of tensors.

        Note: mask tensors are *not* returned here because they vary in
        structure per transition.  The trainer re-builds them from the stored
        mask arrays as needed (or trains with a simplified flat obs approach).
        """
        n      = len(self._transitions)
        indices = np.random.permutation(n)

        action_types = np.array([t.action_type for t in self._transitions],
                                 dtype=np.int64)
        sub_actions  = np.array([t.sub_action  for t in self._transitions],
                                 dtype=np.int64)
        log_probs    = np.array([t.log_prob    for t in self._transitions],
                                 dtype=np.float32)
        advantages   = self._advantages
        returns      = self._returns

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std  = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        for start in range(0, n, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                'action_types': torch.tensor(action_types[idx], device=self.device),
                'sub_actions' : torch.tensor(sub_actions[idx],  device=self.device),
                'old_log_probs': torch.tensor(log_probs[idx],   device=self.device),
                'advantages'  : torch.tensor(advantages[idx],   device=self.device),
                'returns'     : torch.tensor(returns[idx],       device=self.device),
                'indices'     : idx,   # so trainer can look up transitions
            }

    # ──────────────────────────────────────────────────────────────────────
    # Convenience accessors
    # ──────────────────────────────────────────────────────────────────────

    def get_transition(self, idx: int) -> Transition:
        return self._transitions[idx]

    @property
    def mean_reward(self) -> float:
        return float(np.mean([t.reward for t in self._transitions]))

    @property
    def mean_value(self) -> float:
        return float(np.mean([t.value for t in self._transitions]))