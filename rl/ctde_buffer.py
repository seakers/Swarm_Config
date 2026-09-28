# rl/ctde_buffer.py
"""
Rollout buffer for CTDE training.

Each transition stores BOTH:
  - the LOCAL graph (actor input),
  - the GLOBAL graph (centralized critic input),
plus per-node actions/masks and scalar RL quantities. GAE uses centralized-
critic values.
"""

from __future__ import annotations
from typing import List
import numpy as np

from rl.graph_batch import GraphTensors, collate_graphs


class CTDERolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.local_graphs: List[GraphTensors] = []
        self.global_graphs: List[GraphTensors] = []
        self.masks: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []
        self.advantages = None
        self.returns = None

    def __len__(self):
        return len(self.local_graphs)

    def add(self, local_graph, global_graph, mask, action, log_prob,
            reward, value, done):
        self.local_graphs.append(local_graph)
        self.global_graphs.append(global_graph)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def compute_gae(self, last_value, gamma, lam):
        n = len(self)
        adv = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = np.array(self.values, dtype=np.float32)
        rewards = np.array(self.rewards, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
            gae = delta + gamma * lam * next_nonterminal * gae
            adv[t] = gae
        self.advantages = adv
        self.returns = adv + values

    def minibatches(self, minibatch_size, normalize_adv=True):
        n = len(self)
        adv = self.advantages.copy()
        if normalize_adv and n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start: start + minibatch_size]
            local = [self.local_graphs[i] for i in mb]
            glob = [self.global_graphs[i] for i in mb]
            lnf, lei, lef, lbatch, _ = collate_graphs(local)
            gnf, gei, gef, gbatch, _ = collate_graphs(glob)
            actions = np.concatenate([self.actions[i] for i in mb], axis=0)
            masks = np.concatenate([self.masks[i] for i in mb], axis=0)
            yield {
                # actor (local)
                "l_nf": lnf, "l_ei": lei, "l_ef": lef, "l_batch": lbatch,
                # critic (global)
                "g_nf": gnf, "g_ei": gei, "g_ef": gef, "g_batch": gbatch,
                "num_graphs": len(mb),
                "actions": actions, "masks": masks,
                "old_log_probs": np.array([self.log_probs[i] for i in mb],
                                          dtype=np.float32),
                "advantages": adv[mb],
                "returns": self.returns[mb],
            }