# rl/gnn_buffer.py
"""
Rollout buffer for the GNN policy.

Unlike the flat MLP buffer, each transition stores a full graph (variable N and
edge count) alongside the scalar RL quantities. For PPO updates we collate the
stored graphs into disjoint batches per minibatch.
"""

from __future__ import annotations
from typing import List
import numpy as np

from rl.graph_batch import GraphTensors, collate_graphs


class GNNRolloutBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.graphs: List[GraphTensors] = []
        self.masks: List[np.ndarray] = []       # per-step [N, A]
        self.actions: List[np.ndarray] = []     # per-step [N]
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.dones: List[float] = []
        self.advantages: np.ndarray = None
        self.returns: np.ndarray = None

    def __len__(self):
        return len(self.graphs)

    def add(self, graph, mask, action, log_prob, reward, value, done):
        self.graphs.append(graph)
        self.masks.append(mask)
        self.actions.append(action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(float(done))

    def compute_gae(self, last_value: float, gamma: float, lam: float):
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

    def minibatches(self, minibatch_size: int, normalize_adv: bool = True):
        """Yield collated minibatches of (packed graph tensors + targets)."""
        n = len(self)
        adv = self.advantages.copy()
        if normalize_adv and n > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        idx = np.arange(n)
        np.random.shuffle(idx)
        for start in range(0, n, minibatch_size):
            mb = idx[start: start + minibatch_size]
            graphs = [self.graphs[i] for i in mb]
            nf, ei, ef, batch, sizes = collate_graphs(graphs)
            # Stack per-node actions/masks in the same node order as collate.
            actions = np.concatenate([self.actions[i] for i in mb], axis=0)
            masks = np.concatenate([self.masks[i] for i in mb], axis=0)
            yield {
                "node_features": nf,
                "edge_index": ei,
                "edge_features": ef,
                "batch": batch,
                "num_graphs": len(mb),
                "actions": actions,
                "masks": masks,
                "old_log_probs": self.advantages_index(self.log_probs, mb),
                "advantages": adv[mb],
                "returns": self.returns[mb],
            }

    @staticmethod
    def advantages_index(seq, mb):
        return np.array([seq[i] for i in mb], dtype=np.float32)