# controllers/gnn_policy.py
"""
Hand-rolled message-passing GNN policy + value network.

Shared weights across all nodes (permutation-invariant, handles any N).

Message passing (L rounds):
    m_ij   = message_mlp([h_i, h_j, e_ij])
    m_i    = mean_j m_ij           (mean aggregation over neighbors)
    h_i'   = update_mlp([h_i, m_i])

The number of layers L controls information locality:
    L layers => a node's embedding depends on its L-hop neighborhood.
Small L  -> decentralized (local) reasoning.
Large L  -> centralized (whole-graph) reasoning.

Heads:
    actor:  per-node action logits [num_nodes, NUM_ACTIONS]
    critic: a value. For the centralized critic we pool node embeddings per
            graph (mean) to a single value; the policy exposes both node-level
            and graph-level outputs.

The joint action distribution is factored: independent Categorical per node,
joint log-prob = sum of per-node log-probs (masked to legal actions).
"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.transitions import NUM_ACTIONS


def _mlp(in_dim, hidden, out_dim, n_hidden_layers=1):
    layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(n_hidden_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, out_dim)]
    return nn.Sequential(*layers)


class MessagePassingLayer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden):
        super().__init__()
        self.message = _mlp(2 * node_dim + edge_dim, hidden, hidden)
        self.update = _mlp(node_dim + hidden, hidden, node_dim)

    def forward(self, h, edge_index, edge_features, num_nodes):
        """h: [N, node_dim]; edge_index: [2, E] (src, dst)."""
        if edge_index.shape[1] == 0:
            # No edges: aggregated message is zero; update from self only.
            agg = torch.zeros(num_nodes, self.message[-1].out_features,
                              device=h.device, dtype=h.dtype)
        else:
            src, dst = edge_index[0], edge_index[1]
            # Message from src -> dst.
            msg_in = torch.cat([h[dst], h[src], edge_features], dim=-1)
            m = self.message(msg_in)                       # [E, hidden]
            # Mean-aggregate messages at each destination node.
            agg = torch.zeros(num_nodes, m.shape[-1],
                              device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, m)
            deg = torch.zeros(num_nodes, 1, device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones(dst.shape[0], 1,
                                              device=h.device, dtype=h.dtype))
            agg = agg / deg.clamp(min=1.0)
        h_new = self.update(torch.cat([h, agg], dim=-1))
        return h + h_new   # residual connection for stable deep MP


class GNNPolicy(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden=128,
                 n_layers=3, num_actions=NUM_ACTIONS):
        super().__init__()
        self.n_layers = n_layers
        self.num_actions = num_actions

        self.encoder = _mlp(node_dim, hidden, hidden)
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden, edge_dim, hidden)
            for _ in range(n_layers)
        ])
        self.actor_head = _mlp(hidden, hidden, num_actions)
        self.critic_head = _mlp(hidden, hidden, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        # Small policy-head init.
        for m in self.actor_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)

    def embed(self, node_features, edge_index, edge_features):
        num_nodes = node_features.shape[0]
        h = self.encoder(node_features)
        for layer in self.mp_layers:
            h = layer(h, edge_index, edge_features, num_nodes)
        return h  # [N, hidden]

    def node_logits(self, h):
        """Per-node action logits: [N, num_actions]."""
        return self.actor_head(h)

    def graph_value(self, h, batch, num_graphs):
        """Mean-pool node embeddings per graph -> critic value [num_graphs]."""
        pooled = torch.zeros(num_graphs, h.shape[-1],
                             device=h.device, dtype=h.dtype)
        pooled.index_add_(0, batch, h)
        counts = torch.zeros(num_graphs, 1, device=h.device, dtype=h.dtype)
        counts.index_add_(0, batch, torch.ones(batch.shape[0], 1,
                                               device=h.device, dtype=h.dtype))
        pooled = pooled / counts.clamp(min=1.0)
        return self.critic_head(pooled).squeeze(-1)  # [num_graphs]

    # -------------------------------------------------- distribution helpers
    @staticmethod
    def _masked_dist(logits, mask):
        if mask is not None:
            neg_inf = torch.finfo(logits.dtype).min
            logits = torch.where(mask > 0.5, logits,
                                 torch.full_like(logits, neg_inf))
        return Categorical(logits=logits)

    def act(self, node_features, edge_index, edge_features, batch,
            num_graphs, mask=None, deterministic=False):
        """Sample per-node actions.

        Returns:
          actions   [N]        per-node action index
          log_prob  [num_graphs]  summed joint log-prob per graph
          value     [num_graphs]
        """
        h = self.embed(node_features, edge_index, edge_features)
        logits = self.node_logits(h)                    # [N, A]
        dist = self._masked_dist(logits, mask)
        actions = (torch.argmax(dist.logits, dim=-1)
                   if deterministic else dist.sample())  # [N]
        node_lp = dist.log_prob(actions)                 # [N]
        # Sum per-node log-probs within each graph -> joint log-prob per graph.
        log_prob = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        log_prob.index_add_(0, batch, node_lp)
        value = self.graph_value(h, batch, num_graphs)
        return actions, log_prob, value

    def evaluate(self, node_features, edge_index, edge_features, batch,
                 num_graphs, actions, mask=None):
        """Evaluate given per-node actions (for PPO update).

        Returns:
          log_prob [num_graphs], entropy [num_graphs], value [num_graphs]
        """
        h = self.embed(node_features, edge_index, edge_features)
        logits = self.node_logits(h)
        dist = self._masked_dist(logits, mask)
        node_lp = dist.log_prob(actions)                 # [N]
        node_ent = dist.entropy()                        # [N]

        log_prob = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        log_prob.index_add_(0, batch, node_lp)
        entropy = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        entropy.index_add_(0, batch, node_ent)
        value = self.graph_value(h, batch, num_graphs)
        return log_prob, entropy, value