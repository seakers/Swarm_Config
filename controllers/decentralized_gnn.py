# controllers/decentralized_gnn.py
"""
Decentralized GNN actor + centralized critic (CTDE).

ACTOR (used at both training and execution):
  - node features from LOCAL observations only (own + mission),
  - few message-passing layers (bounded communication radius),
  - outputs per-node action logits.

CRITIC (used ONLY during training):
  - node features from the GLOBAL observation,
  - more message-passing layers (whole-graph value estimate),
  - outputs a single graph-level value.

At execution the critic is discarded; only the local actor runs. This makes it
structurally impossible for global state to influence action selection.
"""

from __future__ import annotations
import numpy as np
import torch

from controllers.gnn_policy import GNNPolicy
from rl.local_graph import node_feature_dim_local
from rl.graph_batch import node_feature_dim, EDGE_FEATURE_DIM


class DecentralizedActorCritic:
    """Container holding the local actor and (training-only) centralized critic."""

    def __init__(self, hidden_dim=128, actor_layers=2, critic_layers=4,
                 device="cpu"):
        self.device = device
        # Actor: local features, few MP layers.
        self.actor = GNNPolicy(
            node_dim=node_feature_dim_local(), edge_dim=EDGE_FEATURE_DIM,
            hidden=hidden_dim, n_layers=actor_layers,
        ).to(device)
        # Critic: global features, many MP layers. We reuse GNNPolicy but only
        # consume its graph_value output.
        self.critic = GNNPolicy(
            node_dim=node_feature_dim(), edge_dim=EDGE_FEATURE_DIM,
            hidden=hidden_dim, n_layers=critic_layers,
        ).to(device)

    def parameters(self):
        return list(self.actor.parameters()) + list(self.critic.parameters())

    # ---- Actor: sample/evaluate from LOCAL graph ----
    def act(self, nf, ei, ef, batch, num_graphs, mask=None,
            deterministic=False):
        """Local-only action selection. Value comes from the actor's own
        pooled critic head is NOT used for CTDE; value is provided separately
        by the centralized critic. Here we return actions + log_prob only."""
        h = self.actor.embed(nf, ei, ef)
        logits = self.actor.node_logits(h)
        dist = self.actor._masked_dist(logits, mask)
        actions = (torch.argmax(dist.logits, dim=-1)
                   if deterministic else dist.sample())
        node_lp = dist.log_prob(actions)
        log_prob = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        log_prob.index_add_(0, batch, node_lp)
        return actions, log_prob

    def evaluate_actor(self, nf, ei, ef, batch, num_graphs, actions, mask=None):
        h = self.actor.embed(nf, ei, ef)
        logits = self.actor.node_logits(h)
        dist = self.actor._masked_dist(logits, mask)
        node_lp = dist.log_prob(actions)
        node_ent = dist.entropy()
        log_prob = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        log_prob.index_add_(0, batch, node_lp)
        entropy = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        entropy.index_add_(0, batch, node_ent)
        return log_prob, entropy

    # ---- Critic: value from GLOBAL graph ----
    def value(self, nf, ei, ef, batch, num_graphs):
        h = self.critic.embed(nf, ei, ef)
        return self.critic.graph_value(h, batch, num_graphs)

    def save(self, path, cfg_dict):
        torch.save({
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "actor_node_dim": node_feature_dim_local(),
            "critic_node_dim": node_feature_dim(),
            "edge_dim": EDGE_FEATURE_DIM,
            "train_config": cfg_dict,
        }, path)