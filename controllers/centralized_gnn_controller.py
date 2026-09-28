# controllers/centralized_gnn_controller.py
"""
Centralized GNN controller for evaluation.

Loads a trained GNNPolicy checkpoint and produces a joint action from the
GLOBAL observation. "Centralized" here means:
  - node features carry full module state (from the global observation),
  - the policy uses enough message-passing layers that information propagates
    across the whole spacecraft graph.

Handles any N (no fixed size, no padding). Acts deterministically at eval time.
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from controllers.base import Controller
from controllers.gnn_policy import GNNPolicy
from rl.graph_batch import build_graph_from_global_obs, collate_graphs, node_feature_dim, EDGE_FEATURE_DIM
from rl.action_masking import node_action_mask


class CentralizedGNNController(Controller):
    def __init__(self, env, checkpoint: str, device: str = "cpu",
                 deterministic: bool = True):
        self.env = env
        self.device = device
        self.deterministic = deterministic

        ckpt = torch.load(checkpoint, map_location=device)
        cfg = ckpt["train_config"]
        self.policy = GNNPolicy(
            node_dim=ckpt["node_dim"],
            edge_dim=ckpt["edge_dim"],
            hidden=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("n_layers", 3),
        ).to(device)
        self.policy.load_state_dict(ckpt["state_dict"])
        self.policy.eval()

    def reset(self):
        pass

    def act(self, observation) -> Dict[int, int]:
        graph = build_graph_from_global_obs(observation)
        (nf, ei, ef, batch, sizes) = collate_graphs([graph])
        mask = node_action_mask(self.env, graph.id_order)

        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        with torch.no_grad():
            actions, _, _ = self.policy.act(
                t(nf, torch.float32), t(ei, torch.int64), t(ef, torch.float32),
                t(batch, torch.int64), num_graphs=1,
                mask=t(mask, torch.float32),
                deterministic=self.deterministic,
            )
        actions_np = actions.cpu().numpy()
        return {mid: int(actions_np[i]) for i, mid in enumerate(graph.id_order)}