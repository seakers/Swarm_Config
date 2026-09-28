# controllers/decentralized_gnn_controller.py
"""
Decentralized GNN controller for EXECUTION.

Loads only the trained LOCAL actor. It consumes env.all_local_observations()
exclusively; it never accesses the global observation. This is the enforcement
point for "no global information leaks into decentralized execution" (spec §20).
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import torch

from controllers.base import Controller
from controllers.gnn_policy import GNNPolicy
from rl.local_graph import build_local_graph, node_feature_dim_local
from rl.graph_batch import collate_graphs, EDGE_FEATURE_DIM
from rl.action_masking import node_action_mask


class DecentralizedGNNController(Controller):
    def __init__(self, env, checkpoint: str, device: str = "cpu",
                 deterministic: bool = True):
        self.env = env
        self.device = device
        self.deterministic = deterministic

        ckpt = torch.load(checkpoint, map_location=device)
        cfg = ckpt["train_config"]
        self.actor = GNNPolicy(
            node_dim=ckpt["actor_node_dim"], edge_dim=ckpt["edge_dim"],
            hidden=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("actor_layers", 2),
        ).to(device)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.actor.eval()

    def reset(self):
        pass

    def act(self, observation=None) -> Dict[int, int]:
        # IMPORTANT: we deliberately ignore `observation` (the global obs) and
        # pull ONLY local observations. This is the enforcement point that
        # prevents global state from influencing decentralized execution.
        all_local = self.env.all_local_observations()
        mission = self.env.mission.config.primary_objective

        graph = build_local_graph(all_local, mission, self.env.mission.config.sun_direction, self.env.mission.config.earth_direction)
        nf, ei, ef, batch, _ = collate_graphs([graph])
        mask = node_action_mask(self.env, graph.id_order)

        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=self.device)
        h = self.actor.embed(t(nf, torch.float32), t(ei, torch.int64),
                             t(ef, torch.float32))
        logits = self.actor.node_logits(h)
        dist = self.actor._masked_dist(logits, t(mask, torch.float32))
        with torch.no_grad():
            actions = (torch.argmax(dist.logits, dim=-1)
                       if self.deterministic else dist.sample())
        actions_np = actions.cpu().numpy()
        return {mid: int(actions_np[i]) for i, mid in enumerate(graph.id_order)}