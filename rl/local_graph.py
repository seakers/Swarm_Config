# rl/local_graph.py
"""
Build the spacecraft graph from LOCAL observations only.

Decentralized execution constraint (spec §8, §20, §40):
  each node may use ONLY its own state and information from physically
  connected neighbors. No global state, no distant-module info, no full graph
  beyond the local neighborhood implied by message passing.

This builder derives node features from env.all_local_observations(), which by
construction contains only own + neighbor data. Distant modules' internal
states never enter the tensors.

Node features (LOCAL):
  own module features (same layout as centralized node features)
  + mission one-hot (allowed globally per spec §8)
Edge features:
  neighbor relative position (3) + connection-face one-hot (6)   [same as global]

Topology: an edge (i, j) exists iff j is a physically connected neighbor of i,
which each node knows locally. The resulting edge set is identical to the
global graph's, but no global information is used to build node FEATURES.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from rl.encoders import module_feature_vector, module_feature_dim
from rl.graph_batch import (
    GraphTensors, mission_one_hot, NUM_MISSIONS, EDGE_FEATURE_DIM, _edge_feature,
)


def node_feature_dim_local() -> int:
    # Same as centralized node feature dim (own features + mission one-hot + sun_dir + earth_dir).
    return module_feature_dim() + NUM_MISSIONS + 6


def _own_module_state_from_local(local_obs: dict) -> dict:
    """Reconstruct the module-state dict expected by module_feature_vector
    from a local observation's 'own' block."""
    own = local_obs["own"]
    return {
        "position": own["position"],
        "orientation": own["orientation"],
        "face_capabilities": own["face_capabilities"],
        "battery": own["battery"],
        "temperature": own["temperature"],
        "power_generation": own["power_generation"],
    }


def build_local_graph(all_local_obs: Dict[int, dict],
                      mission: str, sun_direction: List[float], earth_direction: List[float]) -> GraphTensors:
    """Build a graph where node features come ONLY from local observations.

    Args:
        all_local_obs: {module_id: local_observation_dict}
                       (from env.all_local_observations()).
        mission: the mission name (globally known, allowed per spec).
        sun_direction: the direction of the sun (3D vector).
        earth_direction: the direction of the earth (3D vector).
    """
    mvec = mission_one_hot(mission)
    sun = np.asarray(sun_direction, dtype=np.float32)
    earth = np.asarray(earth_direction, dtype=np.float32)
    env_ctx = np.concatenate([mvec, sun, earth]).astype(np.float32)
    id_order = sorted(all_local_obs.keys())
    id_to_row = {mid: i for i, mid in enumerate(id_order)}

    # Node features: OWN state only, + mission conditioning.
    node_feats = np.stack([
        np.concatenate([
            module_feature_vector(_own_module_state_from_local(all_local_obs[mid])),
            env_ctx,
        ])
        for mid in id_order
    ]).astype(np.float32)

    # Edges + edge features derived from each node's neighbor list (local).
    src, dst, efeats = [], [], []
    for mid in id_order:
        for nb in all_local_obs[mid]["neighbors"]:
            nb_id = nb["id"]
            if nb_id not in id_to_row:
                continue
            src.append(id_to_row[mid])
            dst.append(id_to_row[nb_id])
            rel = tuple(int(v) for v in nb["relative_position"])
            efeats.append(_edge_feature(rel))

    edge_index = (np.array([src, dst], dtype=np.int64)
                  if src else np.zeros((2, 0), dtype=np.int64))
    edge_features = (np.stack(efeats).astype(np.float32)
                     if efeats else np.zeros((0, EDGE_FEATURE_DIM),
                                             dtype=np.float32))
    return GraphTensors(node_feats, edge_index, edge_features, id_order)