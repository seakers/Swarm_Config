# rl/graph_batch.py
"""
Convert environment graph observations into padded/packed tensors for the GNN.

We use the "disjoint batch" trick: multiple graphs (of possibly different sizes)
are concatenated into one big graph with a block-diagonal edge structure and a
`batch` vector mapping each node to its graph index. This lets a single forward
pass process a whole minibatch of variable-N spacecraft.

Node features and edge features are built here from the environment's global
observation. Mission is appended to every node (per-node conditioning).
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from rl.encoders import module_feature_vector, module_feature_dim
from environment.geometry import dir_index

# Mission conditioning: fixed ordering so the one-hot is stable.
MISSION_LIST = ["power", "thermal", "comms", "aperture"]
NUM_MISSIONS = len(MISSION_LIST)
MISSION_INDEX = {m: i for i, m in enumerate(MISSION_LIST)}

# Edge features: relative position (3) + connection-face one-hot (6).
EDGE_FEATURE_DIM = 3 + 6


def mission_one_hot(mission: str) -> np.ndarray:
    v = np.zeros(NUM_MISSIONS, dtype=np.float32)
    v[MISSION_INDEX[mission]] = 1.0
    return v


def node_feature_dim() -> int:
    # module features + mission one-hot + sun_dir (3) + earth_dir (3)
    return module_feature_dim() + NUM_MISSIONS + 6


def _edge_feature(rel_pos: Tuple[int, int, int]) -> np.ndarray:
    """Edge feature from the relative position of the neighbor."""
    rel = np.asarray(rel_pos, dtype=np.float32)
    face = np.zeros(6, dtype=np.float32)
    face[dir_index(tuple(int(v) for v in rel_pos))] = 1.0
    return np.concatenate([rel, face]).astype(np.float32)


class GraphTensors:
    """A single graph's tensor representation (numpy; converted to torch later)."""
    def __init__(self, node_features: np.ndarray,
                 edge_index: np.ndarray, edge_features: np.ndarray,
                 id_order: List[int]):
        self.node_features = node_features        # [N, node_dim]
        self.edge_index = edge_index              # [2, E] (src, dst)
        self.edge_features = edge_features        # [E, edge_dim]
        self.id_order = id_order                  # row -> module_id


def build_graph_from_global_obs(global_obs: dict) -> GraphTensors:
    """Build a graph from the environment's GLOBAL observation.

    Node features use full module state (centralized view). Mission is appended
    to every node.
    """
    modules = global_obs["modules"]
    adjacency = global_obs["adjacency"]
    mission = global_obs["mission"]

    id_order = sorted(m["id"] for m in modules)
    id_to_row = {mid: i for i, mid in enumerate(id_order)}
    module_by_id = {m["id"]: m for m in modules}
    pos_by_id = {m["id"]: tuple(int(v) for v in m["position"]) for m in modules}

    mvec = mission_one_hot(mission)
    sun = np.asarray(global_obs["sun_direction"], dtype=np.float32)
    earth = np.asarray(global_obs["earth_direction"], dtype=np.float32)
    env_ctx = np.concatenate([mvec, sun, earth]).astype(np.float32)

    node_feats = np.stack([
        np.concatenate([module_feature_vector(module_by_id[mid]), env_ctx])
        for mid in id_order
    ]).astype(np.float32)

    src, dst, efeats = [], [], []
    for mid in id_order:
        for nb in adjacency[mid]:
            src.append(id_to_row[mid])
            dst.append(id_to_row[nb])
            rel = tuple(pos_by_id[nb][k] - pos_by_id[mid][k] for k in range(3))
            efeats.append(_edge_feature(rel))

    edge_index = (np.array([src, dst], dtype=np.int64)
                  if src else np.zeros((2, 0), dtype=np.int64))
    edge_features = (np.stack(efeats).astype(np.float32)
                     if efeats else np.zeros((0, EDGE_FEATURE_DIM),
                                             dtype=np.float32))
    return GraphTensors(node_feats, edge_index, edge_features, id_order)


def collate_graphs(graphs: List[GraphTensors]):
    """Pack a list of graphs into one disjoint batch.

    Returns numpy arrays:
      node_features [sum_N, node_dim]
      edge_index    [2, sum_E]   (offset into the concatenated node list)
      edge_features [sum_E, edge_dim]
      batch         [sum_N]      graph index per node
      graph_sizes   [B]          number of nodes per graph
    """
    node_chunks, edge_chunks, ef_chunks, batch_chunks = [], [], [], []
    offset = 0
    graph_sizes = []
    for gi, g in enumerate(graphs):
        n = g.node_features.shape[0]
        node_chunks.append(g.node_features)
        if g.edge_index.shape[1] > 0:
            edge_chunks.append(g.edge_index + offset)
            ef_chunks.append(g.edge_features)
        batch_chunks.append(np.full(n, gi, dtype=np.int64))
        graph_sizes.append(n)
        offset += n

    node_features = np.concatenate(node_chunks, axis=0)
    edge_index = (np.concatenate(edge_chunks, axis=1)
                  if edge_chunks else np.zeros((2, 0), dtype=np.int64))
    edge_features = (np.concatenate(ef_chunks, axis=0)
                     if ef_chunks else np.zeros((0, EDGE_FEATURE_DIM),
                                                dtype=np.float32))
    batch = np.concatenate(batch_chunks, axis=0)
    return (node_features, edge_index, edge_features, batch,
            np.array(graph_sizes, dtype=np.int64))