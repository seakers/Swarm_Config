"""
obs_builder.py
==============
Converts raw environment observations and info dicts (produced by
ConstellationEnv / HierarchicalConstellationEnv) into the PyTorch-Geometric
Data objects and mask tensors expected by SwarmGNN and HierarchicalPolicy.

The environment already computes a flat numpy observation and exposes valid
action lists via _valid_actions_cache / _get_action_mask().  This module is
the "bridge" between those raw structures and the GNN's graph inputs.

Usage
-----
    builder = ObservationBuilder(env)
    graph_data, masks = builder.build(obs, info)
    # graph_data is a PyG Data object
    # masks is a dict ready to pass to HierarchicalPolicy.forward()
"""

import numpy as np
import torch
from torch_geometric.data import Data
from typing import Dict, List, Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Feature helpers
# ──────────────────────────────────────────────────────────────────────────────

def _safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + eps)


def _orientation_6d(R: np.ndarray) -> np.ndarray:
    """6-D continuous rotation representation (first two columns of R)."""
    return R[:, :2].T.flatten()  # shape (6,)

class _SyntheticGroup:
    """Wraps a bare SwarmReconfigurationEnv as a single-group placeholder."""
    def __init__(self, env):
        self.group_id  = 0
        self.cube_ids  = list(range(env.num_cubes))
        self.position  = np.zeros(3, dtype=np.float32)
        self.velocity  = np.zeros(3, dtype=np.float32)
        class _Prop:
            remaining_delta_v = 0.0
            max_delta_v       = 1.0
        self.propulsion = _Prop()

# ──────────────────────────────────────────────────────────────────────────────
# Main builder class
# ──────────────────────────────────────────────────────────────────────────────

class ObservationBuilder:
    """
    Builds graph observations and action masks from a live ConstellationEnv.

    Graph node types:
      0 – cube nodes       (one per cube)
      1 – group super-nodes (one per active group)
      2 – global context node (one, contains mission/task info)

    Edges:
      cube–cube   : physical adjacency within each group
      cube–group  : membership edges (cube → its group super-node, bidirectional)
      group–group : communication links
      global–all  : broadcast edges from global node to every other node
    """

    # Dimensionalities that must match gnn_encoder.py defaults
    CUBE_FEAT_DIM   = 25
    GROUP_FEAT_DIM  = 12
    GLOBAL_FEAT_DIM = 16   # stored in global node's feature slot

    def __init__(self,
                 env,
                 max_cubes: int = 64,
                 max_groups: int = 8,
                 max_moves_per_cube: int = 24,
                 max_separation_actions: int = 100,
                 device: str = 'cpu'):
        self.env                    = env
        self.max_cubes              = max_cubes
        self.max_groups             = max_groups
        self.max_moves_per_cube     = max_moves_per_cube
        self.max_separation_actions = max_separation_actions
        self.device                 = torch.device(device)

    # ── public API ────────────────────────────────────────────────────────────

    def build(self, obs: np.ndarray, info: Dict) -> Tuple[Data, Dict]:
        """
        Build graph data and masks from env state.

        Note: obs is accepted for API symmetry but we read directly from
        env.swarm / env.constellation for richer structural information.
        The flat obs can be used as a fallback when sim objects are absent.

        Returns
        -------
        graph_data : torch_geometric.data.Data
        masks      : dict with keys used by HierarchicalPolicy.forward()
        """
        graph_data = self._build_graph()
        masks      = self._build_masks(info)
        return graph_data, masks

    # ── graph construction ────────────────────────────────────────────────────

    def _build_graph(self) -> Data:
        env   = self.env
        swarm = env.swarm

        # Support both SwarmReconfigurationEnv (no constellation)
        # and HierarchicalConstellationEnv (has constellation)
        const = getattr(env, 'constellation', None)

        if const is not None:
            groups = const.get_all_groups()
        else:
            # Fallback: treat the whole swarm as a single synthetic group
            groups = [_SyntheticGroup(env)]
        num_cubes   = env.num_cubes
        num_groups  = len(groups)

        # ── Node features ────────────────────────────────────────────────────
        # Layout: [cube_nodes | group_nodes | global_node]
        max_feat = max(self.CUBE_FEAT_DIM, self.GROUP_FEAT_DIM, self.GLOBAL_FEAT_DIM)

        cube_feats   = self._build_cube_features(swarm, const, groups)   # [N, 25]
        group_feats  = self._build_group_features(const, groups)          # [G, 12]
        global_feats = self._build_global_features(const, groups, env)   # [1, 16]

        # Pad all to same width
        def pad(x, w):
            p = w - x.shape[1]
            if p > 0:
                x = np.concatenate([x, np.zeros((x.shape[0], p))], axis=1)
            return x

        cube_feats  = pad(cube_feats,  max_feat)
        group_feats = pad(group_feats, max_feat)
        global_feats= pad(global_feats,max_feat)

        node_feats = np.concatenate([cube_feats, group_feats, global_feats], axis=0)

        # ── Node types ───────────────────────────────────────────────────────
        node_types = np.array(
            [0] * num_cubes +
            [1] * num_groups +
            [2],              # global
            dtype=np.int64,
        )

        # ── Edges ─────────────────────────────────────────────────────────────
        edge_src, edge_dst = [], []

        global_node_idx = num_cubes + num_groups

        # cube–cube adjacency (physical connections)
        for cid in range(num_cubes):
            cube = swarm.get_cube(cid)
            if cube is None:
                continue
            for neighbor_pos, neighbor_id in swarm._grid.get_neighbors(cube.position).items():
                edge_src.append(cid)
                edge_dst.append(neighbor_id)

        # cube–group membership (bidirectional)
        for g_idx, group in enumerate(groups):
            g_node_idx = num_cubes + g_idx
            for cube_id in group.cube_ids:
                if cube_id < num_cubes:
                    edge_src.append(cube_id)
                    edge_dst.append(g_node_idx)
                    edge_src.append(g_node_idx)
                    edge_dst.append(cube_id)

        # group–group communication
        if const is not None:
            comm_graph = const.get_communication_graph()
            for g_i, group_i in enumerate(groups):
                for g_j, group_j in enumerate(groups):
                    if g_i == g_j:
                        continue
                    if group_j.group_id in comm_graph.get(group_i.group_id, set()):
                        edge_src.append(num_cubes + g_i)
                        edge_dst.append(num_cubes + g_j)

        # global → all (broadcast)
        total_non_global = num_cubes + num_groups
        for i in range(total_non_global):
            edge_src.append(global_node_idx)
            edge_dst.append(i)
            edge_src.append(i)
            edge_dst.append(global_node_idx)

        if edge_src:
            edge_index = torch.tensor(
                [edge_src, edge_dst], dtype=torch.long, device=self.device
            )
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)

        data = Data(
            x         = torch.tensor(node_feats,  dtype=torch.float32, device=self.device),
            edge_index= edge_index,
            node_type = torch.tensor(node_types,  dtype=torch.long,    device=self.device),
            batch     = torch.zeros(len(node_types), dtype=torch.long, device=self.device),
        )

        return data

    # ── Feature builders ──────────────────────────────────────────────────────

    def _build_cube_features(self, swarm, const, groups) -> np.ndarray:
        """Returns [num_cubes, CUBE_FEAT_DIM]."""
        centroid = np.array(swarm.get_center_of_mass(), dtype=np.float32)
        feats    = np.zeros((self.env.num_cubes, self.CUBE_FEAT_DIM), dtype=np.float32)

        # Build cube → group lookup
        cube_to_group = {}
        for g in groups:
            for cid in g.cube_ids:
                cube_to_group[cid] = g

        for cid in range(self.env.num_cubes):
            cube = swarm.get_cube(cid)
            if cube is None:
                continue

            pos = np.array(cube.position, dtype=np.float32)
            rel = pos - centroid

            # Relative position (3)
            feats[cid, 0:3] = rel / (np.linalg.norm(rel) + 1.0)

            # Orientation 6-D (6)
            R = cube.orientation.matrix
            feats[cid, 3:9] = _orientation_6d(R)

            # Face alignment scores (4): solar/antenna/camera/radiator
            # These encode mission-relevant dot products
            face_data = getattr(cube, 'face_functions', None)
            feats[cid, 9:13] = 0.0  # zero-filled if not available

            # Face exposure mask (6) – whether each face is exposed to space
            feats[cid, 13:19] = 0.0

            # Delta-v remaining (1) – from group propulsion
            g = cube_to_group.get(cid)
            if g is not None:
                feats[cid, 19] = g.propulsion.remaining_delta_v / 50.0

            # Connectivity: num neighbours, is boundary (2)
            neighbours = swarm._grid.get_neighbors(pos)
            feats[cid, 20] = len(neighbours) / 6.0
            feats[cid, 21] = float(len(neighbours) < 6)

            # Group membership fraction (1)
            if g is not None:
                feats[cid, 22] = len(g.cube_ids) / max(1, self.env.num_cubes)

            # Cube id (normalised) – helps with ordering (1)
            feats[cid, 23] = cid / max(1, self.env.num_cubes - 1)

            # Placeholder (1)
            feats[cid, 24] = 0.0

        return feats

    def _build_group_features(self, const, groups) -> np.ndarray:
        """Returns [num_groups, GROUP_FEAT_DIM]."""
        feats = np.zeros((len(groups), self.GROUP_FEAT_DIM), dtype=np.float32)

        for g_idx, g in enumerate(groups):
            # Position (3) – normalised to km
            feats[g_idx, 0:3] = np.array(g.position, dtype=np.float32) / 1000.0

            # Velocity (3)
            feats[g_idx, 3:6] = np.array(g.velocity, dtype=np.float32)

            # Num cubes (1)
            feats[g_idx, 6] = len(g.cube_ids) / max(1, self.env.num_cubes)

            # Delta-v remaining (1)
            feats[g_idx, 7] = g.propulsion.remaining_delta_v / 50.0

            # Delta-v used fraction (1)
            max_dv = g.propulsion.max_delta_v
            used   = max_dv - g.propulsion.remaining_delta_v
            feats[g_idx, 8] = used / max(1e-6, max_dv)

            # Group id (normalised) (1)
            max_g = getattr(self.env, 'max_groups',
                    getattr(getattr(self.env, 'separation_reqs', None), 'max_groups', 8))
            feats[g_idx, 9] = g_idx / max(1, max_g - 1)

            # Is primary group (1)
            feats[g_idx, 10] = float(g_idx == 0)

            # Baseline to nearest other group (1)
            bl = const.get_max_baseline()
            feats[g_idx, 11] = bl / 100_000.0

        return feats

    def _build_global_features(self, const, groups, env) -> np.ndarray:
        """Returns [1, GLOBAL_FEAT_DIM]."""
        feats = np.zeros((1, self.GLOBAL_FEAT_DIM), dtype=np.float32)

        feats[0, 0]  = len(groups) / max(1, env.max_groups)
        feats[0, 1]  = const.get_time() / 3600.0
        feats[0, 2]  = const.get_total_delta_v_remaining() / (50.0 * env.num_cubes)
        feats[0, 3]  = float(const.is_constellation_connected())
        feats[0, 4]  = const.get_max_baseline() / 100_000.0
        feats[0, 5]  = env.task.get_progress(const)

        task_info = env.task.get_task_info()
        task_map = {
            'form_constellation': 0, 'rendezvous_and_dock': 1,
            'stereo_imaging': 2,     'multi_point_sensing': 3,
            # sim-side task types:
            'form_cube': 0, 'form_plane': 0, 'form_line': 0,
            'minimize_surface': 1,   'maximize_spread': 2,
        }
        task_type = task_info.get('task_type', task_info.get('mode', ''))
        feats[0, 6] = task_map.get(task_type, 0) / 4.0
        feats[0, 7] = task_info.get('target_num_groups', 
                                    task_info.get('target_size', 2)) / max(1, getattr(self.env, 'max_groups', 8))
        feats[0, 8]  = task_info.get('target_baseline', 0.0) / 100_000.0
        feats[0, 9]  = env.current_step / max(1, env.max_steps)

        return feats

    # ── Mask builders ─────────────────────────────────────────────────────────

    def _build_masks(self, info: Dict) -> Dict:
        """Build all action masks for HierarchicalPolicy.forward()."""
        env    = self.env
        cache  = getattr(env, '_valid_actions_cache', {})

        B = 1  # single environment

        # ── action type mask ─────────────────────────────────────────────────
        at_mask = np.zeros((B, 5), dtype=bool)
        for at in range(5):
            at_mask[0, at] = len(cache.get(at, [])) > 0

        # ── cube move mask ───────────────────────────────────────────────────
        # [1, max_cubes, max_moves_per_cube]
        cm_mask = np.zeros((B, self.max_cubes, self.max_moves_per_cube), dtype=bool)
        for move in cache.get(0, []):
            cid     = move.cube_id
            edge_id = list(type(move.pivot_edge)).index(move.pivot_edge) \
                    if hasattr(move.pivot_edge, 'value') else int(move.pivot_edge)
            dir_id  = 0 if move.direction > 0 else 1
            flat    = edge_id * 2 + dir_id
            if cid < self.max_cubes and flat < self.max_moves_per_cube:
                cm_mask[0, cid, flat] = True

        # ── separation mask ──────────────────────────────────────────────────
        sep_actions  = cache.get(1, [])
        sep_mask     = np.zeros((B, self.max_separation_actions), dtype=bool)
        sep_cube_sets: List[List[List[int]]] = [[]]

        for i, sep_action in enumerate(sep_actions[:self.max_separation_actions]):
            sep_mask[0, i] = True
            # sep_action is a SeparationAction – extract which cubes split off
            cube_ids = list(getattr(sep_action, 'separating_cube_ids',
                             getattr(sep_action, 'cube_ids', [])))
            sep_cube_sets[0].append(cube_ids)

        # ── docking mask ─────────────────────────────────────────────────────
        dock_actions = cache.get(2, [])
        dock_mask    = np.zeros((B, self.max_groups, self.max_groups), dtype=bool)
        groups       = env.constellation.get_all_groups()
        gid_to_idx   = {g.group_id: i for i, g in enumerate(groups)}

        for dock_action in dock_actions:
            i = gid_to_idx.get(getattr(dock_action, 'group1_id',
                               getattr(dock_action, 'source_group_id', -1)), -1)
            j = gid_to_idx.get(getattr(dock_action, 'group2_id',
                               getattr(dock_action, 'target_group_id', -1)), -1)
            if 0 <= i < self.max_groups and 0 <= j < self.max_groups:
                dock_mask[0, i, j] = True

        # ── maneuver mask ─────────────────────────────────────────────────────
        man_actions = cache.get(3, [])
        man_mask    = np.zeros((B, self.max_groups, 7), dtype=bool)

        for man_action in man_actions:
            g_id  = getattr(man_action, 'group_id', -1)
            g_idx = gid_to_idx.get(g_id, -1)
            d_idx = getattr(man_action, 'direction_idx',
                    getattr(man_action, 'direction', 0))
            if isinstance(d_idx, np.ndarray):
                d_idx = int(np.argmax(np.abs(d_idx)))
            if 0 <= g_idx < self.max_groups and 0 <= d_idx < 7:
                man_mask[0, g_idx, d_idx] = True

        # Ensure noop is always valid
        at_mask[0, 4] = True

        def t(x, dtype=torch.bool):
            return torch.tensor(x, dtype=dtype, device=self.device)

        return {
            'action_type'  : t(at_mask),
            'cube_move'    : t(cm_mask),
            'separation'   : t(sep_mask),
            'docking'      : t(dock_mask),
            'maneuver'     : t(man_mask),
            'sep_cube_sets': sep_cube_sets,
        }

    # ── Action decoding ───────────────────────────────────────────────────────

    def decode_action(self, action_type: int, sub_action: int) -> Tuple[int, int]:
        """
        Convert (action_type, sub_action_index) to the format expected by
        HierarchicalConstellationEnv.step(): np.array([action_type, sub_action]).
        """
        return action_type, sub_action