"""
observation_builder.py
======================
Constructs PyTorch Geometric graph observations from Constellation state.

This handles the conversion from the simulation state to the GNN-compatible
graph format with proper node features, edge indices, and batch handling.
"""

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

from core.constellation import Constellation
from core.cube import Face
from core.cube_faces import FaceFunction, FUNCTION_TO_FACE
from mechanics.constellation_moves import ConstellationController
from mechanics.moves import MovementSystem


@dataclass
class ObservationConfig:
    """Configuration for observation construction."""
    # Feature dimensions
    cube_feature_dim: int = 25
    group_feature_dim: int = 12
    env_feature_dim: int = 12
    
    # Normalization parameters
    position_scale: float = 10.0  # Grid units
    velocity_scale: float = 10.0  # m/s
    distance_scale: float = 10000.0  # meters
    delta_v_scale: float = 50.0  # m/s
    
    # Graph structure
    include_global_node: bool = True
    max_cubes: int = 256
    max_groups: int = 8


class ConstellationObservationBuilder:
    """
    Builds graph observations from Constellation state.
    
    Creates a heterogeneous graph with:
    - Cube nodes (type 0): Individual cube states
    - Group nodes (type 1): Group-level states
    - Global node (type 2): Mission context
    
    Edges connect:
    - Cube to cube (physical connections)
    - Cube to group (membership)
    - Group to group (communication links)
    """
    
    def __init__(self, config: Optional[ObservationConfig] = None):
        self.config = config or ObservationConfig()
    
    def build_observation(self,
                          constellation: Constellation,
                          mission_mode: int,
                          sun_direction: Tuple[float, float, float],
                          earth_direction: Tuple[float, float, float],
                          target_direction: Tuple[float, float, float],
                          sun_distance_au: float = 10.0,
                          target_score: float = 0.8) -> Tuple[Data, torch.Tensor, torch.Tensor]:
        """
        Build complete observation from constellation state.
        
        Args:
            constellation: Current constellation state
            mission_mode: Integer mission mode index
            sun_direction: Direction to sun (unit vector)
            earth_direction: Direction to Earth (unit vector)
            target_direction: Direction to science target (unit vector)
            sun_distance_au: Distance from sun in AU
            target_score: Target score for task completion
            
        Returns:
            graph_data: PyG Data object with graph structure
            mode_idx: Mission mode index tensor
            env_features: Environmental feature tensor
        """
        device = torch.device('cpu')  # Will be moved to GPU during training
        
        # Build node features
        cube_features, cube_ids = self._build_cube_features(
            constellation, sun_direction, earth_direction, target_direction
        )
        
        group_features, group_ids = self._build_group_features(constellation)
        
        # Build edge indices
        edge_index = self._build_edge_index(constellation, cube_ids, group_ids)
        
        # Combine all node features
        num_cubes = len(cube_ids)
        num_groups = len(group_ids)
        
        # Pad features to fixed dimensions
        cube_features_padded = self._pad_features(
            cube_features, self.config.cube_feature_dim
        )
        group_features_padded = self._pad_features(
            group_features, self.config.group_feature_dim
        )
        
        # Stack all node features (cubes first, then groups, then global)
        all_features = []
        node_types = []
        
        # Cube nodes
        for cf in cube_features_padded:
            # Pad to max feature dim
            padded = np.zeros(max(self.config.cube_feature_dim, 
                                  self.config.group_feature_dim))
            padded[:len(cf)] = cf
            all_features.append(padded)
            node_types.append(0)
        
        # Group nodes
        for gf in group_features_padded:
            padded = np.zeros(max(self.config.cube_feature_dim,
                                  self.config.group_feature_dim))
            padded[:len(gf)] = gf
            all_features.append(padded)
            node_types.append(1)
        
        # Global node
        if self.config.include_global_node:
            global_features = self._build_global_features(
                constellation, sun_direction, earth_direction, 
                target_direction, sun_distance_au
            )
            padded = np.zeros(max(self.config.cube_feature_dim,
                                  self.config.group_feature_dim))
            padded[:len(global_features)] = global_features
            all_features.append(padded)
            node_types.append(2)
        
        # Convert to tensors
        x = torch.tensor(np.array(all_features), dtype=torch.float32)
        node_type = torch.tensor(node_types, dtype=torch.long)
        
        # Create Data object
        graph_data = Data(
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            num_cubes=num_cubes,
            num_groups=num_groups,
        )
        
        # Store mappings for action decoding
        graph_data.cube_id_map = cube_ids
        graph_data.group_id_map = group_ids
        
        # Build environmental features
        env_features = self._build_env_features(
            sun_direction, earth_direction, target_direction,
            sun_distance_au, target_score, constellation
        )
        env_features = torch.tensor(env_features, dtype=torch.float32).unsqueeze(0)
        
        # Mode index
        mode_idx = torch.tensor([mission_mode], dtype=torch.long)
        
        return graph_data, mode_idx, env_features
    
    def _build_cube_features(self,
                             constellation: Constellation,
                             sun_direction: Tuple[float, float, float],
                             earth_direction: Tuple[float, float, float],
                             target_direction: Tuple[float, float, float]
                             ) -> Tuple[List[np.ndarray], List[int]]:
        """Build feature vectors for each cube."""
        features = []
        cube_ids = []
        
        # Get group centroids for relative positioning
        group_centroids = {}
        for group in constellation.get_all_groups():
            positions = []
            for cid in group.cube_ids:
                cube = constellation.swarm.get_cube(cid)
                if cube:
                    positions.append(cube.position)
            if positions:
                group_centroids[group.group_id] = np.mean(positions, axis=0)
        
        # Build occupied positions set for exposure checking
        occupied = set()
        for cube in constellation.swarm.get_all_cubes():
            occupied.add(cube.position)
        
        sun_dir = np.array(sun_direction)
        earth_dir = np.array(earth_direction)
        target_dir = np.array(target_direction)
        
        for cube in constellation.swarm.get_all_cubes():
            cube_ids.append(cube.cube_id)
            
            # Get group for this cube
            group = constellation.get_group_for_cube(cube.cube_id)
            group_id = group.group_id if group else 0
            centroid = group_centroids.get(group_id, np.zeros(3))
            
            # Relative position within group
            rel_pos = (np.array(cube.position) - centroid) / self.config.position_scale
            
            # Orientation encoding (6D representation for continuity)
            rot_matrix = cube.orientation.matrix
            orientation_6d = rot_matrix[:, :2].flatten()  # First two columns
            
            # Face alignment scores
            solar_face = FUNCTION_TO_FACE.get(FaceFunction.SOLAR_ARRAY)
            antenna_face = FUNCTION_TO_FACE.get(FaceFunction.ANTENNA_HIGH_GAIN)
            camera_face = FUNCTION_TO_FACE.get(FaceFunction.CAMERA)
            radiator_face = FUNCTION_TO_FACE.get(FaceFunction.RADIATOR)
            
            def get_alignment(face, direction):
                if face is None:
                    return 0.0
                global_normal = cube.orientation.get_global_face_normal(face)
                return float(np.dot(global_normal, direction))
            
            solar_alignment = get_alignment(solar_face, sun_dir)
            antenna_alignment = get_alignment(antenna_face, earth_dir)
            camera_alignment = get_alignment(camera_face, target_dir)
            radiator_alignment = get_alignment(radiator_face, -sun_dir)
            
            # Face exposure (which faces are not blocked)
            face_exposure = []
            for face in Face:
                global_normal = cube.orientation.get_global_face_normal(face)
                adj_pos = (
                    cube.position[0] + int(global_normal[0]),
                    cube.position[1] + int(global_normal[1]),
                    cube.position[2] + int(global_normal[2])
                )
                is_exposed = 1.0 if adj_pos not in occupied else 0.0
                face_exposure.append(is_exposed)
            
            # Propulsion
            propulsion = constellation._cube_propulsion.get(cube.cube_id)
            if propulsion:
                delta_v_remaining = propulsion.remaining_delta_v / self.config.delta_v_scale
            else:
                delta_v_remaining = 0.0
            
            # Connectivity info
            connections = constellation.swarm._connections.get_connected_cubes(cube.cube_id)
            num_connections = len(connections) / 6.0  # Normalize by max
            is_boundary = 1.0 if len(connections) < 6 else 0.0
            
            # Assemble feature vector
            feature = np.concatenate([
                rel_pos,  # 3
                orientation_6d,  # 6
                [solar_alignment, antenna_alignment, camera_alignment, radiator_alignment],  # 4
                face_exposure,  # 6
                [delta_v_remaining],  # 1
                [num_connections, is_boundary],  # 2
            ])
            
            features.append(feature)
        
        return features, cube_ids
    
    def _build_group_features(self, 
                              constellation: Constellation) -> Tuple[List[np.ndarray], List[int]]:
        """Build feature vectors for each group."""
        features = []
        group_ids = []
        
        for group in constellation.get_all_groups():
            group_ids.append(group.group_id)
            
            # Position (normalized)
            position = group.position / self.config.distance_scale
            
            # Velocity
            velocity = group.velocity / self.config.velocity_scale
            
            # Group size
            num_cubes = len(group.cube_ids) / constellation.swarm.num_cubes
            
            # Propulsion
            delta_v = group.propulsion.remaining_delta_v / self.config.delta_v_scale
            
            # Communication state
            can_communicate = 1.0 if constellation.is_constellation_connected() else 0.0
            
            feature = np.concatenate([
                position,  # 3
                velocity,  # 3
                [num_cubes],  # 1
                [delta_v],  # 1
                [can_communicate],  # 1
                [0.0, 0.0, 0.0],  # Padding to 12
            ])
            
            features.append(feature)
        
        return features, group_ids
    
    def _build_global_features(self,
                               constellation: Constellation,
                               sun_direction: Tuple[float, float, float],
                               earth_direction: Tuple[float, float, float],
                               target_direction: Tuple[float, float, float],
                               sun_distance_au: float) -> np.ndarray:
        """Build global context features."""
        return np.array([
            constellation.get_num_groups() / self.config.max_groups,
            constellation.get_time() / 3600.0,  # Hours
            constellation.get_total_delta_v_remaining() / (self.config.delta_v_scale * constellation.swarm.num_cubes),
            1.0 if constellation.is_constellation_connected() else 0.0,
            constellation.get_max_baseline() / self.config.distance_scale,
            sun_distance_au / 30.0,  # Normalize to ~Saturn distance
        ])
    
    def _build_env_features(self,
                            sun_direction: Tuple[float, float, float],
                            earth_direction: Tuple[float, float, float],
                            target_direction: Tuple[float, float, float],
                            sun_distance_au: float,
                            target_score: float,
                            constellation: Constellation) -> np.ndarray:
        """Build environmental context features for goal conditioning."""
        return np.array([
            *sun_direction,  # 3
            *earth_direction,  # 3
            *target_direction,  # 3
            sun_distance_au / 30.0,  # 1
            target_score,  # 1
            constellation.get_num_groups() / self.config.max_groups,  # 1
        ])
    
    def _build_edge_index(self,
                          constellation: Constellation,
                          cube_ids: List[int],
                          group_ids: List[int]) -> torch.Tensor:
        """Build edge connectivity for the graph."""
        edges = []
        
        cube_id_to_node = {cid: i for i, cid in enumerate(cube_ids)}
        group_id_to_node = {gid: len(cube_ids) + i for i, gid in enumerate(group_ids)}
        global_node = len(cube_ids) + len(group_ids) if self.config.include_global_node else None
        
        # Cube-to-cube edges (physical connections)
        for conn in constellation.swarm._connections.get_all_connections():
            if conn.cube_id_1 in cube_id_to_node and conn.cube_id_2 in cube_id_to_node:
                n1 = cube_id_to_node[conn.cube_id_1]
                n2 = cube_id_to_node[conn.cube_id_2]
                edges.append([n1, n2])
                edges.append([n2, n1])  # Bidirectional
        
        # Cube-to-group edges (membership)
        for group in constellation.get_all_groups():
            group_node = group_id_to_node[group.group_id]
            for cid in group.cube_ids:
                if cid in cube_id_to_node:
                    cube_node = cube_id_to_node[cid]
                    edges.append([cube_node, group_node])
                    edges.append([group_node, cube_node])
        
        # Group-to-group edges (communication)
        comm_graph = constellation.get_communication_graph()
        for gid_a, neighbors in comm_graph.items():
            if gid_a in group_id_to_node:
                node_a = group_id_to_node[gid_a]
                for gid_b in neighbors:
                    if gid_b in group_id_to_node:
                        node_b = group_id_to_node[gid_b]
                        edges.append([node_a, node_b])
        
        # Global node connections
        if global_node is not None:
            # Connect global to all groups
            for gid, gnode in group_id_to_node.items():
                edges.append([global_node, gnode])
                edges.append([gnode, global_node])
        
        if not edges:
            # Empty graph - add self-loop on first node
            edges = [[0, 0]]
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return edge_index
    
    def _pad_features(self, features: List[np.ndarray], 
                      target_dim: int) -> List[np.ndarray]:
        """Pad feature vectors to target dimension."""
        padded = []
        for f in features:
            if len(f) < target_dim:
                p = np.zeros(target_dim)
                p[:len(f)] = f
                padded.append(p)
            else:
                padded.append(f[:target_dim])
        return padded


class ActionMaskBuilder:
    """
    Builds action masks for the hierarchical action space.
    """
    
    def __init__(self,
                 max_cubes: int = 64,
                 max_groups: int = 8,
                 max_separations: int = 100,
                 num_maneuver_directions: int = 7):
        self.max_cubes = max_cubes
        self.max_groups = max_groups
        self.max_separations = max_separations
        self.num_maneuver_directions = num_maneuver_directions
        self.max_moves_per_cube = 24
    
    def build_action_masks(self,
                           constellation: Constellation,
                           controller: 'ConstellationController',
                           movement: 'MovementSystem') -> Dict[str, torch.Tensor]:
        """
        Build all action masks for current state.
        
        Returns dict with masks for each action type and sub-action.
        """
        masks = {}
        
        # Action type mask
        action_type_mask = torch.zeros(5, dtype=torch.bool)
        
        # Check cube moves
        valid_cube_moves = movement.get_all_valid_moves()
        if valid_cube_moves:
            action_type_mask[0] = True
        
        # Check separations
        valid_separations = controller.get_valid_separation_actions()
        if valid_separations:
            action_type_mask[1] = True
        
        # Check docking
        valid_dockings = controller.get_valid_docking_actions()
        if valid_dockings:
            action_type_mask[2] = True
        
        # Check maneuvers
        valid_maneuvers = controller.get_valid_maneuver_actions()
        if valid_maneuvers:
            action_type_mask[3] = True
        
        # Noop always valid
        action_type_mask[4] = True
        
        masks['action_type'] = action_type_mask.unsqueeze(0)
        
        # Cube move mask
        cube_move_mask = torch.zeros(
            1, self.max_cubes, self.max_moves_per_cube, dtype=torch.bool
        )
        
        # Build cube ID to index mapping
        cube_id_to_idx = {}
        for group in constellation.get_all_groups():
            for i, cid in enumerate(sorted(group.cube_ids)):
                if len(cube_id_to_idx) < self.max_cubes:
                    cube_id_to_idx[cid] = len(cube_id_to_idx)
        
        # Get valid moves and mark in mask
        for move in valid_cube_moves:
            cube_idx = cube_id_to_idx.get(move.cube_id)
            if cube_idx is not None and cube_idx < self.max_cubes:
                # Encode move as index: edge_idx * 2 + (0 if direction > 0 else 1)
                edge_idx = move.pivot_edge.value if hasattr(move.pivot_edge, 'value') else 0
                dir_idx = 0 if move.direction > 0 else 1
                move_idx = edge_idx * 2 + dir_idx
                
                if move_idx < self.max_moves_per_cube:
                    cube_move_mask[0, cube_idx, move_idx] = True
        
        masks['cube_move'] = cube_move_mask
        
        # Separation masks
        separation_valid_mask = torch.zeros(1, self.max_separations, dtype=torch.bool)
        separation_cube_masks = torch.zeros(
            1, self.max_separations, self.max_cubes, dtype=torch.bool
        )
        
        for i, action in enumerate(valid_separations[:self.max_separations]):
            separation_valid_mask[0, i] = True
            
            for cid in action.cube_ids:
                cube_idx = cube_id_to_idx.get(cid)
                if cube_idx is not None:
                    separation_cube_masks[0, i, cube_idx] = True
        
        masks['separation_valid'] = separation_valid_mask
        masks['separation_cube_masks'] = separation_cube_masks
        
        # Store separation actions for decoding
        masks['_separation_actions'] = valid_separations
        
        # Docking mask
        max_docking_pairs = self.max_groups * (self.max_groups - 1) // 2
        docking_mask = torch.zeros(1, max_docking_pairs, dtype=torch.bool)
        
        for i, action in enumerate(valid_dockings[:max_docking_pairs]):
            docking_mask[0, i] = True
        
        masks['docking'] = docking_mask
        masks['_docking_actions'] = valid_dockings
        
        # Maneuver mask
        maneuver_mask = torch.zeros(
            1, self.max_groups * self.num_maneuver_directions, dtype=torch.bool
        )
        
        groups = constellation.get_all_groups()
        for g_idx, group in enumerate(groups[:self.max_groups]):
            if group.propulsion.remaining_delta_v > 0.1:
                for d_idx in range(self.num_maneuver_directions):
                    action_idx = g_idx * self.num_maneuver_directions + d_idx
                    maneuver_mask[0, action_idx] = True
        
        masks['maneuver'] = maneuver_mask
        
        return masks
    
    def decode_action(self,
                      action_type: int,
                      sub_action: int,
                      constellation: 'Constellation',
                      masks: Dict[str, torch.Tensor]) -> Optional[any]:
        """
        Decode action indices back to action objects.
        
        Args:
            action_type: 0=cube_move, 1=separation, 2=docking, 3=maneuver, 4=noop
            sub_action: Index within the action type
            constellation: Current constellation state
            masks: Action masks (contains cached action lists)
            
        Returns:
            Action object or None for noop
        """
        if action_type == 0:  # Cube move
            # Decode sub_action to cube_idx and move_idx
            cube_idx = sub_action // self.max_moves_per_cube
            move_idx = sub_action % self.max_moves_per_cube
            
            # Find cube ID from index
            cube_id_to_idx = {}
            for group in constellation.get_all_groups():
                for cid in sorted(group.cube_ids):
                    if len(cube_id_to_idx) < self.max_cubes:
                        cube_id_to_idx[cid] = len(cube_id_to_idx)
            
            idx_to_cube_id = {v: k for k, v in cube_id_to_idx.items()}
            cube_id = idx_to_cube_id.get(cube_idx)
            
            if cube_id is None:
                return None
            
            # Decode move_idx to edge and direction
            from core.cube import Edge
            edge_idx = move_idx // 2
            direction = 1 if move_idx % 2 == 0 else -1
            
            edges = list(Edge)
            if edge_idx < len(edges):
                from mechanics.moves import HingeMove
                return HingeMove(cube_id, edges[edge_idx], direction)
            return None
        
        elif action_type == 1:  # Separation
            actions = masks.get('_separation_actions', [])
            if sub_action < len(actions):
                return actions[sub_action]
            return None
        
        elif action_type == 2:  # Docking
            actions = masks.get('_docking_actions', [])
            if sub_action < len(actions):
                return actions[sub_action]
            return None
        
        elif action_type == 3:  # Maneuver
            group_idx = sub_action // self.num_maneuver_directions
            direction_idx = sub_action % self.num_maneuver_directions
            
            groups = constellation.get_all_groups()
            if group_idx < len(groups):
                group = groups[group_idx]
                
                # Direction vectors
                directions = [
                    (1, 0, 0), (-1, 0, 0),
                    (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1),
                    (0, 0, 0),  # Zero maneuver
                ]
                
                if direction_idx < len(directions):
                    from mechanics.constellation_moves import ManeuverAction
                    delta_v = tuple(d * 1.0 for d in directions[direction_idx])
                    return ManeuverAction(group.group_id, delta_v)
            return None
        
        elif action_type == 4:  # Noop
            return None
        
        return None


def collate_observations(observations: List[Tuple]) -> Tuple:
    """
    Collate multiple observations into a batch.
    
    Args:
        observations: List of (graph_data, mode_idx, env_features) tuples
        
    Returns:
        Batched (graph_data, mode_idx, env_features)
    """
    from torch_geometric.data import Batch
    
    graph_datas = [obs[0] for obs in observations]
    mode_idxs = torch.cat([obs[1] for obs in observations], dim=0)
    env_features = torch.cat([obs[2] for obs in observations], dim=0)
    
    batched_graph = Batch.from_data_list(graph_datas)
    
    return batched_graph, mode_idxs, env_features