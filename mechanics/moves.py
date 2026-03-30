import numpy as np
from dataclasses import dataclass, FrozenInstanceError, field
from typing import Set, Dict, Optional, Tuple, List

from core.cube import Cube, Orientation, Edge
from core.swarm import Swarm
from core.connections import Connection


@dataclass(frozen=True)
class HingeMove:
    """
    Represents a hinge-pivot move for a cube.
    
    A cube pivots around one of its edges. The rotation will be either
    90° or 180° depending on whether there's a supporting cube to land on.
    
    Attributes:
        cube_id: The cube that is moving
        pivot_edge: Which edge of the cube to pivot around (in cube's local frame)
        direction: +1 or -1 for the two possible rotation directions around the edge
    """
    cube_id: int
    pivot_edge: Edge
    direction: int  # +1 or -1
    
    def __post_init__(self):
        if self.direction not in (-1, +1):
            raise ValueError(f"Direction must be +1 or -1, got {self.direction}")


@dataclass
class MoveResult:
    """
    Result of attempting a move.
    
    Attributes:
        success: Whether the move was executed
        reason: If failed, why it failed
        new_position: The cube's new position (if successful)
        new_orientation: The cube's new orientation (if successful)
        rotation_degrees: Whether the move was 90° or 180°
        broken_connections: Connections that were broken by this move
        new_connections: Connections that were formed by this move
    """
    success: bool
    reason: str = ""
    new_position: Optional[Tuple[int, int, int]] = None
    new_orientation: Optional[Orientation] = None
    rotation_degrees: int = 0  # 90 or 180
    broken_connections: Set[Connection] = field(default_factory=set)
    new_connections: Set[Connection] = field(default_factory=set)


class MovementSystem:
    """
    Handles all movement logic for the swarm.
    
    Movement model:
    - A cube pivots around one of its edges that has support
    - If there's a cube to land on at 90°, the rotation is 90°
    - If there's no support at 90°, the cube swings 180° to the other side
    """
    
    def __init__(self, swarm: Swarm, require_connectivity: bool = False):
        """
        Args:
            swarm: The swarm to control
            require_connectivity: If True, reject moves that would split the swarm
        """
        self.swarm = swarm
        self.require_connectivity = require_connectivity
    
    def get_valid_moves(self, cube_id: int) -> List[HingeMove]:
        """
        Get all valid hinge moves for a specific cube.
        
        Args:
            cube_id: Which cube to get moves for
            
        Returns:
            List of valid HingeMove objects
        """
        cube = self.swarm.get_cube(cube_id)
        if cube is None:
            return []
        
        valid_moves = []
        
        # Check each of the 12 edges
        for edge in Edge:
            # Check both rotation directions
            for direction in [+1, -1]:
                move = HingeMove(cube_id, edge, direction)
                if self._is_move_valid(move):
                    valid_moves.append(move)
        
        return valid_moves
    
    def get_all_valid_moves(self) -> List[HingeMove]:
        """
        Get all valid moves for all cubes in the swarm.
        
        Returns:
            List of all valid HingeMove objects
        """
        all_moves = []
        for cube_id in range(self.swarm.num_cubes):
            all_moves.extend(self.get_valid_moves(cube_id))
        return all_moves
    
    def _is_move_valid(self, move: HingeMove) -> bool:
        """Check if a move is valid without executing it."""
        result = self._compute_move_result(move, dry_run=True)
        return result.success
    
    def execute_move(self, move: HingeMove) -> MoveResult:
        """
        Execute a hinge move, updating the swarm state.
        
        Args:
            move: The move to execute
            
        Returns:
            MoveResult indicating success/failure and state changes
        """
        result = self._compute_move_result(move, dry_run=False)
        
        if result.success:
            self._apply_move(move, result)
        
        return result
    
    def _compute_move_result(self, move: HingeMove, dry_run: bool = True) -> MoveResult:
        """
        Compute what would happen if a move is executed.
        
        This implements the 90°/180° rotation logic:
        1. Check if pivot edge has support
        2. Compute 90° destination
        3. If there's landing support at 90° AND destination is clear -> 90° rotation
        4. Otherwise, compute 180° destination and check that
        
        Args:
            move: The move to analyze
            dry_run: If True, don't modify state, just compute result
            
        Returns:
            MoveResult with details of what would happen
        """
        cube = self.swarm.get_cube(move.cube_id)
        if cube is None:
            return MoveResult(False, "Cube does not exist")
        
        # Step 1: Check if the pivot edge has support (required for any rotation)
        pivot_neighbor_id = self._get_pivot_neighbor(cube, move.pivot_edge)
        if pivot_neighbor_id is None:
            return MoveResult(False, "Pivot edge has no support")
        
        # Step 2: Compute the 90° destination
        pos_90, orient_90 = self._compute_pivot_result(
            cube, move.pivot_edge, move.direction, degrees=90
        )
        
        # Step 3: Check conditions for 90° vs 180° rotation
        dest_90_clear = self.swarm._grid.is_empty(pos_90)
        has_support_90 = self._has_landing_support(
            cube.cube_id, pos_90, pivot_neighbor_id
        )
        
        if dest_90_clear and has_support_90:
            # 90° rotation - cube lands on a supporting neighbor
            rotation_degrees = 90
            new_pos = pos_90
            new_orient = orient_90
        else:
            # Need to try 180° rotation
            rotation_degrees = 180
            
            # Compute 180° destination
            pos_180, orient_180 = self._compute_pivot_result(
                cube, move.pivot_edge, move.direction, degrees=180
            )
            
            # Check if 180° destination is clear
            if not self.swarm._grid.is_empty(pos_180):
                if not dest_90_clear:
                    return MoveResult(False, "Destination blocked at both 90° and 180°")
                else:
                    return MoveResult(False, "No support at 90°, and 180° destination is blocked")
            
            new_pos = pos_180
            new_orient = orient_180
        
        # Step 4: Check if swept path is clear
        swept_cells = self._compute_swept_cells(
            cube, move.pivot_edge, move.direction, rotation_degrees
        )
        for cell in swept_cells:
            if cell != cube.position and cell != new_pos:
                if self.swarm._grid.is_occupied(cell):
                    return MoveResult(False, f"Swept path blocked at {cell}")
        
        # Step 5: Determine which connections break and form
        broken = self._get_connections_broken_by_move(cube, new_pos)
        formed = self._get_connections_formed_by_move(cube, new_pos, new_orient)
        
        # Step 6: Check connectivity constraint if required
        if self.require_connectivity:
            if not self._would_remain_connected(move.cube_id, broken, formed):
                return MoveResult(False, "Move would disconnect swarm")
        
        return MoveResult(
            success=True,
            new_position=new_pos,
            new_orientation=new_orient,
            rotation_degrees=rotation_degrees,
            broken_connections=broken,
            new_connections=formed
        )
    
    def _get_pivot_neighbor(self, cube: Cube, edge: Edge) -> Optional[int]:
        """
        Find the neighbor cube that provides the pivot point for this edge.
        
        Returns the cube_id of the neighbor, or None if no support.
        """
        # Get the two faces adjacent to this edge
        face1, face2 = edge.get_adjacent_faces()
        
        # Check each adjacent face for a bonded neighbor
        for face in [face1, face2]:
            global_normal = cube.orientation.get_global_face_normal(face)
            neighbor_pos = (
                cube.position[0] + global_normal[0],
                cube.position[1] + global_normal[1],
                cube.position[2] + global_normal[2]
            )
            
            neighbor_id = self.swarm._grid.get_cube_at(neighbor_pos)
            if neighbor_id is not None:
                # Check if actually connected (not just adjacent)
                if self.swarm._connections.are_connected(cube.cube_id, neighbor_id):
                    return neighbor_id
        
        # Also check the corner position (for edges where two cubes meet at corner)
        global_normal1 = cube.orientation.get_global_face_normal(face1)
        global_normal2 = cube.orientation.get_global_face_normal(face2)
        
        corner_pos = (
            cube.position[0] + global_normal1[0] + global_normal2[0],
            cube.position[1] + global_normal1[1] + global_normal2[1],
            cube.position[2] + global_normal1[2] + global_normal2[2]
        )
        
        corner_neighbor_id = self.swarm._grid.get_cube_at(corner_pos)
        if corner_neighbor_id is not None:
            return corner_neighbor_id
        
        return None
    
    def _has_landing_support(self, moving_cube_id: int, 
                             landing_pos: Tuple[int, int, int],
                             pivot_neighbor_id: int) -> bool:
        """
        Check if there's a cube to land on at the 90° position.
        
        The landing support must be from a cube OTHER than:
        - The moving cube itself
        
        For now, we consider there to be support if any cube is adjacent
        to the landing position. The pivot neighbor can provide support
        (this is the common case of rolling along a surface).
        """
        # Check all 6 adjacent positions for potential support
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            adj_pos = (landing_pos[0] + dx, landing_pos[1] + dy, landing_pos[2] + dz)
            neighbor_id = self.swarm._grid.get_cube_at(adj_pos)
            
            # Found a cube that isn't the moving cube
            if neighbor_id is not None and neighbor_id != moving_cube_id:
                return True
        
        return False
    
    def _compute_pivot_result(self, cube: Cube, edge: Edge, direction: int,
                              degrees: int = 90) -> Tuple[Tuple[int, int, int], Orientation]:
        """
        Compute where a cube ends up after pivoting around an edge.
        
        Args:
            cube: The cube that's moving
            edge: Which edge to pivot around
            direction: +1 or -1 for rotation direction
            degrees: 90 or 180 degrees of rotation
            
        Returns:
            (new_position, new_orientation)
        """
        # The edge defines the pivot axis
        axis = edge.get_axis()
        
        # Get the edge's position relative to cube center
        edge_offset = edge.get_position_offset()
        
        # Convert edge offset to global frame
        edge_offset_local = np.array(edge_offset)
        edge_offset_global = cube.orientation.matrix @ edge_offset_local
        
        # The pivot point in global coordinates
        pivot_point = np.array(cube.position, dtype=float) + edge_offset_global
        
        # Vector from pivot to cube center
        to_center = np.array(cube.position, dtype=float) - pivot_point
        
        # Find what global axis the edge is parallel to
        axis_vec_local = np.zeros(3)
        axis_vec_local[axis] = 1
        axis_vec_global = cube.orientation.matrix @ axis_vec_local
        
        # Find which global axis this is closest to
        global_axis = int(np.argmax(np.abs(axis_vec_global)))
        axis_sign = int(np.sign(axis_vec_global[global_axis]))
        
        # Compute rotation based on degrees
        if degrees == 90:
            # Single 90° rotation
            rotated_to_center = self._rotate_vector_90(
                to_center, global_axis, direction * axis_sign
            )
            new_orient = cube.orientation.rotate_90(global_axis, direction * axis_sign)
        elif degrees == 180:
            # Two 90° rotations = 180°
            rotated_once = self._rotate_vector_90(
                to_center, global_axis, direction * axis_sign
            )
            rotated_to_center = self._rotate_vector_90(
                rotated_once, global_axis, direction * axis_sign
            )
            orient_once = cube.orientation.rotate_90(global_axis, direction * axis_sign)
            new_orient = orient_once.rotate_90(global_axis, direction * axis_sign)
        else:
            raise ValueError(f"Degrees must be 90 or 180, got {degrees}")
        
        # New position
        new_pos_float = pivot_point + rotated_to_center
        new_pos = tuple(int(round(x)) for x in new_pos_float)
        
        return new_pos, new_orient
    
    def _rotate_vector_90(self, vec: np.ndarray, axis: int, direction: int) -> np.ndarray:
        """Rotate a vector 90 degrees around a coordinate axis."""
        c, s = 0, direction  # cos(90°)=0, sin(90°)=±1
        
        if axis == 0:  # X-axis rotation
            rot = np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s, c]], dtype=float)
        elif axis == 1:  # Y-axis rotation
            rot = np.array([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]], dtype=float)
        else:  # Z-axis rotation
            rot = np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]], dtype=float)
        
        return rot @ vec
    
    def _compute_swept_cells(self, cube: Cube, edge: Edge, direction: int,
                             degrees: int) -> Set[Tuple[int, int, int]]:
        """
        Compute all grid cells that the cube passes through during the pivot.
        
        For 90°: check the corner cell in the arc
        For 180°: check the 90° position plus additional cells in the full arc
        """
        swept = {cube.position}
        
        # Get intermediate position at 90°
        pos_90, _ = self._compute_pivot_result(cube, edge, direction, degrees=90)
        
        if degrees >= 90:
            # Add cells along the 90° arc
            # The key cell is the diagonal that the corner passes through
            edge_offset = np.array(edge.get_position_offset())
            old_to_new_pos = np.array(pos_90) - np.array(cube.position)
            sweep_diff_vector = old_to_new_pos - 2*edge_offset

            swept = set()
            swept.add(tuple(int(round(cube.position[i] + sweep_diff_vector[i])) for i in range(3)))
            swept.add(tuple(int(round(pos_90[i] + sweep_diff_vector[i])) for i in range(3)))
            swept.add(pos_90)
        
        if degrees == 180:
            # Add the 180° position
            pos_180, _ = self._compute_pivot_result(cube, edge, direction, degrees=180)
            swept.add(pos_180)
            
            # Add cells along the second 90° arc (90° to 180°)
            # We need to compute this from the 90° position
            edge_offset_180 = edge_offset - old_to_new_pos
            old_to_new_pos_180 = np.array(pos_180) - np.array(pos_90)
            sweep_diff_vector_180 = old_to_new_pos_180 - 2*edge_offset_180
            swept.add(tuple(int(round(pos_90[i] + sweep_diff_vector_180[i])) for i in range(3)))
            swept.add(tuple(int(round(pos_180[i] + sweep_diff_vector_180[i])) for i in range(3)))
        
        return swept
    
    def _get_connections_broken_by_move(self, cube: Cube, 
                                        new_pos: Tuple[int, int, int]) -> Set[Connection]:
        """
        Determine which connections will break when a cube moves.
        
        A connection breaks if the two cubes are no longer adjacent after the move.
        """
        broken = set()
        
        for conn in self.swarm.get_connections_for_cube(cube.cube_id):
            other_id = conn.get_other_cube(cube.cube_id)
            other_cube = self.swarm.get_cube(other_id)
            
            if other_cube is None:
                continue
            
            # Check if still adjacent after move
            diff = tuple(new_pos[i] - other_cube.position[i] for i in range(3))
            manhattan = sum(abs(d) for d in diff)
            
            if manhattan != 1:  # No longer adjacent
                broken.add(conn)
        
        return broken
    
    def _get_connections_formed_by_move(self, cube: Cube, 
                                        new_pos: Tuple[int, int, int],
                                        new_orient: Orientation) -> Set[Connection]:
        """
        Determine which new connections will form when a cube moves to a new position.
        """
        formed = set()
        
        # Check all 6 directions for neighbors at new position
        for direction in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            neighbor_pos = (
                new_pos[0] + direction[0],
                new_pos[1] + direction[1],
                new_pos[2] + direction[2]
            )
            
            neighbor_id = self.swarm._grid.get_cube_at(neighbor_pos)
            if neighbor_id is None or neighbor_id == cube.cube_id:
                continue
            
            # Skip if already connected (connection won't be "new")
            if self.swarm._connections.are_connected(cube.cube_id, neighbor_id):
                continue
            
            neighbor_cube = self.swarm.get_cube(neighbor_id)
            if neighbor_cube is None:
                continue
            
            # Determine which faces would bond
            # For the moving cube, find which local face points in 'direction' with new orientation
            moving_face = new_orient.get_local_face_for_direction(direction)
            
            # For the neighbor, find which face points back at us
            neg_direction = (-direction[0], -direction[1], -direction[2])
            neighbor_face = neighbor_cube.orientation.get_local_face_for_direction(neg_direction)
            
            new_conn = Connection(cube.cube_id, neighbor_id, moving_face, neighbor_face)
            formed.add(new_conn)
        
        return formed
    
    def _would_remain_connected(self, moving_cube_id: int,
                                broken: Set[Connection],
                                formed: Set[Connection]) -> bool:
        """
        Check if the swarm would remain fully connected after a move.
        """
        # Create a temporary copy of adjacency
        temp_adjacency: Dict[int, Set[int]] = {}
        for conn in self.swarm._connections.get_all_connections():
            c1, c2 = conn.cube_id_1, conn.cube_id_2
            if c1 not in temp_adjacency:
                temp_adjacency[c1] = set()
            if c2 not in temp_adjacency:
                temp_adjacency[c2] = set()
            temp_adjacency[c1].add(c2)
            temp_adjacency[c2].add(c1)
        
        # Remove broken connections
        for conn in broken:
            c1, c2 = conn.cube_id_1, conn.cube_id_2
            if c1 in temp_adjacency:
                temp_adjacency[c1].discard(c2)
            if c2 in temp_adjacency:
                temp_adjacency[c2].discard(c1)
        
        # Add formed connections
        for conn in formed:
            c1, c2 = conn.cube_id_1, conn.cube_id_2
            if c1 not in temp_adjacency:
                temp_adjacency[c1] = set()
            if c2 not in temp_adjacency:
                temp_adjacency[c2] = set()
            temp_adjacency[c1].add(c2)
            temp_adjacency[c2].add(c1)
        
        # BFS to check connectivity
        if not temp_adjacency:
            return True
        
        start = next(iter(temp_adjacency.keys()))
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in temp_adjacency.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        # Check if all cubes with connections are reachable
        all_cubes = set(temp_adjacency.keys())
        return visited == all_cubes
    
    def _apply_move(self, move: HingeMove, result: MoveResult) -> None:
        """
        Apply a validated move to the swarm state.
        """
        cube = self.swarm.get_cube(move.cube_id)
        
        # Update position and orientation
        self.swarm.move_cube(move.cube_id, result.new_position, result.new_orientation)
        
        # Remove broken connections
        for conn in result.broken_connections:
            self.swarm._connections.remove_connection(conn)
        
        # Add new connections
        for conn in result.new_connections:
            self.swarm._connections.add_connection(conn)
        
        # Also check for any connections that should reform
        self._reform_adjacent_connections(move.cube_id)
    
    def _reform_adjacent_connections(self, cube_id: int) -> None:
        """
        Create connections to any adjacent cubes that aren't already connected.
        """
        cube = self.swarm.get_cube(cube_id)
        if cube is None:
            return
        
        neighbors = self.swarm._grid.get_neighbors(cube.position)
        
        for neighbor_pos, neighbor_id in neighbors.items():
            if not self.swarm._connections.are_connected(cube_id, neighbor_id):
                self.swarm.connect_cubes(cube_id, neighbor_id)