import numpy as np
from typing import Tuple, Optional, Dict, Set

from core.connections import Connection, ConnectionGraph
from core.cube import Cube, Orientation
from core.grid import SpatialGrid


class Swarm:
    """
    The main container for a modular spacecraft swarm.
    
    Manages:
    - Collection of Cube objects
    - Spatial positions (via SpatialGrid)
    - Connection topology (via ConnectionGraph)
    - Swarm-level operations and queries
    """
    
    def __init__(self, num_cubes: int = 64):
        """
        Initialize an empty swarm.
        
        Args:
            num_cubes: Number of cubes to create (they start unplaced)
        """
        self.num_cubes = num_cubes
        
        # Core data structures
        self._cubes: Dict[int, Cube] = {}
        self._grid: SpatialGrid = SpatialGrid()
        self._connections: ConnectionGraph = ConnectionGraph()
        
        # Create cube objects (but don't place them yet)
        for i in range(num_cubes):
            self._cubes[i] = Cube(cube_id=i, position=(0, 0, 0))
    
    # -------------------------------------------------------------------------
    # Cube access
    # -------------------------------------------------------------------------
    
    def get_cube(self, cube_id: int) -> Optional[Cube]:
        """Get a cube by ID."""
        return self._cubes.get(cube_id)
    
    def get_all_cubes(self) -> list[Cube]:
        """Get all cubes."""
        return list(self._cubes.values())
    
    def get_cube_at_position(self, position: Tuple[int, int, int]) -> Optional[Cube]:
        """Get the cube at a specific position."""
        cube_id = self._grid.get_cube_at(position)
        if cube_id is not None:
            return self._cubes[cube_id]
        return None
    
    # -------------------------------------------------------------------------
    # Placement and positioning
    # -------------------------------------------------------------------------
    
    def place_cube(self, cube_id: int, position: Tuple[int, int, int], 
                   orientation: Optional[Orientation] = None) -> bool:
        """
        Place a cube at a position in the grid.
        
        Args:
            cube_id: Which cube to place
            position: (x, y, z) grid position
            orientation: Optional orientation (defaults to identity)
            
        Returns:
            True if successful, False if position occupied or invalid cube_id
        """
        if cube_id not in self._cubes:
            return False
        
        if not self._grid.place_cube(cube_id, position):
            return False
        
        # Update the cube's position
        cube = self._cubes[cube_id]
        cube.position = position
        if orientation is not None:
            cube.orientation = orientation
        
        return True
    
    def move_cube(self, cube_id: int, new_position: Tuple[int, int, int],
                  new_orientation: Optional[Orientation] = None) -> bool:
        """
        Move a cube to a new position.
        
        This is a low-level operation that doesn't check physics validity.
        Use the movement system for proper hinge moves.
        
        Args:
            cube_id: Which cube to move
            new_position: Target position
            new_orientation: New orientation (optional)
            
        Returns:
            True if successful
        """
        if cube_id not in self._cubes:
            return False
        
        cube = self._cubes[cube_id]
        
        # Remove from old position
        self._grid.remove_cube(cube_id)
        
        # Place at new position
        if not self._grid.place_cube(cube_id, new_position):
            # Rollback - put it back
            self._grid.place_cube(cube_id, cube.position)
            return False
        
        cube.position = new_position
        if new_orientation is not None:
            cube.orientation = new_orientation
        
        return True
    
    # -------------------------------------------------------------------------
    # Connection management
    # -------------------------------------------------------------------------
    
    def connect_cubes(self, cube_id_1: int, cube_id_2: int) -> bool:
        """
        Create a magnetic connection between two adjacent cubes.
        
        Automatically determines which faces are bonded based on positions.
        
        Args:
            cube_id_1: First cube
            cube_id_2: Second cube
            
        Returns:
            True if connection created, False if not adjacent or already connected
        """
        cube1 = self._cubes.get(cube_id_1)
        cube2 = self._cubes.get(cube_id_2)
        
        if cube1 is None or cube2 is None:
            return False
        
        # Check adjacency
        pos1 = cube1.position
        pos2 = cube2.position
        
        diff = (pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2])
        
        # Must be exactly 1 unit apart in exactly one dimension
        if sum(abs(d) for d in diff) != 1:
            return False
        
        # Determine which faces are touching
        # The direction from cube1 to cube2 tells us which face of cube1 is touching
        face1 = cube1.get_face_pointing_direction(diff)
        
        # Opposite direction for cube2
        neg_diff = (-diff[0], -diff[1], -diff[2])
        face2 = cube2.get_face_pointing_direction(neg_diff)
        
        connection = Connection(cube_id_1, cube_id_2, face1, face2)
        return self._connections.add_connection(connection)
    
    def disconnect_cubes(self, cube_id_1: int, cube_id_2: int) -> bool:
        """
        Remove the connection between two cubes.
        
        Returns:
            True if disconnected, False if weren't connected
        """
        conn = self._connections.get_connection_between(cube_id_1, cube_id_2)
        if conn is None:
            return False
        return self._connections.remove_connection(conn)
    
    def auto_connect_all(self) -> int:
        """
        Automatically create connections between all adjacent cubes.
        
        Useful after placing cubes in a formation.
        
        Returns:
            Number of new connections created
        """
        count = 0
        checked_pairs = set()
        
        for cube_id, cube in self._cubes.items():
            neighbors = self._grid.get_neighbors(cube.position)
            
            for neighbor_pos, neighbor_id in neighbors.items():
                # Create canonical pair to avoid double-checking
                pair = (min(cube_id, neighbor_id), max(cube_id, neighbor_id))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                if self.connect_cubes(cube_id, neighbor_id):
                    count += 1
        
        return count
    
    def get_connections_for_cube(self, cube_id: int) -> Set[Connection]:
        """Get all connections for a cube."""
        return self._connections.get_connections_for_cube(cube_id)
    
    def get_connected_neighbors(self, cube_id: int) -> Set[int]:
        """Get IDs of all cubes connected to this one."""
        return self._connections.get_connected_cubes(cube_id)
    
    def is_connected(self) -> bool:
        """Check if the entire swarm forms a single connected component."""
        return self._connections.is_fully_connected()
    
    def get_connected_components(self) -> list[Set[int]]:
        """Get list of connected components (sets of cube IDs)."""
        return self._connections.find_connected_components()
    
    # -------------------------------------------------------------------------
    # Spatial queries
    # -------------------------------------------------------------------------
    
    def get_bounds(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """Get bounding box of the swarm."""
        return self._grid.get_bounds()
    
    def get_center_of_mass(self) -> Tuple[float, float, float]:
        """Get the geometric center of all cubes."""
        if not self._cubes:
            return (0.0, 0.0, 0.0)
        
        total_x, total_y, total_z = 0.0, 0.0, 0.0
        count = 0
        
        for cube in self._cubes.values():
            total_x += cube.position[0]
            total_y += cube.position[1]
            total_z += cube.position[2]
            count += 1
        
        return (total_x / count, total_y / count, total_z / count)
    
    def get_surface_area(self) -> int:
        """
        Calculate the total exposed surface area of the swarm.
        
        This is the number of cube faces that are not bonded to another cube.
        Lower surface area = more compact = better thermal properties.
        """
        total_faces = len(self._cubes) * 6  # Each cube has 6 faces
        bonded_faces = len(self._connections) * 2  # Each connection bonds 2 faces
        return total_faces - bonded_faces
    
    def get_convex_hull_volume(self) -> float:
        """
        Calculate the volume of the convex hull containing all cubes.
        
        Useful as a metric for how "spread out" the swarm is.
        """
        from scipy.spatial import ConvexHull
        
        positions = [cube.position for cube in self._cubes.values()]
        if len(positions) < 4:
            # Need at least 4 non-coplanar points for 3D hull
            return 0.0
        
        try:
            points = np.array(positions)
            hull = ConvexHull(points)
            return hull.volume
        except:
            # Points might be coplanar
            return 0.0
    
    def get_maximum_extent(self) -> float:
        """
        Get the maximum distance between any two cubes.
        
        Useful as a metric for sparse aperture configurations.
        """
        positions = [np.array(cube.position) for cube in self._cubes.values()]
        if len(positions) < 2:
            return 0.0
        
        max_dist = 0.0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                dist = np.linalg.norm(p1 - p2)
                max_dist = max(max_dist, dist)
        
        return max_dist
    
    def get_planar_area(self, normal: Tuple[float, float, float]) -> float:
        """
        Calculate the projected area of the swarm onto a plane.
        
        Args:
            normal: Normal vector of the plane to project onto
            
        Returns:
            Projected area (approximate, based on cube positions)
        """
        from scipy.spatial import ConvexHull
        
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Create orthonormal basis for the plane
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, [1, 0, 0])
        else:
            u = np.cross(normal, [0, 1, 0])
        u = u / np.linalg.norm(u)
        v = np.cross(normal, u)
        
        # Project all cube positions onto the plane
        projected = []
        for cube in self._cubes.values():
            pos = np.array(cube.position)
            proj_u = np.dot(pos, u)
            proj_v = np.dot(pos, v)
            projected.append([proj_u, proj_v])
        
        if len(projected) < 3:
            return float(len(projected))
        
        try:
            points = np.array(projected)
            hull = ConvexHull(points)
            return hull.volume  # In 2D, "volume" is area
        except:
            return float(len(projected))
    
    # -------------------------------------------------------------------------
    # State management
    # -------------------------------------------------------------------------
    
    def copy(self) -> 'Swarm':
        """Create a deep copy of the swarm."""
        new_swarm = Swarm(0)  # Create empty
        new_swarm.num_cubes = self.num_cubes
        new_swarm._cubes = {cid: cube.copy() for cid, cube in self._cubes.items()}
        new_swarm._grid = self._grid.copy()
        new_swarm._connections = self._connections.copy()
        return new_swarm
    
    def get_state_hash(self) -> int:
        """
        Get a hash of the current configuration.
        
        Useful for detecting repeated states.
        """
        # Hash based on positions and connections
        pos_tuple = tuple(sorted(
            (cid, cube.position) for cid, cube in self._cubes.items()
        ))
        conn_tuple = tuple(sorted(
            (c.cube_id_1, c.cube_id_2) for c in self._connections.get_all_connections()
        ))
        return hash((pos_tuple, conn_tuple))
    
    def __repr__(self) -> str:
        bounds = self.get_bounds()
        dims = tuple(bounds[1][i] - bounds[0][i] + 1 for i in range(3))
        return (f"Swarm({self.num_cubes} cubes, "
                f"{len(self._connections)} connections, "
                f"bounds={dims})")