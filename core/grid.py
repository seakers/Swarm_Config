from typing import Tuple, Dict, Set, Optional


class SpatialGrid:
    """
    Manages the spatial positions of cubes in 3D space.
    
    Uses a dictionary-based sparse representation since cubes can spread out
    arbitrarily and we don't want to allocate a huge 3D array.
    """
    
    def __init__(self):
        # Maps position (x, y, z) -> cube_id
        self._position_to_cube: Dict[Tuple[int, int, int], int] = {}
        # Maps cube_id -> position (x, y, z)
        self._cube_to_position: Dict[int, Tuple[int, int, int]] = {}
    
    def place_cube(self, cube_id: int, position: Tuple[int, int, int]) -> bool:
        """
        Place a cube at a position.
        
        Args:
            cube_id: The cube's ID
            position: (x, y, z) grid position
            
        Returns:
            True if successful, False if position is occupied
        """
        if position in self._position_to_cube:
            return False
        
        # Remove from old position if cube was already placed
        if cube_id in self._cube_to_position:
            old_pos = self._cube_to_position[cube_id]
            del self._position_to_cube[old_pos]
        
        self._position_to_cube[position] = cube_id
        self._cube_to_position[cube_id] = position
        return True
    
    def remove_cube(self, cube_id: int) -> Optional[Tuple[int, int, int]]:
        """
        Remove a cube from the grid.
        
        Args:
            cube_id: The cube to remove
            
        Returns:
            The position the cube was at, or None if not found
        """
        if cube_id not in self._cube_to_position:
            return None
        
        position = self._cube_to_position[cube_id]
        del self._cube_to_position[cube_id]
        del self._position_to_cube[position]
        return position
    
    def get_cube_at(self, position: Tuple[int, int, int]) -> Optional[int]:
        """
        Get the cube ID at a position, or None if empty.
        """
        return self._position_to_cube.get(position)
    
    def get_position_of(self, cube_id: int) -> Optional[Tuple[int, int, int]]:
        """
        Get the position of a cube, or None if not placed.
        """
        return self._cube_to_position.get(cube_id)
    
    def is_empty(self, position: Tuple[int, int, int]) -> bool:
        """Check if a position is empty."""
        return position not in self._position_to_cube
    
    def is_occupied(self, position: Tuple[int, int, int]) -> bool:
        """Check if a position is occupied."""
        return position in self._position_to_cube
    
    def get_neighbors(self, position: Tuple[int, int, int]) -> Dict[Tuple[int, int, int], int]:
        """
        Get all occupied neighbor positions and their cube IDs.
        
        Args:
            position: Center position to check around
            
        Returns:
            Dict mapping neighbor_position -> cube_id for occupied neighbors
        """
        neighbors = {}
        for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
            neighbor_pos = (position[0]+dx, position[1]+dy, position[2]+dz)
            if neighbor_pos in self._position_to_cube:
                neighbors[neighbor_pos] = self._position_to_cube[neighbor_pos]
        return neighbors
    
    def get_all_positions(self) -> Set[Tuple[int, int, int]]:
        """Get all occupied positions."""
        return set(self._position_to_cube.keys())
    
    def get_all_cube_ids(self) -> Set[int]:
        """Get all placed cube IDs."""
        return set(self._cube_to_position.keys())
    
    def get_bounds(self) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Get the bounding box of all placed cubes.
        
        Returns:
            ((min_x, min_y, min_z), (max_x, max_y, max_z))
        """
        if not self._position_to_cube:
            return ((0, 0, 0), (0, 0, 0))
        
        positions = list(self._position_to_cube.keys())
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]
        zs = [p[2] for p in positions]
        
        return ((min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs)))
    
    def copy(self) -> 'SpatialGrid':
        """Return a deep copy of this grid."""
        new_grid = SpatialGrid()
        new_grid._position_to_cube = self._position_to_cube.copy()
        new_grid._cube_to_position = self._cube_to_position.copy()
        return new_grid
    
    def __len__(self) -> int:
        """Number of cubes in the grid."""
        return len(self._position_to_cube)