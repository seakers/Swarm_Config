from dataclasses import dataclass, FrozenInstanceError
from typing import Set, Dict, Optional

from core.cube import Face


@dataclass(frozen=True)
class Connection:
    """
    Represents a magnetic connection between two cubes.
    
    Connections are undirected, so (cube_a, cube_b) == (cube_b, cube_a).
    We normalize by always storing the smaller ID first.
    
    Attributes:
        cube_id_1: First cube ID (always the smaller one)
        cube_id_2: Second cube ID (always the larger one)
        face_1: Which face of cube_1 is bonded (in cube_1's local frame)
        face_2: Which face of cube_2 is bonded (in cube_2's local frame)
    """
    cube_id_1: int
    cube_id_2: int
    face_1: Face
    face_2: Face
    
    def __post_init__(self):
        # Ensure cube_id_1 < cube_id_2 for canonical form
        if self.cube_id_1 > self.cube_id_2:
            # Swap - need to use object.__setattr__ because frozen
            object.__setattr__(self, 'cube_id_1', self.cube_id_2)
            object.__setattr__(self, 'cube_id_2', self.cube_id_1)
            object.__setattr__(self, 'face_1', self.face_2)
            object.__setattr__(self, 'face_2', self.face_1)
    
    def involves_cube(self, cube_id: int) -> bool:
        """Check if this connection involves a specific cube."""
        return cube_id == self.cube_id_1 or cube_id == self.cube_id_2
    
    def get_other_cube(self, cube_id: int) -> int:
        """Get the other cube in this connection."""
        if cube_id == self.cube_id_1:
            return self.cube_id_2
        elif cube_id == self.cube_id_2:
            return self.cube_id_1
        else:
            raise ValueError(f"Cube {cube_id} is not part of this connection")
    
    def get_face_for_cube(self, cube_id: int) -> Face:
        """Get which face is bonded for a specific cube."""
        if cube_id == self.cube_id_1:
            return self.face_1
        elif cube_id == self.cube_id_2:
            return self.face_2
        else:
            raise ValueError(f"Cube {cube_id} is not part of this connection")


class ConnectionGraph:
    """
    Manages the connection topology between cubes.
    
    This is essentially an undirected graph where nodes are cubes and 
    edges are magnetic bonds. Each edge also stores which faces are bonded.
    """
    
    def __init__(self):
        # Set of all connections
        self._connections: Set[Connection] = set()
        # Adjacency list: cube_id -> set of connected cube_ids
        self._adjacency: Dict[int, Set[int]] = {}
        # Quick lookup: cube_id -> set of connections involving that cube
        self._cube_connections: Dict[int, Set[Connection]] = {}
    
    def add_connection(self, connection: Connection) -> bool:
        """
        Add a connection between two cubes.
        
        Args:
            connection: The connection to add
            
        Returns:
            True if added, False if connection already exists
        """
        if connection in self._connections:
            return False
        
        self._connections.add(connection)
        
        # Update adjacency
        for cube_id in [connection.cube_id_1, connection.cube_id_2]:
            if cube_id not in self._adjacency:
                self._adjacency[cube_id] = set()
            if cube_id not in self._cube_connections:
                self._cube_connections[cube_id] = set()
        
        self._adjacency[connection.cube_id_1].add(connection.cube_id_2)
        self._adjacency[connection.cube_id_2].add(connection.cube_id_1)
        self._cube_connections[connection.cube_id_1].add(connection)
        self._cube_connections[connection.cube_id_2].add(connection)
        
        return True
    
    def remove_connection(self, connection: Connection) -> bool:
        """
        Remove a connection.
        
        Returns:
            True if removed, False if connection didn't exist
        """
        if connection not in self._connections:
            return False
        
        self._connections.remove(connection)
        self._adjacency[connection.cube_id_1].discard(connection.cube_id_2)
        self._adjacency[connection.cube_id_2].discard(connection.cube_id_1)
        self._cube_connections[connection.cube_id_1].discard(connection)
        self._cube_connections[connection.cube_id_2].discard(connection)
        
        return True
    
    def remove_all_connections_for_cube(self, cube_id: int) -> Set[Connection]:
        """
        Remove all connections involving a cube.
        
        Returns:
            Set of connections that were removed
        """
        if cube_id not in self._cube_connections:
            return set()
        
        removed = self._cube_connections[cube_id].copy()
        for conn in removed:
            self.remove_connection(conn)
        
        return removed
    
    def get_connections_for_cube(self, cube_id: int) -> Set[Connection]:
        """Get all connections involving a cube."""
        return self._cube_connections.get(cube_id, set()).copy()
    
    def get_connected_cubes(self, cube_id: int) -> Set[int]:
        """Get all cube IDs connected to a given cube."""
        return self._adjacency.get(cube_id, set()).copy()
    
    def are_connected(self, cube_id_1: int, cube_id_2: int) -> bool:
        """Check if two cubes are directly connected."""
        if cube_id_1 not in self._adjacency:
            return False
        return cube_id_2 in self._adjacency[cube_id_1]
    
    def get_connection_between(self, cube_id_1: int, cube_id_2: int) -> Optional[Connection]:
        """Get the connection between two cubes, if it exists."""
        if not self.are_connected(cube_id_1, cube_id_2):
            return None
        
        for conn in self._cube_connections.get(cube_id_1, set()):
            if conn.involves_cube(cube_id_2):
                return conn
        return None
    
    def find_connected_components(self) -> list[Set[int]]:
        """
        Find all connected components in the graph.
        
        Returns:
            List of sets, where each set contains cube IDs in one component
        """
        if not self._adjacency:
            return []
        
        visited = set()
        components = []
        
        for start_cube in self._adjacency.keys():
            if start_cube in visited:
                continue
            
            # BFS to find all cubes in this component
            component = set()
            queue = [start_cube]
            
            while queue:
                cube_id = queue.pop(0)
                if cube_id in visited:
                    continue
                
                visited.add(cube_id)
                component.add(cube_id)
                
                for neighbor in self._adjacency.get(cube_id, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def is_fully_connected(self) -> bool:
        """Check if all cubes form a single connected component."""
        components = self.find_connected_components()
        return len(components) <= 1
    
    def would_disconnect(self, cube_id: int) -> bool:
        """
        Check if removing a cube would split the swarm into multiple components.
        
        This is useful for determining if a cube can safely move.
        """
        if cube_id not in self._adjacency:
            return False
        
        neighbors = list(self._adjacency[cube_id])
        if len(neighbors) <= 1:
            return False  # Leaf node or isolated, can't disconnect others
        
        # Temporarily remove this cube and check connectivity between its neighbors
        # We just need to verify all neighbors can still reach each other
        
        # BFS from first neighbor, avoiding the cube being checked
        start = neighbors[0]
        reachable = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for next_cube in self._adjacency.get(current, set()):
                if next_cube != cube_id and next_cube not in reachable:
                    reachable.add(next_cube)
                    queue.append(next_cube)
        
        # Check if all other neighbors are reachable
        for neighbor in neighbors[1:]:
            if neighbor not in reachable:
                return True  # This neighbor would be disconnected
        
        return False
    
    def get_all_connections(self) -> Set[Connection]:
        """Get all connections."""
        return self._connections.copy()
    
    def copy(self) -> 'ConnectionGraph':
        """Return a deep copy of this graph."""
        new_graph = ConnectionGraph()
        new_graph._connections = self._connections.copy()
        new_graph._adjacency = {k: v.copy() for k, v in self._adjacency.items()}
        new_graph._cube_connections = {k: v.copy() for k, v in self._cube_connections.items()}
        return new_graph
    
    def __len__(self) -> int:
        """Number of connections."""
        return len(self._connections)
    
    def __repr__(self) -> str:
        num_cubes = len(self._adjacency)
        num_connections = len(self._connections)
        return f"ConnectionGraph({num_cubes} cubes, {num_connections} connections)"