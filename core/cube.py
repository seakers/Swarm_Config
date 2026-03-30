import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Set
from enum import Enum, auto


class Face(Enum):
    """
    The six faces of a cube, defined by their outward normal direction.
    Using aerospace/right-hand coordinate convention:
    - X+ is "front" (direction of travel)
    - Y+ is "left" 
    - Z+ is "up"
    """
    POS_X = auto()  # +X face (front)
    NEG_X = auto()  # -X face (back)
    POS_Y = auto()  # +Y face (left)
    NEG_Y = auto()  # -Y face (right)
    POS_Z = auto()  # +Z face (top)
    NEG_Z = auto()  # -Z face (bottom)
    
    def opposite(self) -> 'Face':
        """Return the opposite face."""
        opposites = {
            Face.POS_X: Face.NEG_X,
            Face.NEG_X: Face.POS_X,
            Face.POS_Y: Face.NEG_Y,
            Face.NEG_Y: Face.POS_Y,
            Face.POS_Z: Face.NEG_Z,
            Face.NEG_Z: Face.POS_Z,
        }
        return opposites[self]
    
    def normal_vector(self) -> Tuple[int, int, int]:
        """Return the unit normal vector for this face."""
        normals = {
            Face.POS_X: (1, 0, 0),
            Face.NEG_X: (-1, 0, 0),
            Face.POS_Y: (0, 1, 0),
            Face.NEG_Y: (0, -1, 0),
            Face.POS_Z: (0, 0, 1),
            Face.NEG_Z: (0, 0, -1),
        }
        return normals[self]


class Edge(Enum):
    """
    The 12 edges of a cube, named by the two faces they connect.
    Each edge is shared by exactly two faces.
    """
    # Edges parallel to X-axis (connect Y and Z faces)
    POS_Y_POS_Z = auto()  # top-left edge
    POS_Y_NEG_Z = auto()  # bottom-left edge
    NEG_Y_POS_Z = auto()  # top-right edge
    NEG_Y_NEG_Z = auto()  # bottom-right edge
    
    # Edges parallel to Y-axis (connect X and Z faces)
    POS_X_POS_Z = auto()  # top-front edge
    POS_X_NEG_Z = auto()  # bottom-front edge
    NEG_X_POS_Z = auto()  # top-back edge
    NEG_X_NEG_Z = auto()  # bottom-back edge
    
    # Edges parallel to Z-axis (connect X and Y faces)
    POS_X_POS_Y = auto()  # front-left edge
    POS_X_NEG_Y = auto()  # front-right edge
    NEG_X_POS_Y = auto()  # back-left edge
    NEG_X_NEG_Y = auto()  # back-right edge
    
    def get_axis(self) -> int:
        """
        Return which axis this edge is parallel to (0=X, 1=Y, 2=Z).
        """
        x_parallel = {Edge.POS_Y_POS_Z, Edge.POS_Y_NEG_Z, 
                      Edge.NEG_Y_POS_Z, Edge.NEG_Y_NEG_Z}
        y_parallel = {Edge.POS_X_POS_Z, Edge.POS_X_NEG_Z,
                      Edge.NEG_X_POS_Z, Edge.NEG_X_NEG_Z}
        z_parallel = {Edge.POS_X_POS_Y, Edge.POS_X_NEG_Y,
                      Edge.NEG_X_POS_Y, Edge.NEG_X_NEG_Y}
        
        if self in x_parallel:
            return 0
        elif self in y_parallel:
            return 1
        else:
            return 2
    
    def get_adjacent_faces(self) -> Tuple[Face, Face]:
        """Return the two faces that share this edge."""
        edge_faces = {
            Edge.POS_Y_POS_Z: (Face.POS_Y, Face.POS_Z),
            Edge.POS_Y_NEG_Z: (Face.POS_Y, Face.NEG_Z),
            Edge.NEG_Y_POS_Z: (Face.NEG_Y, Face.POS_Z),
            Edge.NEG_Y_NEG_Z: (Face.NEG_Y, Face.NEG_Z),
            Edge.POS_X_POS_Z: (Face.POS_X, Face.POS_Z),
            Edge.POS_X_NEG_Z: (Face.POS_X, Face.NEG_Z),
            Edge.NEG_X_POS_Z: (Face.NEG_X, Face.POS_Z),
            Edge.NEG_X_NEG_Z: (Face.NEG_X, Face.NEG_Z),
            Edge.POS_X_POS_Y: (Face.POS_X, Face.POS_Y),
            Edge.POS_X_NEG_Y: (Face.POS_X, Face.NEG_Y),
            Edge.NEG_X_POS_Y: (Face.NEG_X, Face.POS_Y),
            Edge.NEG_X_NEG_Y: (Face.NEG_X, Face.NEG_Y),
        }
        return edge_faces[self]
    
    def get_position_offset(self) -> Tuple[float, float, float]:
        """
        Return the position of this edge's center relative to cube center.
        Each component is -0.5, 0, or +0.5.
        """
        offsets = {
            # X-parallel edges (x=0)
            Edge.POS_Y_POS_Z: (0.0, 0.5, 0.5),
            Edge.POS_Y_NEG_Z: (0.0, 0.5, -0.5),
            Edge.NEG_Y_POS_Z: (0.0, -0.5, 0.5),
            Edge.NEG_Y_NEG_Z: (0.0, -0.5, -0.5),
            # Y-parallel edges (y=0)
            Edge.POS_X_POS_Z: (0.5, 0.0, 0.5),
            Edge.POS_X_NEG_Z: (0.5, 0.0, -0.5),
            Edge.NEG_X_POS_Z: (-0.5, 0.0, 0.5),
            Edge.NEG_X_NEG_Z: (-0.5, 0.0, -0.5),
            # Z-parallel edges (z=0)
            Edge.POS_X_POS_Y: (0.5, 0.5, 0.0),
            Edge.POS_X_NEG_Y: (0.5, -0.5, 0.0),
            Edge.NEG_X_POS_Y: (-0.5, 0.5, 0.0),
            Edge.NEG_X_NEG_Y: (-0.5, -0.5, 0.0),
        }
        return offsets[self]


# Mapping from face to its four edges
FACE_EDGES: Dict[Face, Tuple[Edge, Edge, Edge, Edge]] = {
    Face.POS_X: (Edge.POS_X_POS_Z, Edge.POS_X_NEG_Z, 
                 Edge.POS_X_POS_Y, Edge.POS_X_NEG_Y),
    Face.NEG_X: (Edge.NEG_X_POS_Z, Edge.NEG_X_NEG_Z,
                 Edge.NEG_X_POS_Y, Edge.NEG_X_NEG_Y),
    Face.POS_Y: (Edge.POS_Y_POS_Z, Edge.POS_Y_NEG_Z,
                 Edge.POS_X_POS_Y, Edge.NEG_X_POS_Y),
    Face.NEG_Y: (Edge.NEG_Y_POS_Z, Edge.NEG_Y_NEG_Z,
                 Edge.POS_X_NEG_Y, Edge.NEG_X_NEG_Y),
    Face.POS_Z: (Edge.POS_Y_POS_Z, Edge.NEG_Y_POS_Z,
                 Edge.POS_X_POS_Z, Edge.NEG_X_POS_Z),
    Face.NEG_Z: (Edge.POS_Y_NEG_Z, Edge.NEG_Y_NEG_Z,
                 Edge.POS_X_NEG_Z, Edge.NEG_X_NEG_Z),
}


@dataclass
class Orientation:
    """
    Represents the orientation of a cube using a rotation matrix.
    
    The rotation matrix R transforms local coordinates to global coordinates:
        global_vec = R @ local_vec
    
    We use rotation matrices because they compose nicely for 90-degree rotations
    and are easy to interpret (each column is where a local axis points globally).
    """
    matrix: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.int8))
    
    def __post_init__(self):
        """Ensure matrix is the right type."""
        self.matrix = np.array(self.matrix, dtype=np.int8)
    
    def rotate_90(self, axis: int, direction: int) -> 'Orientation':
        """
        Return a new orientation after rotating 90 degrees around an axis.
        
        Args:
            axis: 0=X, 1=Y, 2=Z (global axis)
            direction: +1 for CCW (right-hand rule), -1 for CW
            
        Returns:
            New Orientation after rotation
        """
        # 90-degree rotation matrices
        c, s = 0, direction  # cos(90°)=0, sin(90°)=±1
        
        if axis == 0:  # X-axis rotation
            rot = np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s, c]], dtype=np.int8)
        elif axis == 1:  # Y-axis rotation
            rot = np.array([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]], dtype=np.int8)
        else:  # Z-axis rotation
            rot = np.array([[c, -s, 0],
                           [s, c, 0],
                           [0, 0, 1]], dtype=np.int8)
        
        new_matrix = rot @ self.matrix
        return Orientation(new_matrix)
    
    def get_global_face_normal(self, local_face: Face) -> Tuple[int, int, int]:
        """
        Get the global direction that a local face is pointing.
        
        Args:
            local_face: Which face in the cube's local frame
            
        Returns:
            Global direction as (dx, dy, dz) unit vector
        """
        local_normal = np.array(local_face.normal_vector(), dtype=np.int8)
        global_normal = self.matrix @ local_normal
        return tuple(global_normal.tolist())
    
    def get_local_face_for_direction(self, global_dir: Tuple[int, int, int]) -> Face:
        """
        Get which local face is pointing in a given global direction.
        
        Args:
            global_dir: Global direction as (dx, dy, dz)
            
        Returns:
            The Face (in local frame) that points in that direction
        """
        global_vec = np.array(global_dir, dtype=np.int8)
        # R @ local = global, so local = R^T @ global (R is orthogonal)
        local_vec = self.matrix.T @ global_vec
        
        for face in Face:
            if np.array_equal(local_vec, np.array(face.normal_vector())):
                return face
        
        raise ValueError(f"No face points in direction {global_dir}")
    
    def copy(self) -> 'Orientation':
        """Return a deep copy of this orientation."""
        return Orientation(self.matrix.copy())
    
    def __eq__(self, other: 'Orientation') -> bool:
        """Check if two orientations are equal."""
        if not isinstance(other, Orientation):
            return False
        return np.array_equal(self.matrix, other.matrix)
    
    def __hash__(self) -> int:
        """Hash for use in sets/dicts."""
        return hash(self.matrix.tobytes())
    
    def __repr__(self) -> str:
        # Show where each local axis points globally
        x_dir = tuple(self.matrix[:, 0].tolist())
        y_dir = tuple(self.matrix[:, 1].tolist())
        z_dir = tuple(self.matrix[:, 2].tolist())
        return f"Orientation(X→{x_dir}, Y→{y_dir}, Z→{z_dir})"


@dataclass
class Cube:
    """
    Represents a single cubesat unit in the swarm.
    
    Attributes:
        cube_id: Unique identifier for this cube
        position: (x, y, z) integer grid position
        orientation: Rotation state of the cube
        power: Current power level (0.0 to 1.0)
        is_functional: Whether the cube is operational
        has_camera: Whether this cube has a camera
        has_transmitter: Whether this cube has a transmitter
    """
    cube_id: int
    position: Tuple[int, int, int]
    orientation: Orientation = field(default_factory=Orientation)
    power: float = 1.0
    is_functional: bool = True
    
    # Instrument flags (for future use)
    has_camera: bool = True
    has_transmitter: bool = True
    has_receiver: bool = True
    
    def get_neighbor_position(self, face: Face) -> Tuple[int, int, int]:
        """
        Get the grid position of the cell adjacent to a given face.
        
        This uses the GLOBAL direction the face is pointing (accounting for orientation).
        
        Args:
            face: Which face (in local frame) to check
            
        Returns:
            Grid position (x, y, z) of the adjacent cell
        """
        global_normal = self.orientation.get_global_face_normal(face)
        return (
            self.position[0] + global_normal[0],
            self.position[1] + global_normal[1],
            self.position[2] + global_normal[2]
        )
    
    def get_face_pointing_direction(self, global_dir: Tuple[int, int, int]) -> Face:
        """
        Get which of this cube's faces is pointing in a global direction.
        
        Args:
            global_dir: Direction in global frame
            
        Returns:
            The local Face that points in that direction
        """
        return self.orientation.get_local_face_for_direction(global_dir)
    
    def copy(self) -> 'Cube':
        """Return a deep copy of this cube."""
        return Cube(
            cube_id=self.cube_id,
            position=self.position,
            orientation=self.orientation.copy(),
            power=self.power,
            is_functional=self.is_functional,
            has_camera=self.has_camera,
            has_transmitter=self.has_transmitter,
            has_receiver=self.has_receiver
        )
    
    def __hash__(self) -> int:
        return hash(self.cube_id)
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Cube):
            return False
        return self.cube_id == other.cube_id