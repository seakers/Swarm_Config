# environment/geometry.py
"""
Core geometry for cube modules on a discrete 3D lattice.

Design decisions:
- Positions live on an integer lattice: (x, y, z).
- Orientations are represented as elements of the 24-element rotation group
  of a cube (the proper rotation group, no reflections).
- Face capabilities are attached to the module in its *local* frame; the
  orientation maps local faces to world-facing directions.

This module contains NO mission logic and NO RL logic.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# ---------------------------------------------------------------------------
# Directions
# ---------------------------------------------------------------------------
# We use a canonical ordering for the six axis-aligned unit directions.
# Index -> direction vector.  Also give them names for the six cube faces.

DIRECTIONS: List[Tuple[int, int, int]] = [
    (1, 0, 0),    # 0 +X
    (-1, 0, 0),   # 1 -X
    (0, 1, 0),    # 2 +Y
    (0, -1, 0),   # 3 -Y
    (0, 0, 1),    # 4 +Z
    (0, 0, -1),   # 5 -Z
]

DIRECTION_NAMES = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

# Opposite face index for each face index.
OPPOSITE_FACE = [1, 0, 3, 2, 5, 4]

DIR_TO_INDEX: Dict[Tuple[int, int, int], int] = {d: i for i, d in enumerate(DIRECTIONS)}


def dir_index(vec: Tuple[int, int, int]) -> int:
    return DIR_TO_INDEX[tuple(int(v) for v in vec)]


# ---------------------------------------------------------------------------
# Cube rotation group (24 proper rotations)
# ---------------------------------------------------------------------------
# Each orientation is a 3x3 integer rotation matrix mapping *local* coordinates
# to *world* coordinates.  We enumerate the 24 rotations once and store them.

def _all_rotation_matrices() -> List[np.ndarray]:
    """Generate the 24 proper rotation matrices of the cube."""
    mats = []
    # All signed permutations of axes with determinant +1.
    base = np.eye(3, dtype=int)
    from itertools import permutations, product
    for perm in permutations(range(3)):
        for signs in product([1, -1], repeat=3):
            m = np.zeros((3, 3), dtype=int)
            for row, col in enumerate(perm):
                m[row, col] = signs[row]
            if round(np.linalg.det(m)) == 1:
                mats.append(m)
    # There should be exactly 24.
    assert len(mats) == 24, f"Expected 24 rotations, got {len(mats)}"
    return mats


ROTATIONS: List[np.ndarray] = _all_rotation_matrices()
NUM_ORIENTATIONS = len(ROTATIONS)  # 24

# Precompute a canonical index for each rotation matrix (by bytes) for fast lookup.
_ROT_KEY_TO_INDEX: Dict[bytes, int] = {
    ROTATIONS[i].tobytes(): i for i in range(NUM_ORIENTATIONS)
}


def rotation_index(mat: np.ndarray) -> int:
    """Return the canonical index of a rotation matrix."""
    key = np.asarray(mat, dtype=int).tobytes()
    return _ROT_KEY_TO_INDEX[key]


def compose_orientation(o_outer: int, o_inner: int) -> int:
    """Compose two orientations: result = o_outer applied after o_inner.

    R_result = R_outer @ R_inner
    """
    m = ROTATIONS[o_outer] @ ROTATIONS[o_inner]
    return rotation_index(m)


def world_face_direction(orientation: int, local_face: int) -> Tuple[int, int, int]:
    """Given an orientation and a local face index, return the world direction
    that local face points to."""
    local_vec = np.array(DIRECTIONS[local_face], dtype=int)
    world_vec = ROTATIONS[orientation] @ local_vec
    return tuple(int(v) for v in world_vec)


def world_face_capability_map(orientation: int) -> Dict[int, int]:
    """Return mapping: world_face_index -> local_face_index for a given orientation.

    This tells you, for each world-facing direction, which local face is
    currently pointing that way (so you can look up its capability).
    """
    mapping = {}
    for local_face in range(6):
        wdir = world_face_direction(orientation, local_face)
        world_face = dir_index(wdir)
        mapping[world_face] = local_face
    return mapping


# ---------------------------------------------------------------------------
# Face capability enumeration
# ---------------------------------------------------------------------------
class FaceCapability:
    """Enumeration of abstract face capabilities."""
    NONE = 0
    SOLAR_PANEL = 1
    ANTENNA = 2
    SCIENCE_INSTRUMENT = 3
    RADIATOR = 4
    STRUCTURAL = 5  # generic connector-only face

    NAMES = {
        0: "NONE",
        1: "SOLAR_PANEL",
        2: "ANTENNA",
        3: "SCIENCE_INSTRUMENT",
        4: "RADIATOR",
        5: "STRUCTURAL",
    }



from enum import IntEnum

# ---------------------------------------------------------------------------
# The 12 edges of a cube.
# Each edge is defined by the two (local) face normals whose faces meet at it.
# We store: the two face-normal directions and the axis the edge runs along.
# ---------------------------------------------------------------------------

# Local face normals (same as DIRECTIONS): +X,-X,+Y,-Y,+Z,-Z
# An edge is the intersection of two perpendicular faces.

class Edge(IntEnum):
    PX_PY = 0   # +X & +Y  (runs along Z)
    PX_NY = 1   # +X & -Y
    NX_PY = 2   # -X & +Y
    NX_NY = 3   # -X & -Y
    PX_PZ = 4   # +X & +Z  (runs along Y)
    PX_NZ = 5   # +X & -Z
    NX_PZ = 6   # -X & +Z
    NX_NZ = 7   # -X & -Z
    PY_PZ = 8   # +Y & +Z  (runs along X)
    PY_NZ = 9   # +Y & -Z
    NY_PZ = 10  # -Y & +Z
    NY_NZ = 11  # -Y & -Z


# For each edge: the two local face-normal vectors that meet at it.
EDGE_FACE_NORMALS = {
    Edge.PX_PY: ((1, 0, 0), (0, 1, 0)),
    Edge.PX_NY: ((1, 0, 0), (0, -1, 0)),
    Edge.NX_PY: ((-1, 0, 0), (0, 1, 0)),
    Edge.NX_NY: ((-1, 0, 0), (0, -1, 0)),
    Edge.PX_PZ: ((1, 0, 0), (0, 0, 1)),
    Edge.PX_NZ: ((1, 0, 0), (0, 0, -1)),
    Edge.NX_PZ: ((-1, 0, 0), (0, 0, 1)),
    Edge.NX_NZ: ((-1, 0, 0), (0, 0, -1)),
    Edge.PY_PZ: ((0, 1, 0), (0, 0, 1)),
    Edge.PY_NZ: ((0, 1, 0), (0, 0, -1)),
    Edge.NY_PZ: ((0, -1, 0), (0, 0, 1)),
    Edge.NY_NZ: ((0, -1, 0), (0, 0, -1)),
}


def edge_face_normals(edge: Edge):
    """The two local face-normal unit vectors meeting at this edge."""
    return EDGE_FACE_NORMALS[edge]


def edge_axis(edge: Edge) -> int:
    """The local coordinate axis (0=x,1=y,2=z) the edge runs parallel to.

    The edge runs along the axis perpendicular to both its face normals.
    """
    n1, n2 = EDGE_FACE_NORMALS[edge]
    ax = np.cross(np.array(n1), np.array(n2))
    return int(np.argmax(np.abs(ax)))


def edge_position_offset(edge: Edge):
    """Offset from cube center to the midpoint of the edge (local frame).

    Cube has half-extent 0.5; the edge midpoint sits at (n1 + n2) * 0.5.
    """
    n1, n2 = EDGE_FACE_NORMALS[edge]
    return tuple(0.5 * (a + b) for a, b in zip(n1, n2))


def get_global_face_normal(orientation: int, local_normal) -> Tuple[int, int, int]:
    """Map a local face normal to world coordinates for a given orientation."""
    v = ROTATIONS[orientation] @ np.array(local_normal, dtype=int)
    return tuple(int(x) for x in v)


def rotate_90(orientation: int, global_axis: int, sign: int) -> int:
    """Return the orientation after a 90-degree world-frame rotation about the
    given global axis (0=x,1=y,2=z) in the given sign (+1/-1)."""
    axis_vec = np.zeros(3, dtype=int)
    axis_vec[global_axis] = sign
    R90 = _rotation_90_matrix(axis_vec)
    new_R = np.rint(R90 @ ROTATIONS[orientation]).astype(int)
    return rotation_index(new_R)


def get_local_face_for_direction(orientation: int, world_dir) -> int:
    """Which local face index currently points in the given world direction."""
    mapping = world_face_capability_map(orientation)  # world_face -> local_face
    return mapping[dir_index(tuple(int(v) for v in world_dir))]


def _rotation_90_matrix(axis_vec: np.ndarray) -> np.ndarray:
    """Integer 90-degree rotation matrix about a signed unit axis."""
    x, y, z = (int(round(v)) for v in axis_vec)
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=int)
    return np.eye(3, dtype=int) + K + (K @ K)  # Rodrigues, theta=90