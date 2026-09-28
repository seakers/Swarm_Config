# environment/configuration_builder.py
"""
Helpers to build initial connected configurations of N modules with sensible
default face capabilities. Keeps N a parameter, never hard-coded.
"""

from __future__ import annotations
from typing import List, Tuple

from .geometry import FaceCapability

Pos = Tuple[int, int, int]


def line_configuration(n: int) -> List[Pos]:
    """N modules in a straight connected line along +X."""
    return [(x, 0, 0) for x in range(n)]


def L_configuration(n: int) -> List[Pos]:
    """An L-shape (connected) for variety."""
    positions = []
    half = (n + 1) // 2
    for x in range(half):
        positions.append((x, 0, 0))
    for y in range(1, n - half + 1):
        positions.append((half - 1, y, 0))
    return positions[:n]


def rand_configuration(n: int, seed: int = 0) -> List[Pos]:
    """Random connected configuration of N modules.  Uses a fixed seed for
    reproducibility."""
    import random
    random.seed(seed)
    positions = [(0, 0, 0)]
    while len(positions) < n:
        # Pick a random existing module and a random direction to add a new one.
        base = random.choice(positions)
        dir = random.choice([(1, 0, 0), (-1, 0, 0),
                             (0, 1, 0), (0, -1, 0),
                             (0, 0, 1), (0, 0, -1)])
        new_pos = (base[0] + dir[0], base[1] + dir[1], base[2] + dir[2])
        if new_pos not in positions:
            positions.append(new_pos)
    return positions


def default_face_capabilities(n: int) -> List[List[int]]:
    """Give every module solar panels on +Z (local faces 4), an antenna
    on +X (local face 0), a radiator on -X (local 1), science on +Y (2)."""
    caps = []
    for _ in range(n):
        module_caps = [FaceCapability.STRUCTURAL] * 6
        module_caps[4] = FaceCapability.SOLAR_PANEL   # +Z
        module_caps[0] = FaceCapability.ANTENNA       # +X
        module_caps[1] = FaceCapability.RADIATOR      # -X
        module_caps[2] = FaceCapability.SCIENCE_INSTRUMENT  # +Y
        caps.append(module_caps)
    return caps


def default_orientations(n: int) -> List[int]:
    """All modules start in canonical orientation 0 (identity)."""
    return [0] * n


def random_orientations(n: int, seed: int = 0) -> List[int]:
    """Random orientations for each module, using a fixed seed for
    reproducibility."""
    import random
    random.seed(seed)
    return [random.randint(0, 23) for _ in range(n)]