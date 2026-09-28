# environment/graph.py
"""
Derive the connectivity graph from geometry.

Two modules are connected if they occupy face-adjacent lattice cells
(Manhattan distance 1). Connectivity is derived from geometry, NOT stored
independently, exactly as the spec requires.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple
from collections import deque

from .geometry import DIRECTIONS, dir_index, OPPOSITE_FACE


def build_adjacency(positions_by_id: Dict[int, Tuple[int, int, int]]
                    ) -> Dict[int, List[int]]:
    """Return adjacency list: module_id -> list of connected module_ids.

    Two modules are adjacent iff their positions differ by exactly one unit
    along a single axis.
    """
    pos_to_id: Dict[Tuple[int, int, int], int] = {
        pos: mid for mid, pos in positions_by_id.items()
    }
    adjacency: Dict[int, List[int]] = {mid: [] for mid in positions_by_id}
    for mid, pos in positions_by_id.items():
        for d in DIRECTIONS:
            neighbor_pos = (pos[0] + d[0], pos[1] + d[1], pos[2] + d[2])
            if neighbor_pos in pos_to_id:
                adjacency[mid].append(pos_to_id[neighbor_pos])
    return adjacency


def connection_face(pos_a: Tuple[int, int, int],
                    pos_b: Tuple[int, int, int]) -> int:
    """World face index of module A that touches module B.

    Requires A and B to be face-adjacent.
    """
    d = (pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2])
    return dir_index(d)


def is_connected(positions_by_id: Dict[int, Tuple[int, int, int]]) -> bool:
    """Return True if the modules form a single connected component."""
    if not positions_by_id:
        return True
    adjacency = build_adjacency(positions_by_id)
    start = next(iter(positions_by_id))
    seen: Set[int] = {start}
    queue = deque([start])
    while queue:
        node = queue.popleft()
        for nb in adjacency[node]:
            if nb not in seen:
                seen.add(nb)
                queue.append(nb)
    return len(seen) == len(positions_by_id)


def is_connected_without(positions_by_id: Dict[int, Tuple[int, int, int]],
                         excluded_id: int) -> bool:
    """Check connectivity of the structure when one module is removed.

    Used to verify a module is not an articulation point (cut vertex) whose
    movement would disconnect the structure. If the remaining structure is
    empty, we treat it as connected (trivially).
    """
    reduced = {mid: pos for mid, pos in positions_by_id.items()
               if mid != excluded_id}
    return is_connected(reduced)


def count_neighbors(positions_by_id: Dict[int, Tuple[int, int, int]],
                    module_id: int) -> int:
    adjacency = build_adjacency(positions_by_id)
    return len(adjacency[module_id])