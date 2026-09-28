"""
Pivoting-cube movement system, ported from the reference HingeMove design.

An action = pivot around one of the 12 cube edges in a direction (+1/-1).
The rotation resolves to 90 degrees if there is landing support at the 90-degree
cell, otherwise 180 degrees. Physics is computed by rotating the cube-center
vector about the pivot point. Swept cells are checked for collisions.

This module is the single geometric source of truth for reconfiguration.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from .geometry import (
    Edge, edge_axis, edge_position_offset, get_global_face_normal,
    rotate_90, get_local_face_for_direction, ROTATIONS,
)
from .graph import is_connected_without


Pos = Tuple[int, int, int]

ACTION_STAY = 0


# ---------------------------------------------------------------------------
# Action indexing: (edge, direction) -> action id (1..24). STAY = 0.
# ---------------------------------------------------------------------------
HINGE_TABLE: List[Tuple[Edge, int]] = [
    (edge, direction) for edge in Edge for direction in (+1, -1)
]
NUM_HINGES = len(HINGE_TABLE)      # 24
NUM_ACTIONS = 1 + NUM_HINGES       # 25


def action_to_hinge(action: int) -> Optional[Tuple[Edge, int]]:
    if action == ACTION_STAY:
        return None
    return HINGE_TABLE[action - 1]


# ---------------------------------------------------------------------------
# Core pivot math (ported).
# ---------------------------------------------------------------------------
def _rotate_vector_90(vec: np.ndarray, axis: int, direction: int) -> np.ndarray:
    c, s = 0, direction
    if axis == 0:
        rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    elif axis == 1:
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    else:
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    return rot @ vec


def _compute_pivot_result(pos: Pos, orientation: int, edge: Edge,
                          direction: int, degrees: int
                          ) -> Tuple[Pos, int]:
    """Where the cube ends up after pivoting `degrees` around `edge`."""
    local_axis = edge_axis(edge)
    edge_offset_local = np.array(edge_position_offset(edge), dtype=float)
    edge_offset_global = ROTATIONS[orientation] @ edge_offset_local
    pivot_point = np.array(pos, dtype=float) + edge_offset_global
    to_center = np.array(pos, dtype=float) - pivot_point

    # World axis the edge runs along, with sign.
    axis_local = np.zeros(3)
    axis_local[local_axis] = 1
    axis_global = ROTATIONS[orientation] @ axis_local
    global_axis = int(np.argmax(np.abs(axis_global)))
    axis_sign = int(np.sign(axis_global[global_axis]))
    rot_sign = direction * axis_sign

    steps = 1 if degrees == 90 else 2
    rotated = to_center
    new_orient = orientation
    for _ in range(steps):
        rotated = _rotate_vector_90(rotated, global_axis, rot_sign)
        new_orient = rotate_90(new_orient, global_axis, rot_sign)

    new_pos_f = pivot_point + rotated
    new_pos = tuple(int(round(x)) for x in new_pos_f)
    return new_pos, new_orient


def _has_landing_support(moving_id: int, landing: Pos,
                         positions_by_id: Dict[int, Pos]) -> bool:
    """Any cube (other than the mover) face-adjacent to the landing cell."""
    occ = {p: i for i, p in positions_by_id.items()}
    for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        adj = (landing[0]+d[0], landing[1]+d[1], landing[2]+d[2])
        nid = occ.get(adj)
        if nid is not None and nid != moving_id:
            return True
    return False


def _get_pivot_neighbor(pos: Pos, orientation: int, edge: Edge,
                        positions_by_id: Dict[int, Pos]) -> Optional[int]:
    """Find a neighbor cube that provides the pivot point for this edge.

    Checks the two faces adjacent to the edge, and the corner cell where the
    two faces meet. Since connectivity here is derived from geometry, any
    face-adjacent occupied cell counts as "connected".
    """
    from .geometry import edge_face_normals
    occ = {p: i for i, p in positions_by_id.items()}
    n1_local, n2_local = edge_face_normals(edge)

    # Two adjacent faces.
    for local_normal in (n1_local, n2_local):
        gnorm = get_global_face_normal(orientation, local_normal)
        npos = (pos[0]+gnorm[0], pos[1]+gnorm[1], pos[2]+gnorm[2])
        if npos in occ:
            return occ[npos]

    # Corner cell (both faces' normals summed).
    g1 = get_global_face_normal(orientation, n1_local)
    g2 = get_global_face_normal(orientation, n2_local)
    corner = (pos[0]+g1[0]+g2[0], pos[1]+g1[1]+g2[1], pos[2]+g1[2]+g2[2])
    if corner in occ:
        return occ[corner]

    return None


def _compute_swept_cells(pos: Pos, orientation: int, edge: Edge,
                         direction: int, degrees: int) -> Set[Pos]:
    """Grid cells the cube passes through during the pivot (ported)."""
    swept: Set[Pos] = set()

    pos_90, _ = _compute_pivot_result(pos, orientation, edge, direction, 90)

    edge_offset_local = np.array(edge_position_offset(edge), dtype=float)
    edge_offset_global = ROTATIONS[orientation] @ edge_offset_local
    edge_offset = np.rint(edge_offset_global).astype(int)  # near-integer for axis-aligned

    old_to_new = np.array(pos_90) - np.array(pos)
    sweep_diff = old_to_new - 2 * edge_offset_global

    swept.add(tuple(int(round(pos[i] + sweep_diff[i])) for i in range(3)))
    swept.add(tuple(int(round(pos_90[i] + sweep_diff[i])) for i in range(3)))
    swept.add(pos_90)

    if degrees == 180:
        pos_180, _ = _compute_pivot_result(pos, orientation, edge, direction, 180)
        swept.add(pos_180)
        edge_offset_180 = edge_offset_global - old_to_new
        old_to_new_180 = np.array(pos_180) - np.array(pos_90)
        sweep_diff_180 = old_to_new_180 - 2 * edge_offset_180
        swept.add(tuple(int(round(pos_90[i] + sweep_diff_180[i])) for i in range(3)))
        swept.add(tuple(int(round(pos_180[i] + sweep_diff_180[i])) for i in range(3)))

    return swept


# ---------------------------------------------------------------------------
# Move resolution (ported from _compute_move_result).
# ---------------------------------------------------------------------------
@dataclass
class ResolvedMove:
    success: bool
    reason: str = ""
    new_position: Optional[Pos] = None
    new_orientation: Optional[int] = None
    degrees: int = 0


def resolve_hinge(pos: Pos, orientation: int, module_id: int,
                  edge: Edge, direction: int,
                  positions_by_id: Dict[int, Pos],
                  require_connected: bool = True) -> ResolvedMove:
    """Resolve a single hinge move for one module (dry, no mutation).

    Implements the 90/180 decision:
      1. Pivot edge must have support.
      2. Compute 90 destination. If it's clear AND has landing support -> 90.
      3. Else compute 180 destination; must be clear -> 180.
      4. Swept path must be clear.
      5. (Optional) connectivity preserved.
    """
    occupied = set(positions_by_id.values())

    # 1. Pivot support.
    pivot_neighbor = _get_pivot_neighbor(pos, orientation, edge, positions_by_id)
    if pivot_neighbor is None:
        return ResolvedMove(False, "pivot edge has no support")

    # 2. 90-degree destination.
    pos_90, orient_90 = _compute_pivot_result(pos, orientation, edge, direction, 90)
    dest_90_clear = pos_90 not in occupied
    has_support_90 = _has_landing_support(module_id, pos_90, positions_by_id)

    if dest_90_clear and has_support_90:
        degrees, new_pos, new_orient = 90, pos_90, orient_90
    else:
        # 3. 180-degree.
        pos_180, orient_180 = _compute_pivot_result(
            pos, orientation, edge, direction, 180)
        if pos_180 in occupied:
            reason = ("dest blocked at both 90 and 180" if not dest_90_clear
                      else "no support at 90, and 180 dest blocked")
            return ResolvedMove(False, reason)
        degrees, new_pos, new_orient = 180, pos_180, orient_180

    # 4. Swept path.
    swept = _compute_swept_cells(pos, orientation, edge, direction, degrees)
    for cell in swept:
        if cell != pos and cell != new_pos and cell in occupied:
            return ResolvedMove(False, f"swept path blocked at {cell}")

    # 5. Connectivity.
    if require_connected:
        trial = dict(positions_by_id)
        trial[module_id] = new_pos
        from .graph import is_connected
        if not is_connected(trial):
            return ResolvedMove(False, "move would disconnect swarm")

    return ResolvedMove(True, "", new_pos, new_orient, degrees)


# ---------------------------------------------------------------------------
# Environment-facing API (same signatures as before).
# ---------------------------------------------------------------------------
def legal_moves_for_module(module_id: int,
                           positions_by_id: Dict[int, Pos],
                           orientations_by_id: Dict[int, int],
                           require_connected: bool = True
                           ) -> Dict[int, Tuple[Pos, int]]:
    """Return {action_index: (new_pos, new_orient)} for one module.

    action_index 1..24 maps to HINGE_TABLE[action-1] = (edge, direction).
    """
    pos = positions_by_id[module_id]
    orientation = orientations_by_id[module_id]

    legal: Dict[int, Tuple[Pos, int]] = {}
    for a_idx, (edge, direction) in enumerate(HINGE_TABLE):
        rm = resolve_hinge(pos, orientation, module_id, edge, direction,
                           positions_by_id, require_connected)
        if rm.success:
            legal[a_idx + 1] = (rm.new_position, rm.new_orientation)
    return legal


def all_legal_single_moves(positions_by_id: Dict[int, Pos],
                           orientations_by_id: Dict[int, int],
                           require_connected: bool = True
                           ) -> Dict[int, Dict[int, Tuple[Pos, int]]]:
    return {
        mid: legal_moves_for_module(mid, positions_by_id, orientations_by_id,
                                    require_connected)
        for mid in positions_by_id
    }


def apply_single_move(positions_by_id: Dict[int, Pos],
                      orientations_by_id: Dict[int, int],
                      module_id: int, action: int,
                      require_connected: bool = True
                      ) -> Tuple[Dict[int, Pos], Dict[int, int], bool]:
    new_pos = dict(positions_by_id)
    new_orient = dict(orientations_by_id)
    if action == ACTION_STAY:
        return new_pos, new_orient, True
    legal = legal_moves_for_module(module_id, positions_by_id,
                                   orientations_by_id, require_connected)
    if action not in legal:
        return new_pos, new_orient, False
    dest, dest_orient = legal[action]
    new_pos[module_id] = dest
    new_orient[module_id] = dest_orient
    return new_pos, new_orient, True