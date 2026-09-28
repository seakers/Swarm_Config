# environment/conflict_resolution.py
"""
Conflict resolution for simultaneous multi-module actions.

Policy (initial, per spec): REJECT-BOTH.
If two proposed actions conflict, neither executes.

Conflict types detected:
  1. Same destination cell (two modules -> same target).
  2. Destination is another module's current cell that is ALSO moving into
     this module's target (swap) -- treated conservatively as conflict.
  3. Destination equals a stationary module's current cell.
  4. Simultaneous moves that would each rely on the other as support
     (support pulled out from under a mover) -- conservative rejection.
  5. Post-move global connectivity violation.

The environment provides per-module legal move tables; this module only
resolves conflicts among *individually legal* proposals.
"""

from __future__ import annotations
from typing import Dict, List, Tuple

from .graph import is_connected

Pos = Tuple[int, int, int]


def resolve(positions_by_id: Dict[int, Pos],
            orientations_by_id: Dict[int, int],
            proposals: Dict[int, Tuple[Pos, int]],
            require_connected: bool = True
            ) -> Tuple[Dict[int, Tuple[Pos, int]], Dict[int, str]]:
    """Resolve simultaneous proposals under reject-both.

    Args:
        positions_by_id: current positions.
        proposals: {module_id: (dest_pos, dest_orient)} -- only moving modules.
                   (STAY modules omitted.)
    Returns:
        accepted: {module_id: (dest_pos, dest_orient)} that will execute.
        rejections: {module_id: reason_str} for rejected proposals.
    """
    rejections: Dict[int, str] = {}
    moving = dict(proposals)

    # --- 1. Same-destination conflicts ---
    dest_count: Dict[Pos, List[int]] = {}
    for mid, (dest, _) in moving.items():
        dest_count.setdefault(dest, []).append(mid)
    for dest, mids in dest_count.items():
        if len(mids) > 1:
            for mid in mids:
                rejections[mid] = f"same_destination:{dest}"

    # --- 2. Destination occupied by a stationary module ---
    moving_ids = set(moving.keys())
    stationary_positions = {
        mid: pos for mid, pos in positions_by_id.items()
        if mid not in moving_ids
    }
    stationary_cells = set(stationary_positions.values())
    for mid, (dest, _) in moving.items():
        if mid in rejections:
            continue
        if dest in stationary_cells:
            rejections[mid] = f"dest_occupied_stationary:{dest}"

    # --- 3. Swap / support-removal conflicts ---
    # A mover's destination equals another mover's *current* cell.
    current_of_mover = {mid: positions_by_id[mid] for mid in moving_ids}
    cell_to_mover = {pos: mid for mid, pos in current_of_mover.items()}
    for mid, (dest, _) in moving.items():
        if mid in rejections:
            continue
        if dest in cell_to_mover and cell_to_mover[dest] != mid:
            other = cell_to_mover[dest]
            # Both reject (conservative reject-both).
            rejections[mid] = f"transit_conflict_with:{other}"
            rejections.setdefault(other, f"transit_conflict_with:{mid}")

    accepted = {mid: mv for mid, mv in moving.items() if mid not in rejections}

    # --- 4. Global connectivity check on the accepted joint move ---
    if require_connected and accepted:
        trial_positions = dict(positions_by_id)
        for mid, (dest, _) in accepted.items():
            trial_positions[mid] = dest
        if not is_connected(trial_positions):
            # Conservative: reject the entire moving set this step.
            for mid in list(accepted.keys()):
                rejections[mid] = "joint_connectivity_violation"
            accepted = {}

    return accepted, rejections