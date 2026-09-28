# tests/test_transitions.py
"""Tests for legal move generation and single-move application."""

from environment.transitions import (
    legal_moves_for_module, all_legal_single_moves, apply_single_move,
    resolve_hinge,
)
from environment.geometry import Edge
from environment.graph import is_connected


def _line(n):
    return {i: (i, 0, 0) for i in range(n)}


def test_isolated_module_cannot_move():
    pos = {0: (0, 0, 0)}
    orient = {0: 0}
    assert legal_moves_for_module(0, pos, orient) == {}


def test_middle_of_line_cannot_move_disconnect():
    # Middle module is a cut vertex -> no legal moves.
    pos = _line(3)
    orient = {i: 0 for i in range(3)}
    assert legal_moves_for_module(1, pos, orient) == {}


def test_end_module_can_pivot():
    # A 2x1 block: an end can pivot around its neighbor.
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0)}
    orient = {i: 0 for i in range(3)}
    # Module 0 has neighbor 1. Pivoting around a shared support.
    moves = legal_moves_for_module(0, pos, orient)
    # Every produced move must keep structure connected.
    for action, (dest, _) in moves.items():
        trial = dict(pos)
        trial[0] = dest
        assert is_connected(trial)


def test_apply_move_does_not_mutate():
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0)}
    orient = {i: 0 for i in range(3)}
    moves = legal_moves_for_module(0, pos, orient)
    if moves:
        action = next(iter(moves))
        new_pos, new_orient, ok = apply_single_move(pos, orient, 0, action)
        assert ok
        assert pos[0] == (0, 0, 0)  # original unchanged


def test_stay_always_valid():
    pos = _line(3)
    orient = {i: 0 for i in range(3)}
    new_pos, new_orient, ok = apply_single_move(pos, orient, 1, 0)
    assert ok
    assert new_pos == pos


def test_line_interior_frozen_ends_move():
    pos = {i: (i, 0, 0) for i in range(4)}
    orient = {i: 0 for i in range(4)}
    assert legal_moves_for_module(1, pos, orient) == {}
    assert legal_moves_for_module(2, pos, orient) == {}
    assert len(legal_moves_for_module(0, pos, orient)) > 0
    assert len(legal_moves_for_module(3, pos, orient)) > 0


# tests/test_transitions.py  (continued)

def test_line_end_uses_180_degree_move():
    """M0 in a line has no 90-degree support, so it must swing 180 degrees and
    land diagonally adjacent to its neighbor."""
    pos = {i: (i, 0, 0) for i in range(4)}
    orient = {i: 0 for i in range(4)}
    moves = legal_moves_for_module(0, pos, orient)
    assert len(moves) > 0
    for action, (dest, _) in moves.items():
        # Every landing must be face-adjacent to the remaining structure and
        # keep the whole thing connected + non-overlapping.
        trial = dict(pos)
        trial[0] = dest
        assert is_connected(trial)
        assert len(set(trial.values())) == len(trial)
        # 180-degree landings for a line end are diagonal: |dx|+|dy|+|dz| == 2
        # relative to M0's start, landing next to M1 at (1,0,0).
        d = (dest[0] - 0, dest[1] - 0, dest[2] - 0)
        assert sum(abs(x) for x in d) == 2


def test_specified_negative_pivot_about_PX_PY():
    """The exact example: negative rotation about the +X/+Y edge of a line-end.
    Expected 180-degree landing at (1,1,0)."""
    pos = {0: (0, 0, 0), 1: (1, 0, 0)}
    orient = {0: 0, 1: 0}
    rm = resolve_hinge((0, 0, 0), 0, 0, Edge.PX_PY, -1, pos,
                       require_connected=True)
    assert rm.success
    assert rm.degrees == 180
    assert rm.new_position == (1, 1, 0)


def test_90_degree_when_support_present():
    """If there IS a landing substrate at the 90-degree cell, we get a 90-degree
    move instead of 180."""
    # L-shape: M0 at origin, M1 at +X support, M2 at (1,1,0) provides the
    # landing substrate at M0's 90-degree destination (0,1,0)'s neighbor.
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (1, 1, 0)}
    orient = {0: 0, 1: 0, 2: 0}
    rm = resolve_hinge((0, 0, 0), 0, 0, Edge.PX_PY, -1, pos,
                       require_connected=True)
    assert rm.success
    # 90-degree destination for negative PX_PY is (0,1,0); it is empty and has
    # landing support from M2 at (1,1,0).
    assert rm.degrees == 90
    assert rm.new_position == (0, 1, 0)


def test_apply_move_does_not_mutate():
    pos = {0: (0, 0, 0), 1: (1, 0, 0)}
    orient = {0: 0, 1: 0}
    moves = legal_moves_for_module(0, pos, orient)
    if moves:
        action = next(iter(moves))
        new_pos, new_orient, ok = apply_single_move(pos, orient, 0, action)
        assert ok
        assert pos[0] == (0, 0, 0)  # original unchanged


def test_orientation_stays_valid_after_pivot():
    """Every pivot must leave the module in one of the 24 valid orientations."""
    from environment.geometry import NUM_ORIENTATIONS
    pos = {0: (0, 0, 0), 1: (1, 0, 0)}
    orient = {0: 0, 1: 0}
    for action, (dest, new_o) in legal_moves_for_module(0, pos, orient).items():
        assert 0 <= new_o < NUM_ORIENTATIONS