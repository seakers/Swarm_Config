# tests/test_conflict_resolution.py
"""Tests for simultaneous action conflict resolution (reject-both)."""

from environment.conflict_resolution import resolve


def test_same_destination_rejects_both():
    pos = {0: (0, 0, 0), 1: (2, 0, 0), 2: (1, 1, 0)}
    proposals = {
        0: ((1, 0, 0), 0),
        1: ((1, 0, 0), 0),
    }
    accepted, rej = resolve(pos, {i: 0 for i in pos}, proposals,
                            require_connected=False)
    assert 0 in rej and 1 in rej
    assert accepted == {}


def test_non_conflicting_accepted():
    pos = {0: (0, 0, 0), 1: (3, 0, 0), 2: (1, 0, 0), 3: (2, 0, 0)}
    proposals = {
        0: ((0, 1, 0), 0),
        1: ((3, 1, 0), 0),
    }
    accepted, rej = resolve(pos, {i: 0 for i in pos}, proposals,
                            require_connected=False)
    assert set(accepted.keys()) == {0, 1}
    assert rej == {}


def test_transit_conflict():
    pos = {0: (0, 0, 0), 1: (1, 0, 0)}
    # 0 wants to move into 1's current cell.
    proposals = {0: ((1, 0, 0), 0)}
    accepted, rej = resolve(pos, {i: 0 for i in pos}, proposals,
                            require_connected=False)
    # Destination occupied by stationary module 1 -> rejected.
    assert 0 in rej
    assert accepted == {}


def test_connectivity_rejection():
    # Accepting a move that disconnects the joint structure is rejected.
    pos = {0: (0, 0, 0), 1: (1, 0, 0)}
    proposals = {1: ((1, 5, 0), 0)}  # would fly off, disconnecting
    accepted, rej = resolve(pos, {i: 0 for i in pos}, proposals,
                            require_connected=True)
    assert accepted == {}
    assert 1 in rej