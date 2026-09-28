# tests/test_graph.py
"""Tests for connectivity graph derivation."""

from environment.graph import (
    build_adjacency, is_connected, is_connected_without, connection_face,
)


def test_line_connectivity():
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    assert is_connected(pos)
    adj = build_adjacency(pos)
    assert adj[1] == [0, 2] or set(adj[1]) == {0, 2}


def test_disconnected_detected():
    pos = {0: (0, 0, 0), 1: (5, 0, 0)}
    assert not is_connected(pos)


def test_articulation_point():
    # Line: removing the middle disconnects.
    pos = {0: (0, 0, 0), 1: (1, 0, 0), 2: (2, 0, 0)}
    assert not is_connected_without(pos, 1)
    # Removing an end keeps the rest connected.
    assert is_connected_without(pos, 0)


def test_connection_face():
    assert connection_face((0, 0, 0), (1, 0, 0)) == 0   # +X
    assert connection_face((0, 0, 0), (0, 0, -1)) == 5  # -Z