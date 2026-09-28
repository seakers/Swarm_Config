# tests/test_edges.py
"""Tests for the 12-edge enumeration and orientation helpers."""

import numpy as np
from environment.geometry import (
    Edge, edge_face_normals, edge_axis, edge_position_offset,
    get_global_face_normal, rotate_90, get_local_face_for_direction,
    ROTATIONS, NUM_ORIENTATIONS, world_face_direction,
)


def test_twelve_edges():
    assert len(list(Edge)) == 12


def test_edge_normals_perpendicular():
    for e in Edge:
        n1, n2 = edge_face_normals(e)
        assert np.dot(n1, n2) == 0  # faces meeting at an edge are perpendicular


def test_edge_axis_perpendicular_to_both_normals():
    for e in Edge:
        n1, n2 = edge_face_normals(e)
        ax = edge_axis(e)
        assert n1[ax] == 0 and n2[ax] == 0


def test_edge_offset_is_half_diagonal():
    for e in Edge:
        n1, n2 = edge_face_normals(e)
        off = edge_position_offset(e)
        expected = tuple(0.5 * (a + b) for a, b in zip(n1, n2))
        assert off == expected


def test_rotate_90_stays_valid():
    for o in range(NUM_ORIENTATIONS):
        for axis in range(3):
            for sign in (+1, -1):
                new_o = rotate_90(o, axis, sign)
                assert 0 <= new_o < NUM_ORIENTATIONS


def test_rotate_90_four_times_is_identity():
    for o in range(NUM_ORIENTATIONS):
        cur = o
        for _ in range(4):
            cur = rotate_90(cur, 2, +1)  # spin about +Z four times
        assert cur == o


def test_global_face_normal_matches_world_face_direction():
    for o in range(NUM_ORIENTATIONS):
        # local +X face normal maps to whatever world dir local face 0 points.
        gn = get_global_face_normal(o, (1, 0, 0))
        assert gn == world_face_direction(o, 0)


def test_local_face_for_direction_is_inverse():
    for o in range(NUM_ORIENTATIONS):
        for local_face in range(6):
            wdir = world_face_direction(o, local_face)
            recovered = get_local_face_for_direction(o, wdir)
            assert recovered == local_face