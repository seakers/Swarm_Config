# tests/test_geometry.py
"""Tests for the cube rotation group and orientation math."""

import numpy as np
import pytest

from environment.geometry import (
    ROTATIONS, NUM_ORIENTATIONS, compose_orientation, rotation_index,
    world_face_direction, DIRECTIONS, OPPOSITE_FACE, dir_index,
)


def test_24_rotations():
    assert NUM_ORIENTATIONS == 24


def test_all_rotations_are_proper():
    for R in ROTATIONS:
        assert round(np.linalg.det(R)) == 1
        # Orthonormal integer matrix.
        assert np.array_equal(R @ R.T, np.eye(3, dtype=int))


def test_rotations_unique():
    keys = {R.tobytes() for R in ROTATIONS}
    assert len(keys) == 24


def test_identity_is_present():
    idx = rotation_index(np.eye(3, dtype=int))
    assert 0 <= idx < 24


def test_composition_closed():
    # Composing any two orientations yields a valid orientation index.
    for a in range(NUM_ORIENTATIONS):
        for b in range(NUM_ORIENTATIONS):
            c = compose_orientation(a, b)
            assert 0 <= c < NUM_ORIENTATIONS


def test_composition_associative_sample():
    a, b, c = 3, 7, 19
    left = compose_orientation(compose_orientation(a, b), c)
    right = compose_orientation(a, compose_orientation(b, c))
    assert left == right


def test_world_face_directions_are_permutation():
    # For any orientation, the six local faces map to the six world dirs.
    for o in range(NUM_ORIENTATIONS):
        world_dirs = {world_face_direction(o, lf) for lf in range(6)}
        assert world_dirs == set(DIRECTIONS)


def test_opposite_faces_stay_opposite():
    for o in range(NUM_ORIENTATIONS):
        for lf in range(6):
            wd = world_face_direction(o, lf)
            wd_opp = world_face_direction(o, OPPOSITE_FACE[lf])
            assert tuple(-np.array(wd)) == tuple(wd_opp)