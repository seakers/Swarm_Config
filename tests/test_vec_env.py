# tests/test_vec_env.py
"""Subprocess vec-env must produce identical transitions to the serial path."""

import numpy as np
import pytest

from rl.vec_env import (
    TaskSampler, SubprocVecEnv, SerialVecEnv, make_vec_env,
)


def _samplers(n_envs, base_seed=0):
    return [TaskSampler(base_seed + k, [4, 6], ["power"], "line",
                        (0.0, 0.0, 1.0), (0.0, 1.0, 0.0),
                        max_steps=12, reward_mode="improvement", offset=k)
            for k in range(n_envs)]


def _fixed_actions(state):
    """Deterministic policy: every module takes its first legal non-STAY action
    (falling back to STAY), so both vec envs receive identical actions."""
    joints = []
    for (gobs, lobs, mask, id_order, r, d, obj) in state:
        joint = {}
        for row, mid in enumerate(id_order):
            legal = np.nonzero(mask[row] > 0.5)[0]
            nonstay = [a for a in legal if a != 0]
            joint[mid] = int(nonstay[0]) if nonstay else 0
        joints.append(joint)
    return joints


def test_reset_view_shape():
    vec = SerialVecEnv(_samplers(4))
    state = vec.reset()
    assert len(state) == 4
    for (gobs, lobs, mask, id_order, r, d, obj) in state:
        assert mask.shape[0] == len(id_order)
        assert r == 0.0 and d is False
        assert np.isfinite(obj)
    vec.close()


def test_subproc_matches_serial():
    n_envs, n_steps = 4, 8
    ser = SerialVecEnv(_samplers(n_envs, base_seed=7))
    sub = SubprocVecEnv(_samplers(n_envs, base_seed=7), n_workers=2)

    s_state = ser.reset()
    p_state = sub.reset()

    # Initial objectives must match env-for-env.
    assert [round(x[6], 6) for x in s_state] == [round(x[6], 6) for x in p_state]

    for _ in range(n_steps):
        s_actions = _fixed_actions(s_state)
        p_actions = _fixed_actions(p_state)
        assert s_actions == p_actions           # same policy, same state

        s_state = ser.step(s_actions)
        p_state = sub.step(p_actions)

        s_rew = [round(x[4], 6) for x in s_state]
        p_rew = [round(x[4], 6) for x in p_state]
        s_done = [x[5] for x in s_state]
        p_done = [x[5] for x in p_state]
        s_obj = [round(x[6], 6) for x in s_state]
        p_obj = [round(x[6], 6) for x in p_state]

        assert s_rew == p_rew
        assert s_done == p_done
        assert s_obj == p_obj

    ser.close()
    sub.close()


def test_fallback_to_serial():
    vec = make_vec_env(_samplers(2), n_workers=1, use_subproc=True)
    assert isinstance(vec, SerialVecEnv)
    vec.close()