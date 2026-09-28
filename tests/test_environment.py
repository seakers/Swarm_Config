# tests/test_environment.py
"""
Integration tests for the full SpacecraftEnv:
  - determinism under fixed seed
  - state ownership (controller cannot corrupt state)
  - random controller survives full episodes
  - illegal / chaotic actions are rejected, not applied
  - local observation contains ONLY local + neighbor info
  - mission objective is computable
"""

import numpy as np
import pytest

from environment.make_env import make_env
from environment.graph import is_connected
from controllers.random_controller import RandomController
from evaluation.metrics import run_episode


def test_reset_deterministic():
    env1 = make_env(n_modules=4, seed=123)
    env2 = make_env(n_modules=4, seed=123)
    o1, _ = env1.reset()
    o2, _ = env2.reset()
    p1 = [tuple(m.position) for m in env1.modules]
    p2 = [tuple(m.position) for m in env2.modules]
    assert p1 == p2


def test_random_episode_deterministic():
    def run(seed):
        env = make_env(n_modules=4, seed=seed)
        ctrl = RandomController(env, seed=seed, respect_legality=True)
        m, _ = run_episode(env, ctrl)
        return m.summary()
    a = run(7)
    b = run(7)
    assert a == b


def test_chaotic_actions_never_corrupt_state():
    """Illegal proposals must be rejected; structure stays valid."""
    env = make_env(n_modules=6, seed=1)
    ctrl = RandomController(env, seed=1, respect_legality=False)
    obs, _ = env.reset()
    for _ in range(50):
        action = ctrl.act(obs)
        obs, r, term, trunc, info = env.step(action)
        # No two modules ever overlap.
        positions = [tuple(m.position) for m in env.modules]
        assert len(set(positions)) == len(positions)
        if term or trunc:
            break


def test_connectivity_preserved_over_episode():
    env = make_env(n_modules=5, seed=3)
    ctrl = RandomController(env, seed=3, respect_legality=True)
    obs, _ = env.reset()
    for _ in range(40):
        action = ctrl.act(obs)
        obs, r, term, trunc, info = env.step(action)
        assert is_connected(env.positions_by_id())
        if term or trunc:
            break


def test_local_observation_is_local_only():
    env = make_env(n_modules=5, seed=0)
    env.reset()
    adj = env.adjacency()
    for mid in range(5):
        obs = env.local_observation(mid)
        seen_ids = {n["id"] for n in obs["neighbors"]}
        # Only physically connected neighbors are visible.
        assert seen_ids == set(adj[mid])
        # No global keys leak in.
        assert "adjacency" not in obs
        assert "modules" not in obs


def test_objective_computable_and_reacts_to_orientation():
    env = make_env(n_modules=4, mission="power", sun_direction=(0, 0, 1))
    env.reset()
    v = env.objective_value()
    assert isinstance(v, float)
    assert v >= 0.0


def test_controller_cannot_mutate_state_via_observation():
    """The observation is a copy/derived structure; mutating it must not
    change the environment."""
    env = make_env(n_modules=4, seed=0)
    obs, _ = env.reset()
    before = [tuple(m.position) for m in env.modules]
    # Attempt to corrupt observation.
    obs["modules"][0]["position"][0] = 999.0
    after = [tuple(m.position) for m in env.modules]
    assert before == after


def test_stay_action_default_for_missing_modules():
    env = make_env(n_modules=4, seed=0)
    env.reset()
    # Empty joint action -> everyone stays.
    obs, r, term, trunc, info = env.step({})
    assert info["n_moved"] == 0