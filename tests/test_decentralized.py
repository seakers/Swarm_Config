# tests/test_decentralized.py
"""
Tests for the decentralized GNN + CTDE.

The MOST IMPORTANT tests here verify that decentralized EXECUTION uses ONLY
local information — no global state can influence action selection. If these
fail, the centralized-vs-decentralized experiment is invalid.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from environment.make_env import make_env
from environment.transitions import NUM_ACTIONS
from rl.local_graph import build_local_graph, node_feature_dim_local
from rl.graph_batch import node_feature_dim, collate_graphs
from rl.action_masking import node_action_mask
from controllers.decentralized_gnn import DecentralizedActorCritic


# --------------------------------------------------- local graph structure
def test_local_graph_matches_global_topology():
    """The local graph's EDGES must match the global adjacency (physical
    connections are locally known), even though node FEATURES are local-only."""
    env = make_env(n_modules=6, seed=1)
    env.reset()
    all_local = env.all_local_observations()
    g = build_local_graph(all_local, env.mission.config.primary_objective)

    total_deg = sum(len(v) for v in env.adjacency().values())
    assert g.edge_index.shape == (2, total_deg)
    assert g.node_features.shape == (6, node_feature_dim_local())


def test_local_features_use_only_own_state():
    """A module's node feature vector must depend ONLY on its own state and the
    mission -- NOT on any other module's internal state. We verify by mutating
    a DISTANT module's battery and confirming a non-neighbor's node features
    are unchanged."""
    env = make_env(n_modules=6, shape="line", seed=0)
    env.reset()
    mission = env.mission.config.primary_objective

    all_local = env.all_local_observations()
    g_before = build_local_graph(all_local, mission)

    # Mutate module 5's battery (module 0 is far away, not a neighbor).
    env._module(5).battery = 0.123456
    all_local_after = env.all_local_observations()
    g_after = build_local_graph(all_local_after, mission)

    # Module 0's node features must be identical (it can't see module 5).
    row0 = g_before.id_order.index(0)
    assert np.allclose(g_before.node_features[row0],
                       g_after.node_features[row0])


# --------------------------------------------------- execution leak prevention
def _tiny_checkpoint(tmp):
    """Train a tiny CTDE model and return its checkpoint path."""
    from rl.ctde_trainer import CTDETrainer, CTDETrainConfig
    from rl.ppo import PPOConfig
    cfg = CTDETrainConfig(
        n_modules_choices=[4, 6], mission_choices=["power"],
        max_steps=12, total_steps=384, rollout_len=192,
        hidden_dim=16, actor_layers=2, critic_layers=3, seed=0,
        checkpoint_dir=tmp, checkpoint_name="dec_test.pt",
    )
    trainer = CTDETrainer(cfg, PPOConfig(n_epochs=1, minibatch_size=64,
                                         target_kl=None))
    return trainer.train()


def test_decentralized_controller_ignores_global_observation():
    """The decentralized controller must produce the SAME action whether or not
    the global observation it is handed is corrupted -- because it must not use
    it. We pass a deliberately garbage global observation and confirm the action
    is identical to passing None."""
    from controllers.decentralized_gnn_controller import DecentralizedGNNController

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _tiny_checkpoint(tmp)
        env = make_env(n_modules=4, mission="power", seed=3)
        env.reset()
        ctrl = DecentralizedGNNController(env, checkpoint=ckpt)

        action_with_none = ctrl.act(None)

        # Hand it a corrupted "global observation". If the controller respects
        # locality, this must not change the output at all.
        garbage = {"modules": "CORRUPTED", "adjacency": None,
                   "sun_direction": np.array([9, 9, 9]), "mission": "nonsense"}
        action_with_garbage = ctrl.act(garbage)

        assert action_with_none == action_with_garbage


def test_decentralized_controller_only_calls_local_obs(monkeypatch):
    """Assert the controller NEVER calls env.global_observation() during act().
    We monkeypatch global_observation to raise if touched."""
    from controllers.decentralized_gnn_controller import DecentralizedGNNController

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _tiny_checkpoint(tmp)
        env = make_env(n_modules=4, mission="power", seed=4)
        env.reset()
        ctrl = DecentralizedGNNController(env, checkpoint=ckpt)

        def _forbidden(*a, **k):
            raise AssertionError("decentralized execution accessed GLOBAL state!")

        monkeypatch.setattr(env, "global_observation", _forbidden)
        # Should run purely on local observations without touching global.
        action = ctrl.act(None)
        assert isinstance(action, dict)
        assert set(action.keys()) == {m.id for m in env.modules}


def test_decentralized_produces_only_legal_actions():
    from controllers.decentralized_gnn_controller import DecentralizedGNNController
    from evaluation.metrics import run_episode

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _tiny_checkpoint(tmp)
        env = make_env(n_modules=6, mission="power", max_steps=15, seed=5)
        ctrl = DecentralizedGNNController(env, checkpoint=ckpt)
        metrics, _ = run_episode(env, ctrl)
        assert metrics.total_invalid == 0   # action masking guarantees legality
        assert metrics.steps > 0


def test_decentralized_handles_unseen_n():
    """One decentralized checkpoint should run on N it wasn't trained on
    (shared weights + local graph => size-agnostic)."""
    from controllers.decentralized_gnn_controller import DecentralizedGNNController
    from evaluation.metrics import run_episode

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _tiny_checkpoint(tmp)   # trained on N in {4,6}
        for n in (4, 6, 8, 10):
            env = make_env(n_modules=n, mission="power", max_steps=12, seed=n)
            ctrl = DecentralizedGNNController(env, checkpoint=ckpt)
            metrics, _ = run_episode(env, ctrl)
            assert metrics.steps > 0


# --------------------------------------------------- CTDE structure
def test_actor_critic_separate_node_dims():
    """Actor uses local node features; critic uses global node features. Their
    input dims are the same layout here, but they are SEPARATE networks."""
    ac = DecentralizedActorCritic(hidden_dim=16, actor_layers=2,
                                  critic_layers=4)
    # Separate parameter sets.
    actor_ids = {id(p) for p in ac.actor.parameters()}
    critic_ids = {id(p) for p in ac.critic.parameters()}
    assert actor_ids.isdisjoint(critic_ids)
    # Critic has more MP layers than actor (global vs local reasoning).
    assert len(ac.critic.mp_layers) > len(ac.actor.mp_layers)


def test_ctde_training_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        ckpt = _tiny_checkpoint(tmp)
        assert os.path.exists(ckpt)
        loaded = torch.load(ckpt, map_location="cpu")
        # Checkpoint carries BOTH actor and critic; execution uses only actor.
        assert "actor_state_dict" in loaded
        assert "critic_state_dict" in loaded