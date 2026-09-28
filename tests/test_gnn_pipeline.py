# tests/test_gnn_pipeline.py
"""
Tests for the GNN pipeline: graph construction, batching, permutation
invariance, variable-N handling, message-passing locality, a short training
smoke test, and controller round-trip.
"""

import os
import tempfile
import numpy as np
import torch
import pytest

from environment.make_env import make_env
from environment.transitions import NUM_ACTIONS
from rl.graph_batch import (
    build_graph_from_global_obs, collate_graphs, node_feature_dim,
    EDGE_FEATURE_DIM,
)
from rl.action_masking import node_action_mask
from controllers.gnn_policy import GNNPolicy


# ------------------------------------------------------------- graph build
def test_graph_matches_adjacency():
    env = make_env(n_modules=5, seed=0)
    obs, _ = env.reset()
    g = build_graph_from_global_obs(obs)
    assert g.node_features.shape == (5, node_feature_dim())
    # Edge count = sum of adjacency degrees.
    total_deg = sum(len(v) for v in env.adjacency().values())
    assert g.edge_index.shape == (2, total_deg)
    assert g.edge_features.shape == (total_deg, EDGE_FEATURE_DIM)


def test_collate_disjoint_batch():
    env4 = make_env(n_modules=4, seed=0); o4, _ = env4.reset()
    env6 = make_env(n_modules=6, seed=1); o6, _ = env6.reset()
    g4 = build_graph_from_global_obs(o4)
    g6 = build_graph_from_global_obs(o6)
    nf, ei, ef, batch, sizes = collate_graphs([g4, g6])
    assert nf.shape[0] == 10                 # 4 + 6 nodes
    assert list(sizes) == [4, 6]
    assert batch.tolist() == [0]*4 + [1]*6
    # Edge indices for graph 1 must be offset by 4.
    assert ei[:, g4.edge_index.shape[1]:].min() >= 4


# ------------------------------------------------------------- policy
def test_gnn_handles_variable_n():
    policy = GNNPolicy(node_feature_dim(), EDGE_FEATURE_DIM,
                       hidden=32, n_layers=2)
    for n in (4, 6, 8):
        env = make_env(n_modules=n, seed=0); obs, _ = env.reset()
        g = build_graph_from_global_obs(obs)
        nf, ei, ef, batch, _ = collate_graphs([g])
        mask = node_action_mask(env, g.id_order)
        t = lambda x, dt: torch.as_tensor(x, dtype=dt)
        actions, lp, val = policy.act(
            t(nf, torch.float32), t(ei, torch.int64), t(ef, torch.float32),
            t(batch, torch.int64), num_graphs=1, mask=t(mask, torch.float32))
        assert actions.shape == (n,)
        assert lp.shape == (1,)
        assert val.shape == (1,)


def test_gnn_permutation_invariance():
    """Relabeling module ids must not change the set of (position -> action)
    decisions under a deterministic policy. The GNN is permutation-equivariant:
    permuting node order permutes outputs identically."""
    policy = GNNPolicy(node_feature_dim(), EDGE_FEATURE_DIM,
                       hidden=32, n_layers=2)
    policy.eval()

    env = make_env(n_modules=5, seed=0)
    obs, _ = env.reset()
    g = build_graph_from_global_obs(obs)

    t = lambda x, dt: torch.as_tensor(x, dtype=dt)

    def deterministic_actions(graph):
        nf, ei, ef, batch, _ = collate_graphs([graph])
        with torch.no_grad():
            actions, _, _ = policy.act(
                t(nf, torch.float32), t(ei, torch.int64),
                t(ef, torch.float32), t(batch, torch.int64),
                num_graphs=1, mask=None, deterministic=True)
        return actions.numpy()

    # Original decisions, keyed by module id (id_order maps row->id).
    a1 = deterministic_actions(g)
    decisions_1 = {g.id_order[i]: int(a1[i]) for i in range(len(g.id_order))}

    # Build a permuted graph: shuffle node rows and remap edge indices.
    perm = np.array([3, 0, 4, 1, 2])  # a fixed permutation of 5 rows
    inv = np.argsort(perm)
    g_perm_nf = g.node_features[perm]
    # Edge indices must be remapped through the inverse permutation.
    remapped = inv[g.edge_index]
    from rl.graph_batch import GraphTensors
    g_perm = GraphTensors(
        node_features=g_perm_nf,
        edge_index=remapped,
        edge_features=g.edge_features,
        id_order=[g.id_order[p] for p in perm],
    )

    a2 = deterministic_actions(g_perm)
    decisions_2 = {g_perm.id_order[i]: int(a2[i]) for i in range(len(perm))}

    # Same module must get the same action regardless of ordering.
    assert decisions_1 == decisions_2


def test_message_passing_locality():
    """With L message-passing layers, a node's embedding depends only on its
    L-hop neighborhood. Changing a far-away node (> L hops) must not change a
    node's output. We verify on a line graph."""
    # Line of 6 nodes: 0-1-2-3-4-5. With 1 MP layer, node 0 sees only node 1.
    policy = GNNPolicy(node_feature_dim(), EDGE_FEATURE_DIM,
                       hidden=32, n_layers=1)
    policy.eval()
    env = make_env(n_modules=6, shape="line", seed=0)
    obs, _ = env.reset()
    g = build_graph_from_global_obs(obs)

    t = lambda x, dt: torch.as_tensor(x, dtype=dt)

    def node0_logits(graph):
        with torch.no_grad():
            h = policy.embed(t(graph.node_features, torch.float32),
                             t(graph.edge_index, torch.int64),
                             t(graph.edge_features, torch.float32))
            return policy.node_logits(h)[0].numpy()  # node 0's logits

    base = node0_logits(g)

    # Perturb a FAR node (node 5, which is 5 hops from node 0). With 1 MP
    # layer, node 0's output must be unchanged.
    from rl.graph_batch import GraphTensors
    nf2 = g.node_features.copy()
    nf2[5] += 10.0  # large perturbation to the far node's features
    g2 = GraphTensors(nf2, g.edge_index, g.edge_features, g.id_order)
    far = node0_logits(g2)
    assert np.allclose(base, far, atol=1e-5)

    # Perturb the DIRECT neighbor (node 1). Node 0's output SHOULD change.
    nf3 = g.node_features.copy()
    nf3[1] += 10.0
    g3 = GraphTensors(nf3, g.edge_index, g.edge_features, g.id_order)
    near = node0_logits(g3)
    assert not np.allclose(base, near, atol=1e-5)


def test_masked_actions_legal():
    policy = GNNPolicy(node_feature_dim(), EDGE_FEATURE_DIM,
                       hidden=16, n_layers=1)
    env = make_env(n_modules=5, seed=2)
    obs, _ = env.reset()
    g = build_graph_from_global_obs(obs)
    nf, ei, ef, batch, _ = collate_graphs([g])
    mask = node_action_mask(env, g.id_order)
    t = lambda x, dt: torch.as_tensor(x, dtype=dt)
    for _ in range(20):
        actions, _, _ = policy.act(
            t(nf, torch.float32), t(ei, torch.int64), t(ef, torch.float32),
            t(batch, torch.int64), num_graphs=1,
            mask=t(mask, torch.float32), deterministic=False)
        a = actions.numpy()
        for row in range(len(g.id_order)):
            assert mask[row, a[row]] > 0.5


# --------------------------------------------------- training smoke + ckpt
def test_gnn_training_smoke_and_roundtrip():
    from rl.gnn_trainer import GNNPPOTrainer, GNNTrainConfig
    from rl.ppo import PPOConfig
    from controllers.centralized_gnn_controller import CentralizedGNNController
    from evaluation.metrics import run_episode

    with tempfile.TemporaryDirectory() as tmp:
        cfg = GNNTrainConfig(
            n_modules_choices=[4, 6],       # train across two sizes
            mission_choices=["power"],
            max_steps=15, total_steps=512, rollout_len=256,
            hidden_dim=16, n_layers=2, seed=0,
            checkpoint_dir=tmp, checkpoint_name="gnn_test.pt",
        )
        ppo_cfg = PPOConfig(n_epochs=1, minibatch_size=64, target_kl=None)
        trainer = GNNPPOTrainer(cfg, ppo_cfg)
        ckpt = trainer.train()
        assert os.path.exists(ckpt)

        # The SAME checkpoint must run on N it wasn't fixed to (4, 6, and 8).
        for n in (4, 6, 8):
            env = make_env(n_modules=n, mission="power", max_steps=15, seed=0)
            ctrl = CentralizedGNNController(env, checkpoint=ckpt)
            metrics, _ = run_episode(env, ctrl)
            assert metrics.steps > 0
            assert metrics.total_invalid == 0   # masking guarantees legality