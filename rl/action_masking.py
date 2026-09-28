# rl/action_masking.py
"""
Action masking. Produces a boolean legality mask of shape (N, NUM_ACTIONS)
from the environment's legal_actions() dict, so policies can zero out the
probability of illegal actions before sampling.
"""

from __future__ import annotations
from typing import Dict, List
import numpy as np

from environment.transitions import NUM_ACTIONS, ACTION_STAY


def legal_action_mask(env) -> np.ndarray:
    """Return a (N, NUM_ACTIONS) float mask: 1.0 = legal, 0.0 = illegal.

    Row order matches sorted module ids. STAY is always legal.
    """
    legal = env.legal_actions()   # {module_id: [legal action indices]}
    ids = sorted(legal.keys())
    mask = np.zeros((len(ids), NUM_ACTIONS), dtype=np.float32)
    for row, mid in enumerate(ids):
        for a in legal[mid]:
            mask[row, a] = 1.0
        mask[row, ACTION_STAY] = 1.0  # ensure STAY always available
    return mask


def module_id_order(env) -> List[int]:
    """The module-id ordering used consistently for policy rows."""
    return sorted(m.id for m in env.modules)


def node_action_mask(env, id_order) -> np.ndarray:
    """Per-node legal-action mask [N, NUM_ACTIONS] in the given id order.

    `id_order` must match the node ordering used by the graph builder, so the
    mask rows align with the GNN's node rows.
    """
    legal = env.legal_actions()
    mask = np.zeros((len(id_order), NUM_ACTIONS), dtype=np.float32)
    for row, mid in enumerate(id_order):
        for a in legal[mid]:
            mask[row, a] = 1.0
        mask[row, ACTION_STAY] = 1.0
    return mask