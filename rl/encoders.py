# rl/encoders.py
"""
Observation encoders: turn the environment's structured observations into
fixed-size tensors for neural policies.

The centralized MLP encoder flattens the full global state into one vector.
It is tied to a fixed number of modules N (that's the point of the MLP
baseline; the GNN will relax this later).
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from environment.geometry import NUM_ORIENTATIONS, FaceCapability

# Number of distinct face-capability categories (for one-hot encoding).
NUM_CAPABILITIES = len(FaceCapability.NAMES)


def _one_hot(index: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= index < size:
        v[index] = 1.0
    return v


def module_feature_vector(module_state: dict) -> np.ndarray:
    """Per-module feature vector (fixed length), used by both MLP and GNN.

    Features:
      position (3)
      orientation one-hot (24)
      face capabilities one-hot per face (6 * NUM_CAPABILITIES)
      battery (1), temperature (1), power_generation (1)
    """
    pos = np.asarray(module_state["position"], dtype=np.float32)
    orient = _one_hot(int(module_state["orientation"]), NUM_ORIENTATIONS)
    caps = module_state["face_capabilities"]
    cap_feats = np.concatenate([_one_hot(int(c), NUM_CAPABILITIES) for c in caps])
    scalars = np.array([
        float(module_state["battery"]),
        float(module_state["temperature"]) / 100.0,   # rough normalization
        float(module_state["power_generation"]),
    ], dtype=np.float32)
    return np.concatenate([pos, orient, cap_feats, scalars]).astype(np.float32)


def module_feature_dim() -> int:
    return 3 + NUM_ORIENTATIONS + 6 * NUM_CAPABILITIES + 3


def global_feature_vector(global_obs: dict, n_modules: int) -> np.ndarray:
    """Flatten the full global observation into one fixed-size vector.

    Layout: [module_0 feats | module_1 feats | ... | sun_direction (3)].
    Modules are ordered by id (global_obs['modules'] is id-ordered).
    """
    modules = global_obs["modules"]
    assert len(modules) == n_modules, (
        f"MLP encoder expects fixed N={n_modules}, got {len(modules)}")
    per_module = [module_feature_vector(m) for m in modules]
    sun = np.asarray(global_obs["sun_direction"], dtype=np.float32)
    return np.concatenate(per_module + [sun]).astype(np.float32)


def global_feature_dim(n_modules: int) -> int:
    return n_modules * module_feature_dim() + 3