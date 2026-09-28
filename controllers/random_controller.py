# controllers/random_controller.py
"""
Random controller. Chooses uniformly among LEGAL actions per module (including
STAY). Used to stress-test the simulator: it must survive arbitrary sequences
of legal and illegal proposals without crashing or corrupting state.

We also provide a "chaotic" mode that proposes uniformly over ALL actions
(0..6) regardless of legality, to verify the environment correctly rejects
illegal / conflicting proposals.
"""

from __future__ import annotations
from typing import Dict
import numpy as np

from .base import Controller
from environment.actions import NUM_ACTIONS


class RandomController(Controller):
    def __init__(self, env, seed: int = 0, respect_legality: bool = True):
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.respect_legality = respect_legality

    def act(self, observation=None) -> Dict[int, int]:
        action = {}
        if self.respect_legality:
            legal = self.env.legal_actions()
            for mid, acts in legal.items():
                action[mid] = int(self.rng.choice(acts))
        else:
            for m in self.env.modules:
                action[m.id] = int(self.rng.integers(0, NUM_ACTIONS))
        return action