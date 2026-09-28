# controllers/decentralized_wrapper.py
"""
Wraps a per-module policy so decentralized controllers can be driven with the
same act(observation) interface. Each module gets ONLY its local observation.

This enforces at the interface level that decentralized execution cannot see
global state: the driver passes all_local_observations(), and the wrapper hands
each module policy only its own local dict.
"""

from __future__ import annotations
from typing import Callable, Dict

from .base import Controller


class DecentralizedController(Controller):
    def __init__(self, env, per_module_policy: Callable[[Dict], int]):
        """
        per_module_policy: function mapping a single module's local observation
                           to an action index. Shared parameters => same
                           callable used for every module.
        """
        self.env = env
        self.policy = per_module_policy

    def act(self, observation=None) -> Dict[int, int]:
        local_obs = self.env.all_local_observations()
        return {mid: int(self.policy(obs)) for mid, obs in local_obs.items()}