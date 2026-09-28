# controllers/base.py
"""
Common controller interface. The environment never needs to know which
concrete controller it is talking to.
"""

from __future__ import annotations
from typing import Dict


class Controller:
    """Base controller interface.

    A controller consumes an observation and returns a JOINT action:
        {module_id: action_index}

    For centralized controllers the observation is the global observation.
    For decentralized controllers the driver passes per-module local
    observations (see DecentralizedController).
    """

    def reset(self):
        pass

    def act(self, observation) -> Dict[int, int]:
        raise NotImplementedError