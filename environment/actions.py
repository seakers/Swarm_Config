# environment/actions.py  (REPLACE)
"""
Action space: STAY + 24 hinge pivots (12 edges x 2 directions).
The canonical definition lives in transitions.py (HINGE_TABLE); we re-export
here so controllers/tests have a stable import path.
"""
from .transitions import (
    ACTION_STAY, NUM_ACTIONS, NUM_HINGES, HINGE_TABLE, action_to_hinge,
)
from .geometry import Edge


def action_name(action: int) -> str:
    if action == ACTION_STAY:
        return "STAY"
    edge, direction = HINGE_TABLE[action - 1]
    return f"HINGE({edge.name},dir={direction:+d})"