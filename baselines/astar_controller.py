# baselines/astar_controller.py
"""
Adapter exposing A* as a step-by-step Controller.

A* is a PLANNER: it computes a full sequence of single-module moves once, then
this controller replays them one per environment step (all other modules STAY).

Because A* uses single-module moves, each environment step issues a joint action
in which exactly one module acts. This is fully consistent with the environment's
simultaneous-action machinery (a joint action with one mover).
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple

from controllers.base import Controller
from environment.transitions import ACTION_STAY
from baselines.astar import AStarPlanner, AStarResult


class AStarController(Controller):
    def __init__(self, env,
                 target_positions: Optional[Dict] = None,
                 target_orientations: Optional[Dict] = None,
                 movement_cost: float = 1.0,
                 max_expansions: int = 50_000):
        self.env = env
        self.target_positions = target_positions
        self.target_orientations = target_orientations
        self.planner = AStarPlanner(env, movement_cost=movement_cost,
                                    max_expansions=max_expansions)
        self._plan: List[Tuple[int, int]] = []
        self._idx = 0
        self.result: Optional[AStarResult] = None

    def reset(self):
        """Plan once, from the environment's current (freshly reset) state."""
        if self.target_positions is not None:
            self.result = self.planner.plan_to_target(
                self.target_positions, self.target_orientations)
        else:
            self.result = self.planner.plan_maximize_objective()
        self._plan = list(self.result.path)
        self._idx = 0

    def act(self, observation=None) -> Dict[int, int]:
        joint = {m.id: ACTION_STAY for m in self.env.modules}
        if self.result is None:
            self.reset()
        if self._idx < len(self._plan):
            mid, action = self._plan[self._idx]
            joint[mid] = action
            self._idx += 1
        # Once the plan is exhausted, everyone STAYs (episode will truncate).
        return joint