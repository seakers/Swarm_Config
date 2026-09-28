# baselines/greedy.py
"""
Greedy heuristic controller.

At each timestep it selects a joint action that maximally improves the mission
objective for the *next* state, evaluated by simulating candidate moves against
a COPY of the current geometry. It never mutates the live environment.

Two modes:
  - "joint"      : exhaustively evaluate combinations of single-module moves.
                   Correct for small N but combinatorial. We bound the search
                   by only considering at most `max_movers` modules moving at
                   once (default 1 or 2), which keeps it tractable while still
                   capturing coordinated improvements.
  - "sequential" : greedily assign moves one module at a time, committing the
                   best improving move and re-evaluating for the next module.
                   Scales to larger N; may miss jointly-optimal moves.

Both modes only ever propose moves the environment considers legal, and both
account for conflicts by simulating the joint transition the same way the
environment resolves it.
"""

from __future__ import annotations
from itertools import combinations, product
from typing import Dict, List, Optional, Tuple

from controllers.base import Controller
from environment.transitions import (
    all_legal_single_moves, ACTION_STAY,
)
from environment.conflict_resolution import resolve
from environment.graph import is_connected

Pos = Tuple[int, int, int]


class GreedyController(Controller):
    def __init__(self, env, mode: str = "joint", max_movers: int = 1):
        """
        Args:
            env:        the SpacecraftEnv (used read-only for evaluation).
            mode:       "joint" or "sequential".
            max_movers: in "joint" mode, the max number of modules allowed to
                        move simultaneously in a candidate (bounds the search).
        """
        self.env = env
        self.mode = mode
        self.max_movers = max_movers
        self._direction = 1.0 if env.mission.config.direction == "maximize" else -1.0

    def reset(self):
        pass

    # ------------------------------------------------------------------ act
    def act(self, observation=None) -> Dict[int, int]:
        positions = self.env.positions_by_id()
        orientations = self.env.orientations_by_id()
        require_conn = self.env.mission.config.require_connected

        legal = all_legal_single_moves(positions, orientations,
                                       require_connected=require_conn)

        if self.mode == "joint":
            return self._act_joint(positions, orientations, legal, require_conn)
        elif self.mode == "sequential":
            return self._act_sequential(positions, orientations, legal, require_conn)
        else:
            raise ValueError(f"Unknown greedy mode {self.mode}")

    # ------------------------------------------------------- objective helper
    def _objective_of(self, positions: Dict[int, Pos],
                      orientations: Dict[int, int]) -> float:
        """Directed objective for a hypothetical configuration.

        We temporarily construct the objective from module copies so nothing
        in the live env is touched.
        """
        # Build lightweight module views for the mission evaluator.
        modules = []
        for m in self.env.modules:
            mc = m.copy()
            mc.position = positions[m.id]
            mc.orientation = orientations[m.id]
            modules.append(mc)
        return self._direction * self.env.mission.objective(modules)

    def _current_directed_objective(self) -> float:
        return self._objective_of(self.env.positions_by_id(),
                                  self.env.orientations_by_id())

    def _apply_accepted(self, positions, orientations, accepted):
        """Return new (positions, orientations) after applying accepted moves."""
        new_pos = dict(positions)
        new_orient = dict(orientations)
        for mid, (dest, dorient) in accepted.items():
            new_pos[mid] = dest
            new_orient[mid] = dorient
        return new_pos, new_orient

    # ------------------------------------------------------------- joint mode
    def _act_joint(self, positions, orientations, legal, require_conn):
        base_obj = self._current_directed_objective()
        best_obj = base_obj
        best_action = {mid: ACTION_STAY for mid in positions}

        movers = [mid for mid, acts in legal.items() if acts]  # modules with moves

        # Consider subsets of up to `max_movers` modules moving simultaneously.
        for k in range(1, self.max_movers + 1):
            for subset in combinations(movers, k):
                # Cartesian product of each mover's legal actions.
                action_lists = [list(legal[mid].keys()) for mid in subset]
                for combo in product(*action_lists):
                    proposals = {
                        mid: legal[mid][a]
                        for mid, a in zip(subset, combo)
                    }
                    accepted, _ = resolve(positions, orientations, proposals,
                                          require_connected=require_conn)
                    if len(accepted) != len(proposals):
                        continue  # some proposal was rejected -> skip candidate
                    new_pos, new_orient = self._apply_accepted(
                        positions, orientations, accepted)
                    obj = self._objective_of(new_pos, new_orient)
                    if obj > best_obj + 1e-9:
                        best_obj = obj
                        best_action = {mid: ACTION_STAY for mid in positions}
                        for mid, a in zip(subset, combo):
                            best_action[mid] = a

        return best_action

    # -------------------------------------------------------- sequential mode
    def _act_sequential(self, positions, orientations, legal, require_conn):
        """Assign moves one module at a time, committing improving moves."""
        chosen = {mid: ACTION_STAY for mid in positions}
        chosen_proposals: Dict[int, Tuple[Pos, int]] = {}
        cur_obj = self._current_directed_objective()

        # Process modules in a fixed order (id order) for determinism.
        for mid in sorted(legal.keys()):
            best_local_obj = cur_obj
            best_local_action = ACTION_STAY
            best_local_proposal = None

            for a, (dest, dorient) in legal[mid].items():
                trial = dict(chosen_proposals)
                trial[mid] = (dest, dorient)
                accepted, _ = resolve(positions, orientations, trial,
                                      require_connected=require_conn)
                if mid not in accepted:
                    continue  # this move conflicts with already-chosen ones
                if len(accepted) != len(trial):
                    continue  # committing this move breaks a prior one
                new_pos, new_orient = self._apply_accepted(
                    positions, orientations, accepted)
                obj = self._objective_of(new_pos, new_orient)
                if obj > best_local_obj + 1e-9:
                    best_local_obj = obj
                    best_local_action = a
                    best_local_proposal = (dest, dorient)

            if best_local_action != ACTION_STAY:
                chosen[mid] = best_local_action
                chosen_proposals[mid] = best_local_proposal
                cur_obj = best_local_obj

        return chosen