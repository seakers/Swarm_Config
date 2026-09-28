# baselines/astar.py
"""
A* / best-first search baseline over the configuration graph.

State  = geometric Configuration (hashable).
Edge   = a single-module hinge move (one module pivots).
Cost   = movement_cost per move (uniform).

Two modes:

  1. GOAL-DIRECTED (target_configuration provided):
       Classic A*. g = accumulated move cost, h = admissible lower bound on
       remaining moves * movement_cost. Returns the least-cost path to the goal.
       This gives a KNOWN OPTIMUM to validate other controllers against.

  2. OBJECTIVE-MAXIMIZING (no target):
       Branch-and-bound best-first search maximizing the mission's directed
       objective minus movement cost. Because the reachable configuration graph
       is large, expansion is bounded by `max_expansions`. Returns the best
       configuration + the path of moves to reach it.

Expansion uses SINGLE-module moves only (branching factor ~= movers * ~4),
which keeps A* tractable for small N and lets it scale poorly (as expected /
scientifically useful) as N grows.
"""

from __future__ import annotations
import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from environment.configuration import Configuration
from environment.transitions import all_legal_single_moves, ACTION_STAY

Pos = Tuple[int, int, int]


# ---------------------------------------------------------------------------
# Search node bookkeeping
# ---------------------------------------------------------------------------
@dataclass(order=True)
class _PQItem:
    priority: float
    counter: int                      # tie-breaker for stable ordering
    positions: Dict[int, Pos] = field(compare=False)
    orientations: Dict[int, int] = field(compare=False)
    g_moves: int = field(compare=False, default=0)      # number of moves so far
    path: List[Tuple[int, int]] = field(compare=False, default_factory=list)
    # path is a list of (module_id, action_index) applied in order.


@dataclass
class AStarResult:
    success: bool
    reason: str = ""
    path: List[Tuple[int, int]] = field(default_factory=list)  # (module_id, action)
    final_positions: Optional[Dict[int, Pos]] = None
    final_orientations: Optional[Dict[int, int]] = None
    final_objective: float = 0.0
    moves: int = 0
    expansions: int = 0               # nodes expanded (planning cost proxy)


def _state_key(positions: Dict[int, Pos],
               orientations: Dict[int, int]) -> Tuple:
    """Hashable identity-preserving key for a configuration."""
    return tuple(sorted(
        (mid, positions[mid], orientations[mid]) for mid in positions
    ))


def _mismatch_count(positions: Dict[int, Pos],
                    orientations: Dict[int, int],
                    target_positions: Dict[int, Pos],
                    target_orientations: Dict[int, int]) -> int:
    """How many modules are not at their target pose. Admissible-heuristic base:
    each mismatched module needs at least one move."""
    n = 0
    for mid in positions:
        if (positions[mid] != target_positions[mid]
                or orientations[mid] != target_orientations[mid]):
            n += 1
    return n


class AStarPlanner:
    def __init__(self, env,
                 movement_cost: float = 1.0,
                 max_expansions: int = 50_000):
        """
        Args:
            env:            SpacecraftEnv (read-only, for legality + objective).
            movement_cost:  per-move cost used as edge weight.
            max_expansions: hard cap on node expansions (bounds runtime).
        """
        self.env = env
        self.movement_cost = movement_cost
        self.max_expansions = max_expansions
        self.require_conn = env.mission.config.require_connected
        self._direction = 1.0 if env.mission.config.direction == "maximize" else -1.0

    # ------------------------------------------------------------- objective
    def _objective_of(self, positions, orientations) -> float:
        modules = []
        for m in self.env.modules:
            mc = m.copy()
            mc.position = positions[m.id]
            mc.orientation = orientations[m.id]
            modules.append(mc)
        return self._direction * self.env.mission.objective(modules)

    # ------------------------------------------------------- successor gen
    def _successors(self, positions, orientations):
        """Yield (module_id, action, new_positions, new_orientations)."""
        legal = all_legal_single_moves(positions, orientations,
                                       require_connected=self.require_conn)
        for mid, moves in legal.items():
            for action, (dest, dorient) in moves.items():
                new_pos = dict(positions)
                new_orient = dict(orientations)
                new_pos[mid] = dest
                new_orient[mid] = dorient
                yield mid, action, new_pos, new_orient

    # ================================================= GOAL-DIRECTED A*
    def plan_to_target(self,
                       target_positions: Dict[int, Pos],
                       target_orientations: Dict[int, int]) -> AStarResult:
        start_pos = self.env.positions_by_id()
        start_orient = self.env.orientations_by_id()

        counter = itertools.count()
        start_h = self._heuristic(start_pos, start_orient,
                                  target_positions, target_orientations)
        start_item = _PQItem(
            priority=start_h, counter=next(counter),
            positions=start_pos, orientations=start_orient,
            g_moves=0, path=[],
        )
        open_heap: List[_PQItem] = [start_item]
        best_g: Dict[Tuple, int] = {_state_key(start_pos, start_orient): 0}
        expansions = 0

        goal_key = _state_key(target_positions, target_orientations)

        while open_heap:
            item = heapq.heappop(open_heap)
            key = _state_key(item.positions, item.orientations)

            # Stale entry (a better path to this state was already found).
            if item.g_moves > best_g.get(key, float("inf")):
                continue

            if key == goal_key:
                return AStarResult(
                    success=True, path=item.path,
                    final_positions=item.positions,
                    final_orientations=item.orientations,
                    final_objective=self._objective_of(item.positions,
                                                        item.orientations),
                    moves=item.g_moves, expansions=expansions,
                )

            if expansions >= self.max_expansions:
                return AStarResult(False, "max_expansions reached",
                                   expansions=expansions)
            expansions += 1

            for mid, action, npos, norient in self._successors(
                    item.positions, item.orientations):
                nkey = _state_key(npos, norient)
                ng = item.g_moves + 1
                if ng < best_g.get(nkey, float("inf")):
                    best_g[nkey] = ng
                    h = self._heuristic(npos, norient,
                                        target_positions, target_orientations)
                    heapq.heappush(open_heap, _PQItem(
                        priority=(ng * self.movement_cost
                                  + h * self.movement_cost),
                        counter=next(counter),
                        positions=npos, orientations=norient,
                        g_moves=ng, path=item.path + [(mid, action)],
                    ))

        return AStarResult(False, "goal unreachable within search",
                           expansions=expansions)

    def _heuristic(self, positions, orientations,
                   target_positions, target_orientations) -> float:
        """Admissible: at least `mismatched` moves remain (each move fixes at
        most one module's pose). This never overestimates."""
        return float(_mismatch_count(positions, orientations,
                                     target_positions, target_orientations))

    # ============================================ OBJECTIVE-MAXIMIZING search
    def plan_maximize_objective(self) -> AStarResult:
        """Branch-and-bound best-first search for the configuration maximizing
        (directed_objective - movement_cost * moves), bounded by expansions."""
        start_pos = self.env.positions_by_id()
        start_orient = self.env.orientations_by_id()

        counter = itertools.count()
        start_obj = self._objective_of(start_pos, start_orient)

        # Priority = negative net value (heapq is a min-heap; we want max value).
        def net_value(obj, moves):
            return obj - self.movement_cost * moves

        start_item = _PQItem(
            priority=-net_value(start_obj, 0), counter=next(counter),
            positions=start_pos, orientations=start_orient,
            g_moves=0, path=[],
        )
        open_heap: List[_PQItem] = [start_item]
        visited: Dict[Tuple, float] = {
            _state_key(start_pos, start_orient): net_value(start_obj, 0)
        }

        best = AStarResult(
            success=True, path=[],
            final_positions=start_pos, final_orientations=start_orient,
            final_objective=start_obj, moves=0, expansions=0,
        )
        best_value = net_value(start_obj, 0)
        expansions = 0

        while open_heap and expansions < self.max_expansions:
            item = heapq.heappop(open_heap)
            expansions += 1

            cur_obj = self._objective_of(item.positions, item.orientations)
            cur_value = net_value(cur_obj, item.g_moves)
            if cur_value > best_value + 1e-9:
                best_value = cur_value
                best = AStarResult(
                    success=True, path=item.path,
                    final_positions=item.positions,
                    final_orientations=item.orientations,
                    final_objective=cur_obj, moves=item.g_moves,
                    expansions=expansions,
                )

            for mid, action, npos, norient in self._successors(
                    item.positions, item.orientations):
                nkey = _state_key(npos, norient)
                nobj = self._objective_of(npos, norient)
                nval = net_value(nobj, item.g_moves + 1)
                # Revisit only if we found a strictly better net value here.
                if nval > visited.get(nkey, float("-inf")) + 1e-9:
                    visited[nkey] = nval
                    heapq.heappush(open_heap, _PQItem(
                        priority=-nval, counter=next(counter),
                        positions=npos, orientations=norient,
                        g_moves=item.g_moves + 1,
                        path=item.path + [(mid, action)],
                    ))

        best.expansions = expansions
        return best