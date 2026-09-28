# evaluation/benchmarking.py
"""
Benchmarking harness.

Runs a set of controllers against a shared set of scenarios (mission x
starting configuration) and records comparable metrics for each. Designed to be
extended: adding a new controller = one entry in build_controller_registry().

Fairness guarantee: every controller runs on the SAME reset state for a given
(mission, config_seed) scenario, because make_env is deterministic in its seed
and every controller calls env.reset() before its episode.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional
import numpy as np

from environment.make_env import make_env
from evaluation.metrics import run_episode, EpisodeMetrics


# ---------------------------------------------------------------------------
# Scenario definition
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One benchmark task: a mission on a specific starting configuration."""
    mission: str
    n_modules: int
    shape: str
    config_seed: int          # seed controlling the (randomized) start config
    max_steps: int
    sun_direction: tuple = (0.0, 0.0, 1.0)

    @property
    def scenario_id(self) -> str:
        return (f"{self.mission}_n{self.n_modules}_{self.shape}"
                f"_seed{self.config_seed}")

    def build_env(self):
        """Construct a fresh, deterministic environment for this scenario."""
        return make_env(
            n_modules=self.n_modules,
            shape=self.shape,
            mission=self.mission,
            sun_direction=self.sun_direction,
            max_steps=self.max_steps,
            seed=self.config_seed,
        )


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class RunRecord:
    scenario_id: str
    mission: str
    n_modules: int
    config_seed: int
    controller: str
    # Mission performance
    initial_objective: float
    final_objective: float
    improvement: float
    pct_improvement: float
    distance_from_best: float          # best_known_objective - final_objective
    # Reconfiguration efficiency
    steps: int
    total_moves: int
    total_conflicts: int
    total_invalid: int
    total_reward: float
    ever_disconnected: bool
    # Computational performance
    wall_time_s: float
    planning_expansions: Optional[int] = None   # A* only
    plan_length: Optional[int] = None           # A* only
    # Metadata
    run_index: int = 0
    eval_seed: int = 0

    def to_row(self) -> dict:
        return {
            "scenario_id": self.scenario_id, "mission": self.mission,
            "n_modules": self.n_modules, "config_seed": self.config_seed,
            "controller": self.controller, "run_index": self.run_index,
            "eval_seed": self.eval_seed,
            "initial_objective": self.initial_objective,
            "final_objective": self.final_objective,
            "improvement": self.improvement,
            "pct_improvement": self.pct_improvement,
            "distance_from_best": self.distance_from_best,
            "steps": self.steps, "total_moves": self.total_moves,
            "total_conflicts": self.total_conflicts,
            "total_invalid": self.total_invalid,
            "total_reward": self.total_reward,
            "ever_disconnected": self.ever_disconnected,
            "wall_time_s": self.wall_time_s,
            "planning_expansions": self.planning_expansions,
            "plan_length": self.plan_length,
        }


# ---------------------------------------------------------------------------
# Controller registry — the single place to register new methods (incl. PPO).
# ---------------------------------------------------------------------------
def build_controller_registry(gnn_checkpoint: str = None,
                              decentralized_checkpoint: str = None,
                              device: str = "cpu",) -> Dict[str, Callable]:
    """Return {name: factory(env, scenario) -> Controller}.

    Args:
        gnn_checkpoint:            centralized-GNN checkpoint (any N, any mission).
        decentralized_checkpoint:  decentralized-GNN (CTDE) actor checkpoint.
    """
    from controllers.random_controller import RandomController
    from baselines.greedy import GreedyController
    from baselines.astar_controller import AStarController

    # Each entry: name -> (factory, is_stochastic)
    registry: Dict[str, tuple] = {
        "random": (lambda env, sc: RandomController(
            env, seed=sc.config_seed, respect_legality=True), True),
        "greedy_joint": (lambda env, sc: GreedyController(
            env, mode="joint", max_movers=1), False),
        "greedy_seq": (lambda env, sc: GreedyController(
            env, mode="sequential"), False),
        "astar": (lambda env, sc: AStarController(
            env, target_positions=None, target_orientations=None,
            movement_cost=0.1, max_expansions=20_000), False),
    }

    if gnn_checkpoint:
        from controllers.centralized_gnn_controller import CentralizedGNNController
        registry["centralized_gnn"] = (
            lambda env, sc: CentralizedGNNController(
                env, checkpoint=gnn_checkpoint, device=device, deterministic=False), True)

    if decentralized_checkpoint:
        from controllers.decentralized_gnn_controller import DecentralizedGNNController
        registry["decentralized_gnn"] = (
            lambda env, sc: DecentralizedGNNController(
                env, checkpoint=decentralized_checkpoint, device=device, deterministic=False), True)

    return registry


# ---------------------------------------------------------------------------
# Reference optimum (best-known objective) via objective-maximizing A*
# ---------------------------------------------------------------------------
def compute_best_known(scenario: Scenario,
                       max_expansions: int = 40_000) -> float:
    """Best-known directed objective for a scenario, used for the
    distance-from-optimum metric. Uses objective-maximizing A* with a generous
    expansion budget. For large N this is a lower bound on the true optimum."""
    from baselines.astar import AStarPlanner
    env = scenario.build_env()
    env.reset()
    planner = AStarPlanner(env, movement_cost=0.0, max_expansions=max_expansions)
    result = planner.plan_maximize_objective()
    # movement_cost=0 => pure objective maximization for the reference.
    return result.final_objective


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(scenario: Scenario, controller_name: str,
               controller_factory: Callable, best_known: float,
               run_index: int = 0, eval_seed: int = 0,
               collect_trace: bool = False) -> RunRecord:
    import torch
    torch.manual_seed(eval_seed)
    np.random.seed(eval_seed)

    env = scenario.build_env()
    # For the random controller, thread the per-run seed in.
    controller = controller_factory(env, scenario)
    if hasattr(controller, "rng"):
        controller.rng = np.random.default_rng(eval_seed)

    t0 = time.perf_counter()
    metrics, frames = run_episode(env, controller,
                                  collect_frames=collect_trace)
    wall = time.perf_counter() - t0

    planning_expansions = None
    plan_length = None
    if hasattr(controller, "result") and controller.result is not None:
        planning_expansions = controller.result.expansions
        plan_length = controller.result.moves

    rec = RunRecord(
        scenario_id=scenario.scenario_id,
        mission=scenario.mission,
        n_modules=scenario.n_modules,
        config_seed=scenario.config_seed,
        controller=controller_name,
        run_index=run_index,
        eval_seed=eval_seed,
        initial_objective=metrics.initial_objective,
        final_objective=metrics.final_objective,
        improvement=metrics.improvement,
        pct_improvement=metrics.pct_improvement,
        distance_from_best=best_known - metrics.final_objective,
        steps=metrics.steps,
        total_moves=metrics.total_moves,
        total_conflicts=metrics.total_conflicts,
        total_invalid=metrics.total_invalid,
        total_reward=metrics.total_reward,
        ever_disconnected=metrics.ever_disconnected,
        wall_time_s=wall,
        planning_expansions=planning_expansions,
        plan_length=plan_length,
    )
    # Attach frames for trace-saving (not serialized into the record row).
    rec._frames = frames if collect_trace else None
    rec._metrics = metrics 
    return rec