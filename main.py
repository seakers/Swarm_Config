# main.py
"""
Main benchmarking entry point.

Runs every registered controller against a shared set of scenarios
(mission x starting configuration) and saves all results under
results/<datestring>/.

Usage:
    python main.py
    python main.py --missions power thermal --n 4 6 --config-seeds 0 1 2
    python main.py --controllers random greedy_joint astar --n 4
    python main.py --no-best-known        # skip the A* reference optimum
"""

from __future__ import annotations
import argparse
import itertools
import torch
import os
import json

from evaluation.benchmarking import (
    Scenario, RunRecord, build_controller_registry,
    compute_best_known, run_single,
)
from evaluation.results_manager import (
    make_results_dir, write_manifest, write_records, write_summary,
)
from evaluation.visualization import plot_all_results


def build_scenarios(missions, n_list, shapes, config_seeds,
                    max_steps, sun_direction) -> list:
    """Cartesian product of the requested scenario axes."""
    scenarios = []
    for mission, n, shape, cseed in itertools.product(
            missions, n_list, shapes, config_seeds):
        scenarios.append(Scenario(
            mission=mission,
            n_modules=n,
            shape=shape,
            config_seed=cseed,
            max_steps=max_steps,
            sun_direction=tuple(sun_direction),
        ))
    return scenarios


def main():
    params = {
        "missions": ["power", "thermal", "comms", "aperture"],
        "n": [64],
        "shapes": ["random"],
        "config_seeds": [0, 1, 2], # [0, 1, 2],
        "controllers": ['random', 'greedy_joint', 'greedy_seq', 'astar', 'centralized_gnn', 'decentralized_gnn'], # astar
        "gnn_checkpoint": "checkpoints/gnn_centralized.pt",
        "decentralized_checkpoint": "checkpoints/gnn_decentralized.pt",
        "max_steps": 200,
        "sun_direction": [0.0, 0.0, 1.0],
        "best_known_expansions": 10_000,
        "no_best_known": False,
        "results_base": "results",
        "repeats": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "replot": None, # "results/20260925_013037", #path to existing results dir to replot only (no new runs)
    }

    if params["replot"]:
        with open(os.path.join(params["replot"], "records.json")) as f:
            rows = json.load(f)
        # Reconstruct minimal RunRecord objects from rows.
        records = [RunRecord(**{k: row[k] for k in row
                                if k in RunRecord.__dataclass_fields__})
                   for row in rows]
        plot_all_results(records, os.path.join(params["replot"], "plots"))
        print(f"Replotted into {os.path.join(params["replot"], 'plots')}")
        return

    # --- Build scenarios (shared across all controllers) ---
    scenarios = build_scenarios(
        params["missions"], params["n"], params["shapes"],
        params["config_seeds"], params["max_steps"],
        params["sun_direction"],
    )

    # --- Resolve controller set ---
    registry = build_controller_registry(
        gnn_checkpoint=params["gnn_checkpoint"],
        decentralized_checkpoint=params["decentralized_checkpoint"],
        device=params["device"],
    )
    if params["controllers"]:
        unknown = [c for c in params["controllers"] if c not in registry]
        if unknown:
            raise SystemExit(f"Unknown controllers: {unknown}. "
                             f"Available: {sorted(registry)}")
        controller_names = params["controllers"]
    else:
        controller_names = list(registry.keys())

    # --- Set up results directory + manifest ---
    results_dir = make_results_dir(params["results_base"])
    write_manifest(results_dir, scenarios, controller_names, extra={
        "missions": params["missions"],
        "n": params["n"],
        "shapes": params["shapes"],
        "config_seeds": params["config_seeds"],
        "max_steps": params["max_steps"],
        "sun_direction": params["sun_direction"],
        "best_known_expansions": params["best_known_expansions"],
        "no_best_known": params["no_best_known"],
        "repeats": params["repeats"],
    })
    print(f"Results directory: {results_dir}")
    print(f"Scenarios: {len(scenarios)} | Controllers: {controller_names}")

    # --- Run ---
    records = []
    best_traces = {}
    for si, scenario in enumerate(scenarios):
        print("\n" + "=" * 70)
        print(f"[{si + 1}/{len(scenarios)}] Scenario: {scenario.scenario_id}")
        print("=" * 70)

        # Reference optimum (shared per scenario) for distance-from-best metric.
        if params["no_best_known"]:
            best_known = float("nan")
        else:
            print("  Computing best-known objective (objective-max A*)...")
            best_known = compute_best_known(
                scenario, max_expansions=params["best_known_expansions"])
            print(f"  best_known objective = {best_known:.3f}")

        for name in controller_names:
            factory, is_stochastic = registry[name]
            n_runs = params["repeats"] if is_stochastic else 1
            run_finals = []
            for run_idx in range(n_runs):
                eval_seed = scenario.config_seed * 1000 + run_idx
                try:
                    record = run_single(
                        scenario, name, factory, best_known,
                        run_index=run_idx, eval_seed=eval_seed)
                except (FileNotFoundError, ValueError) as e:
                    print(f"  {name:<18} SKIPPED ({type(e).__name__}: {e})")
                    break
                records.append(record)
                key = (scenario.scenario_id, name)
                cur = best_traces.get(key)
                cand = (record.final_objective, -record.total_moves)
                if cur is None or cand > (cur[0], -cur[1]):
                    best_traces[key] = (record.final_objective,
                                        record.total_moves,
                                        record._metrics)
                run_finals.append(record.final_objective)
            if run_finals:
                import numpy as _np
                arr = _np.array(run_finals)
                extra = ""
                if records and records[-1].planning_expansions is not None:
                    extra = (f" | expansions={records[-1].planning_expansions}"
                             f" plan_len={records[-1].plan_length}")
                print(f"  {name:<18} final={arr.mean():>8.3f}±{arr.std():<6.3f} "
                      f"runs={len(run_finals)}{extra}")

    # --- Persist everything ---
    write_records(results_dir, records)
    summary = write_summary(results_dir, records)

    traces_dir = os.path.join(results_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    for (sid, ctrl), (fobj, moves, metrics) in best_traces.items():
        trace = {
            "scenario_id": sid,
            "controller": ctrl,
            "final_objective": fobj,
            "total_moves": moves,
            "initial_config": metrics.initial_config,
            "action_trace": metrics.action_trace,
        }
        fname = f"{sid}__{ctrl}.json"
        with open(os.path.join(traces_dir, fname), "w") as f:
            json.dump(trace, f, indent=2)
    print(f"  - traces/        (best move sequence per scenario+method)")

    plots_dir = os.path.join(results_dir, "plots")
    plot_all_results(records, plots_dir)
    print(f"  - plots/         (scaling + reward-vs-moves per mission)")

    print("\n" + summary)
    print(f"\nAll results saved to: {results_dir}")
    print("  - manifest.json  (run configuration + seeds)")
    print("  - records.json   (full per-run metrics)")
    print("  - records.csv    (flat, for plotting)")
    print("  - summary.txt     (human-readable comparison)")


if __name__ == "__main__":
    main()