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
    ap = argparse.ArgumentParser(description="Modular spacecraft benchmark.")
    ap.add_argument("--missions", nargs="+",
                    default=["power", "thermal", "comms", "aperture"])
    ap.add_argument("--n", nargs="+", type=int, default=[4, 6, 8])
    ap.add_argument("--shapes", nargs="+", default=["random"])
    ap.add_argument("--config-seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--controllers", nargs="+", default=None,
                    help="subset of registered controllers (default: all)")
    ap.add_argument("--gnn-checkpoint", type=str,
                    default="checkpoints/gnn_centralized.pt")
    ap.add_argument("--decentralized-checkpoint", type=str,
                    default="checkpoints/gnn_decentralized.pt")
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--sun-direction", nargs=3, type=float,
                    default=[0.0, 0.0, 1.0])
    ap.add_argument("--best-known-expansions", type=int, default=10_000)
    ap.add_argument("--no-best-known", action="store_true")
    ap.add_argument("--results-base", type=str, default="results")
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--eval-device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    choices=["cpu", "cuda"])
    ap.add_argument("--replot", type=str, default=None,
                    help="path to existing results dir to replot only")
    args = ap.parse_args()

    if args.replot:
        with open(os.path.join(args.replot, "records.json")) as f:
            rows = json.load(f)
        # Reconstruct minimal RunRecord objects from rows.
        records = [RunRecord(**{k: row[k] for k in row
                                if k in RunRecord.__dataclass_fields__})
                   for row in rows]
        plot_all_results(records, os.path.join(args.replot, "plots"))
        print(f"Replotted into {os.path.join(args.replot, 'plots')}")
        return

    # --- Build scenarios (shared across all controllers) ---
    scenarios = build_scenarios(
        args.missions, args.n, args.shapes,
        args.config_seeds, args.max_steps,
        args.sun_direction,
    )

    # --- Resolve controller set ---
    registry = build_controller_registry(
        gnn_checkpoint=args.gnn_checkpoint,
        decentralized_checkpoint=args.decentralized_checkpoint,
        device=args.device,
    )
    if args.controllers:
        unknown = [c for c in args.controllers if c not in registry]
        if unknown:
            raise SystemExit(f"Unknown controllers: {unknown}. "
                             f"Available: {sorted(registry)}")
        controller_names = args.controllers
    else:
        controller_names = list(registry.keys())

    # --- Set up results directory + manifest ---
    results_dir = make_results_dir(args.results_base)
    write_manifest(results_dir, scenarios, controller_names, extra={
        "missions": args.missions,
        "n": args.n,
        "shapes": args.shapes,
        "config_seeds": args.config_seeds,
        "max_steps": args.max_steps,
        "sun_direction": args.sun_direction,
        "best_known_expansions": args.best_known_expansions,
        "no_best_known": args.no_best_known,
        "repeats": args.repeats,
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
        if args.no_best_known:
            best_known = float("nan")
        else:
            print("  Computing best-known objective (objective-max A*)...")
            best_known = compute_best_known(
                scenario, max_expansions=args.best_known_expansions)
            print(f"  best_known objective = {best_known:.3f}")

        for name in controller_names:
            factory, is_stochastic = registry[name]
            n_runs = args.repeats if is_stochastic else 1
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