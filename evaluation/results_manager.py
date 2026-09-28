# evaluation/results_manager.py
"""
Results persistence. Creates a dated results directory and writes:
  - manifest.json  : run configuration, seeds, controller list, timestamp
  - records.json   : full list of RunRecord dicts
  - records.csv    : same data, flat, for quick plotting / spreadsheets
  - summary.txt    : human-readable per-scenario comparison table

Everything is keyed by a datestring id so runs never overwrite each other (§34).
"""

from __future__ import annotations
import csv
import json
import os
import platform
import sys
from datetime import datetime
from dataclasses import asdict
from typing import Dict, List
import numpy as np

from evaluation.benchmarking import RunRecord, Scenario


def make_results_dir(base: str = "results") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, stamp)
    os.makedirs(path, exist_ok=True)
    return path


def write_manifest(results_dir: str, scenarios: List[Scenario],
                   controller_names: List[str], extra: Dict = None):
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "controllers": controller_names,
        "scenarios": [asdict(s) for s in scenarios],
        "extra": extra or {},
    }
    with open(os.path.join(results_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def write_records(results_dir: str, records: List[RunRecord]):
    rows = [r.to_row() for r in records]
    # JSON
    with open(os.path.join(results_dir, "records.json"), "w") as f:
        json.dump(rows, f, indent=2)
    # CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with open(os.path.join(results_dir, "records.csv"), "w",
                  newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def write_summary(results_dir: str, records):
    by_scenario = {}
    for r in records:
        by_scenario.setdefault(r.scenario_id, {}).setdefault(
            r.controller, []).append(r)

    lines = ["=" * 104, "BENCHMARK SUMMARY (mean ± std over runs)", "=" * 104]
    for sid, by_ctrl in sorted(by_scenario.items()):
        lines += ["", f"Scenario: {sid}", "-" * 104]
        lines.append(f"{'controller':<18}{'final_obj':>18}{'improv':>16}"
                     f"{'moves':>14}{'reward':>16}{'time_s':>12}{'runs':>6}")
        # Sort controllers by mean final objective (desc).
        def mean_final(recs):
            return np.mean([x.final_objective for x in recs])
        for ctrl, recs in sorted(by_ctrl.items(),
                                 key=lambda kv: -mean_final(kv[1])):
            fo = np.array([x.final_objective for x in recs])
            im = np.array([x.improvement for x in recs])
            mv = np.array([x.total_moves for x in recs])
            rw = np.array([x.total_reward for x in recs])
            tm = np.array([x.wall_time_s for x in recs])
            def ms(a):
                return f"{a.mean():.3f}±{a.std():.3f}"
            lines.append(
                f"{ctrl:<18}{ms(fo):>18}{ms(im):>16}{ms(mv):>14}"
                f"{ms(rw):>16}{ms(tm):>12}{len(recs):>6}")
    lines += ["", "=" * 104]
    text = "\n".join(lines)
    with open(os.path.join(results_dir, "summary.txt"), "w") as f:
        f.write(text)
    return text