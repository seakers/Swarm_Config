# evaluation/metrics.py
"""
Episode metric collection. Kept controller-agnostic so every controller
(random, greedy, A*, PPO) records identical metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import csv
import os
import numpy as np


@dataclass
class EpisodeMetrics:
    initial_objective: float = 0.0
    final_objective: float = 0.0
    steps: int = 0
    total_moves: int = 0
    total_conflicts: int = 0
    total_invalid: int = 0
    total_reward: float = 0.0
    ever_disconnected: bool = False
    objective_trace: List[float] = field(default_factory=list)
    action_trace: List[dict] = field(default_factory=list)
    initial_config: dict = field(default_factory=dict)

    @property
    def improvement(self) -> float:
        return self.final_objective - self.initial_objective

    @property
    def pct_improvement(self) -> float:
        if abs(self.initial_objective) < 1e-9:
            return float("inf") if self.improvement > 0 else 0.0
        return 100.0 * self.improvement / abs(self.initial_objective)

    def summary(self) -> dict:
        return {
            "initial_objective": round(self.initial_objective, 4),
            "final_objective": round(self.final_objective, 4),
            "improvement": round(self.improvement, 4),
            "pct_improvement": round(self.pct_improvement, 2),
            "steps": self.steps,
            "total_moves": self.total_moves,
            "total_conflicts": self.total_conflicts,
            "total_invalid": self.total_invalid,
            "total_reward": round(self.total_reward, 4),
            "ever_disconnected": self.ever_disconnected,
        }


def run_episode(env, controller, collect_frames=False):
    """Run one full episode; return (metrics, frames)."""
    from evaluation.visualization import snapshot
    obs, info = env.reset()
    controller.reset() if hasattr(controller, "reset") else None

    metrics = EpisodeMetrics()
    metrics.initial_objective = env.objective_value()
    metrics.initial_config = {
        "positions": {int(m.id): list(m.position) for m in env.modules},
        "orientations": {int(m.id): int(m.orientation) for m in env.modules},
        "face_capabilities": {int(m.id): list(m.face_capabilities)
                              for m in env.modules},
        "mission": env.mission.config.primary_objective,
        "n_modules": len(env.modules),
        "sun_direction": list(env.mission.config.sun_direction),
        "earth_direction": list(env.mission.config.earth_direction),
        "max_steps": env.mission.config.max_steps,
    }
    metrics.objective_trace.append(env.objective_value())

    frames = []
    if collect_frames:
        frames.append(snapshot(env))

    terminated = truncated = False
    while not (terminated or truncated):
        action = controller.act(obs)
        metrics.action_trace.append({int(k): int(v) for k, v in action.items()})
        obs, reward, terminated, truncated, info = env.step(action)
        metrics.steps += 1
        metrics.total_moves += info["n_moved"]
        metrics.total_conflicts += info["n_conflicts"]
        metrics.total_invalid += info["n_invalid"]
        metrics.total_reward += reward
        metrics.objective_trace.append(info["objective"])
        if not info["connected"]:
            metrics.ever_disconnected = True
        if collect_frames:
            frames.append(snapshot(env))

    metrics.final_objective = env.objective_value()
    return metrics, frames


def save_training_log(log_path: str, history: dict):
    """Write training history (dict of equal-length lists) to CSV."""
    keys = list(history.keys())
    n = len(history[keys[0]]) if keys else 0
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for i in range(n):
            w.writerow([history[k][i] for k in keys])


def plot_training_curves(png_path: str, history: dict):
    """Plot key training curves. Optional (requires matplotlib)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    metrics = [("ep_return", "Episode return"),
               ("ep_objective", "Episode objective"),
               ("policy_loss", "Policy loss"),
               ("value_loss", "Value loss"),
               ("entropy", "Entropy"),
               ("kl", "Approx KL")]
    avail = [(k, t) for k, t in metrics if k in history and history[k]]
    if not avail:
        return
    ncol = 2
    nrow = (len(avail) + 1) // 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3 * nrow))
    axes = np.array(axes).reshape(-1)
    x = history.get("step", list(range(len(history[avail[0][0]]))))
    for ax, (k, title) in zip(axes, avail):
        ax.plot(x[:len(history[k])], history[k])
        ax.set_title(title)
        ax.set_xlabel("env step")
        ax.grid(alpha=0.3)
    for ax in axes[len(avail):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)