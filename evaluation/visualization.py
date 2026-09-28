# evaluation/visualization.py
"""
Basic 3D visualization of the spacecraft using matplotlib. Renders cubes,
face capabilities (by color on face centers), and connections. Also supports
animating a sequence of configurations captured over an episode.

Visualization is NOT required during training.
"""

from __future__ import annotations
from typing import List, Optional
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    _MPL = True
except Exception:
    _MPL = False

from environment.geometry import (
    world_face_direction, FaceCapability, DIRECTIONS,
)

_CAP_COLORS = {
    FaceCapability.NONE: (0.7, 0.7, 0.7),
    FaceCapability.SOLAR_PANEL: (0.1, 0.2, 0.8),
    FaceCapability.ANTENNA: (0.9, 0.6, 0.1),
    FaceCapability.SCIENCE_INSTRUMENT: (0.1, 0.8, 0.3),
    FaceCapability.RADIATOR: (0.9, 0.1, 0.1),
    FaceCapability.STRUCTURAL: (0.6, 0.6, 0.6),
}


def _cube_faces(center, size=0.9):
    """Return the 6 face polygons of a cube centered at `center`."""
    cx, cy, cz = center
    r = size / 2.0
    corners = {}
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners[(sx, sy, sz)] = (cx + sx * r, cy + sy * r, cz + sz * r)
    faces = {
        0: [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],   # +X
        1: [(-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)],  # -X
        2: [(-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1)],   # +Y
        3: [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],  # -Y
        4: [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],   # +Z
        5: [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1)],  # -Z
    }
    return {fi: [corners[c] for c in cs] for fi, cs in faces.items()}


def render_modules(modules, ax=None, title="", axis_limits=None,
                   sun_direction=None, earth_direction=None):
    if not _MPL:
        raise RuntimeError("matplotlib not available")
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    ax.clear()
    all_pos = np.array([m.position for m in modules], dtype=float)

    for m in modules:
        faces = _cube_faces(m.position)
        for world_face, poly in faces.items():
            # Which capability points along this world face?
            cap = m.capability_facing(world_face)
            color = _CAP_COLORS.get(cap, (0.6, 0.6, 0.6))
            pc = Poly3DCollection([poly], alpha=0.85)
            pc.set_facecolor(color)
            pc.set_edgecolor((0, 0, 0))
            ax.add_collection3d(pc)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor=(0, 0, 0),
            label=FaceCapability.NAMES.get(cap, str(cap)).replace("_", " ").title(),
        )
        for cap, color in _CAP_COLORS.items()
    ]
    ax.legend(handles=legend_handles, title="Face capability", loc="upper left")

    # Bounds.  Supplying fixed limits is useful for animations, since
    # autoscaling each frame makes the scene appear to move.
    if axis_limits is None:
        mn = all_pos.min(axis=0) - 1
        mx = all_pos.max(axis=0) + 1
    else:
        mn, mx = axis_limits
    ax.set_xlim(mn[0], mx[0])
    ax.set_ylim(mn[1], mx[1])
    ax.set_zlim(mn[2], mx[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))

    # Draw directions as arrows starting at the origin.
    arrow_length = 0.8 * float(np.max(np.asarray(mx) - np.asarray(mn)))
    for direction, color, label in (
        (sun_direction, "orange", "Sun"),
        (earth_direction, "blue", "Earth"),
    ):
        if direction is None:
            continue
        direction = np.asarray(direction, dtype=float).reshape(-1)
        if direction.size != 3:
            raise ValueError(f"{label.lower()}_direction must have 3 components")
        norm = np.linalg.norm(direction)
        if norm == 0:
            continue
        vector = direction / norm * arrow_length
        ax.quiver(0, 0, 0, *vector, color=color, label=label,
                  arrow_length_ratio=0.12, linewidth=2)
        ax.text(*vector + (0, 0, 0.1), label, color=color, ha="center", va="center")
    if title:
        ax.set_title(title)
    return ax


def animate_configurations(frames: List[List], interval=600, save_path=None,
                           sun_direction=None, earth_direction=None):
    """Animate a list of module-list snapshots (deep-copied per frame)."""
    if not _MPL:
        raise RuntimeError("matplotlib not available")
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Compute one equal, cubic set of limits from every frame so neither the
    # scale nor the viewpoint's coordinate range changes during playback.
    positions = [m.position for frame in frames for m in frame]
    if not positions:
        raise ValueError("frames must contain at least one module")
    all_pos = np.asarray(positions, dtype=float)
    lower = all_pos.min(axis=0) - 1
    upper = all_pos.max(axis=0) + 1
    center = (lower + upper) / 2.0
    half_range = max(float(np.max(upper - lower)) / 2.0, 1.0)
    axis_limits = (center - half_range, center + half_range)

    def update(i):
        render_modules(
            frames[i], ax=ax, title=f"Step {i}", axis_limits=axis_limits,
            sun_direction=sun_direction, earth_direction=earth_direction,
        )

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, repeat=True)
    if save_path:
        anim.save(save_path, writer="pillow")
    return anim


def snapshot(env):
    """Deep-copy current modules for later animation."""
    return [m.copy() for m in env.modules]


# evaluation/visualization.py  (append)

def _aggregate(records, x_key, y_key):
    """Group records by (controller, x_key); return {controller: (xs, means, stds)}."""
    import numpy as np
    from collections import defaultdict
    grouped = defaultdict(lambda: defaultdict(list))
    for r in records:
        row = r.to_row()
        grouped[row["controller"]][row[x_key]].append(row[y_key])
    out = {}
    for ctrl, xmap in grouped.items():
        xs = sorted(xmap.keys())
        means = [float(np.mean(xmap[x])) for x in xs]
        stds = [float(np.std(xmap[x])) for x in xs]
        out[ctrl] = (xs, means, stds)
    return out


def plot_scaling(records, out_dir, mission, y_key="final_objective",
                 y_label=None):
    """Line plot of a metric vs N, one line per controller, for one mission."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    import os
    recs = [r for r in records if r.mission == mission]
    if not recs:
        return
    data = _aggregate(recs, "n_modules", y_key)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ctrl, (xs, means, stds) in sorted(data.items()):
        ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=ctrl)
    ax.set_xlabel("Number of modules (N)")
    ax.set_ylabel(y_label or y_key)
    ax.set_title(f"{mission}: {y_key} vs N")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"scaling_{mission}_{y_key}.png"), dpi=110)
    plt.close(fig)


def plot_reward_vs_moves(records, out_dir, mission):
    """Scatter of total_reward vs total_moves, colored by controller."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    import os
    from collections import defaultdict
    recs = [r for r in records if r.mission == mission]
    if not recs:
        return
    by_ctrl = defaultdict(lambda: ([], []))
    for r in recs:
        by_ctrl[r.controller][0].append(r.total_moves)
        by_ctrl[r.controller][1].append(r.total_reward)
    fig, ax = plt.subplots(figsize=(7, 5))
    for ctrl, (moves, rewards) in sorted(by_ctrl.items()):
        ax.scatter(moves, rewards, alpha=0.6, label=ctrl, s=30)
    ax.set_xlabel("Total moves")
    ax.set_ylabel("Total reward")
    ax.set_title(f"{mission}: reward vs reconfiguration cost")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"reward_vs_moves_{mission}.png"), dpi=110)
    plt.close(fig)


def plot_all_results(records, out_dir):
    """Generate the standard suite of plots for every mission present."""
    missions = sorted({r.mission for r in records})
    for mission in missions:
        plot_scaling(records, out_dir, mission, "final_objective",
                     "Final objective")
        plot_scaling(records, out_dir, mission, "total_moves",
                     "Total moves")
        plot_scaling(records, out_dir, mission, "distance_from_best",
                     "Distance from best-known")
        plot_reward_vs_moves(records, out_dir, mission)