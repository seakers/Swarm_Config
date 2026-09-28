# scripts/audit_pivot.py
"""
Visual + numeric audit of a single hinge pivot.

Runs one specific HingeMove on a small configuration, prints the resolved
result (90 vs 180, destination, swept cells), and animates before -> after.

Usage:
    python -m scripts.audit_pivot --edge PX_PY --dir -1
    python -m scripts.audit_pivot --shape line --n 4 --module 0 --edge PX_PY --dir -1
"""
from __future__ import annotations
import argparse

from environment.make_env import make_env
from environment.geometry import Edge
from environment.transitions import resolve_hinge, _compute_swept_cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--shape", type=str, default="line")
    ap.add_argument("--module", type=int, default=0)
    ap.add_argument("--edge", type=str, default="PX_PY")
    ap.add_argument("--dir", type=int, default=-1, choices=[-1, 1])
    ap.add_argument("--no-visualize", action="store_true")
    args = ap.parse_args()

    env = make_env(n_modules=args.n, shape=args.shape)
    env.reset()

    positions = env.positions_by_id()
    orientations = env.orientations_by_id()
    m = env._module(args.module)
    edge = Edge[args.edge]

    print(f"Module {args.module} at {m.position}, orient {m.orientation}")
    print(f"Pivot edge {edge.name}, direction {args.dir:+d}")

    rm = resolve_hinge(m.position, m.orientation, args.module, edge, args.dir,
                       positions, require_connected=True)
    print(f"  success={rm.success} reason='{rm.reason}'")
    if rm.success:
        print(f"  degrees={rm.degrees}")
        print(f"  new_position={rm.new_position}")
        print(f"  new_orientation={rm.new_orientation}")
        swept = _compute_swept_cells(m.position, m.orientation, edge,
                                     args.dir, rm.degrees)
        print(f"  swept cells (world): {sorted(swept)}")

    if not args.no_visualize and rm.success:
        from evaluation.visualization import render_modules, snapshot
        import matplotlib.pyplot as plt

        before = snapshot(env)
        # Apply the move.
        m.position = rm.new_position
        m.orientation = rm.new_orientation
        env._update_module_physics()
        after = snapshot(env)

        fig = plt.figure(figsize=(11, 5))
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        render_modules(before, ax=ax1, title="Before")
        render_modules(after, ax=ax2,
                       title=f"After ({rm.degrees}°) -> {rm.new_position}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()