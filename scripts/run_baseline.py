# scripts/run_baseline.py
"""
Run a non-learning baseline controller on the environment and report metrics.

Usage:
    python -m scripts.run_baseline --controller greedy --n 4
    python -m scripts.run_baseline --controller greedy --mode sequential --n 6
    python -m scripts.run_baseline --controller random --n 4
    python -m scripts.run_baseline --controller greedy --n 4 --visualize
"""
from __future__ import annotations
import argparse

from environment.make_env import make_env
from controllers.random_controller import RandomController
from baselines.greedy import GreedyController
from baselines.astar_controller import AStarController
from evaluation.metrics import run_episode


def make_controller(name, env, args):
    if name == "random":
        return RandomController(env, seed=args.seed, respect_legality=True)
    if name == "greedy":
        return GreedyController(env, mode=args.mode, max_movers=args.max_movers)
    if name == "astar":
        target_pos = target_orient = None
        # If a target is desired, capture the CURRENT config as goal after
        # scrambling — handled by scripts that set it explicitly. Here we
        # default to objective-maximizing (no target).
        return AStarController(
            env,
            target_positions=target_pos,
            target_orientations=target_orient,
            movement_cost=args.astar_cost,
            max_expansions=args.astar_expansions,
        )
    raise ValueError(f"Unknown controller {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", type=str, default="greedy",
                    choices=["random", "greedy", "astar"])
    ap.add_argument("--astar-cost", type=float, default=1.0)
    ap.add_argument("--astar-expansions", type=int, default=50_000)
    ap.add_argument("--mode", type=str, default="joint",
                    choices=["joint", "sequential"])
    ap.add_argument("--max-movers", type=int, default=1)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--shape", type=str, default="random", choices=["line", "L", "random"])
    ap.add_argument("--mission", type=str, default="power")
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--visualize", action="store_true")
    args = ap.parse_args()

    env = make_env(n_modules=args.n, shape=args.shape, mission=args.mission,
                   max_steps=args.steps, seed=args.seed, sun_direction=(0, 0, 1))
    controller = make_controller(args.controller, env, args)

    metrics, frames = run_episode(env, controller,
                                  collect_frames=args.visualize)

    print("=" * 60)
    print(f"Controller: {args.controller} "
          f"({args.mode if args.controller == 'greedy' else ''})")
    print(f"N={args.n} shape={args.shape} mission={args.mission} seed={args.seed}")
    print("=" * 60)
    for k, v in metrics.summary().items():
        print(f"  {k}: {v}")

    if args.controller == "astar" and controller.result is not None:
        r = controller.result
        print("-" * 60)
        print(f"  A* planning: success={r.success} reason='{r.reason}'")
        print(f"  A* plan length (moves): {r.moves}")
        print(f"  A* node expansions: {r.expansions}")

    if args.visualize:
        from evaluation.visualization import animate_configurations
        import matplotlib.pyplot as plt
        anim = animate_configurations(frames, interval=600)
        plt.show()


if __name__ == "__main__":
    main()