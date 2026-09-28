"""
Interactive demo:
  1. Create an environment.
  2. Each step, print the legal moves for every module in human-readable form.
  3. Ask the user to pick a module and a move (or STAY / quit).
  4. Apply it, record a frame.
  5. When finished, animate the whole sequence in 3D.

Usage:
    python -m scripts.interactive_demo
    python -m scripts.interactive_demo --n 4 --shape line
    python -m scripts.interactive_demo --n 4 --save session.gif
"""

from __future__ import annotations
import argparse

from environment.make_env import make_env
from environment.actions import action_name
from evaluation.visualization import animate_configurations, snapshot


def print_legal_moves(env):
    """Print each module's legal moves and return the legal-action dict."""
    legal = env.legal_actions()
    print("\n--- Legal moves ---")
    for mid in sorted(legal.keys()):
        m = env._module(mid)
        actions = legal[mid]
        movable = [a for a in actions if a != 0]
        status = f"{len(movable)} move option(s)" if movable else "FROZEN (STAY only)"
        print(f"  Module {mid} at {m.position} (orient {m.orientation}): {status}")
        for a in actions:
            print(f"       [{a:2d}] {action_name(a)}")
    return legal


def prompt_move(env, legal):
    """Ask the user which module + action to take.

    Returns a joint action dict, or None to quit.
    """
    movable = [mid for mid, acts in legal.items() if len(acts) > 1]
    if not movable:
        print("\nNo module can move. Only STAY is available.")
        print("Press ENTER to STAY, or type 'q' to quit.")
        raw = input("> ").strip().lower()
        if raw == "q":
            return None
        return {mid: 0 for mid in legal}

    while True:
        print(f"\nMovable modules: {movable}")
        raw = input("Pick a module id (or 's' to STAY all, 'q' to quit): ").strip().lower()

        if raw == "q":
            return None
        if raw == "s":
            return {mid: 0 for mid in legal}

        # Parse module id.
        try:
            mid = int(raw)
        except ValueError:
            print("  Please enter a valid module id, 's', or 'q'.")
            continue
        if mid not in legal:
            print(f"  Module {mid} does not exist.")
            continue
        if len(legal[mid]) <= 1:
            print(f"  Module {mid} is frozen (no moves available). Pick another.")
            continue

        # Show that module's options and ask for the action.
        options = legal[mid]
        print(f"\n  Module {mid} options:")
        for a in options:
            print(f"       [{a:2d}] {action_name(a)}")
        act_raw = input("  Pick an action index (or 'b' to go back): ").strip().lower()

        if act_raw == "b":
            continue
        try:
            action = int(act_raw)
        except ValueError:
            print("  Please enter a valid action index.")
            continue
        if action not in options:
            print(f"  Action {action} is not legal for module {mid}.")
            continue

        # Build joint action: chosen module moves, everyone else stays.
        joint = {m: 0 for m in legal}
        joint[mid] = action
        return joint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="number of modules")
    ap.add_argument("--shape", type=str, default="line", choices=["line", "L"])
    ap.add_argument("--mission", type=str, default="power")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interval", type=int, default=800, help="ms per frame")
    ap.add_argument("--save", type=str, default=None, help="save gif to this path")
    args = ap.parse_args()

    # 1. Create the environment.
    env = make_env(n_modules=args.n, shape=args.shape,
                   mission=args.mission, seed=args.seed)
    env.reset()

    print("=" * 60)
    print(f"Interactive demo: {args.n} modules, shape={args.shape}, "
          f"mission={args.mission}")
    print("=" * 60)
    print(f"Initial objective: {env.objective_value():.3f}")
    print("\nAt each step: choose a module, then an action.")
    print("  's' = everyone stays this step")
    print("  'q' = finish and animate")

    frames = [snapshot(env)]  # starting configuration
    step = 0

    # 2-4. Interactive loop.
    while True:
        print("\n" + "=" * 60)
        print(f"STEP {step} | objective = {env.objective_value():.3f}")
        legal = print_legal_moves(env)

        joint = prompt_move(env, legal)
        if joint is None:
            print("\nFinishing session.")
            break

        obs, reward, term, trunc, info = env.step(joint)
        step += 1

        # Report what happened.
        moved_desc = []
        for mid, a in joint.items():
            if a != 0:
                moved_desc.append(f"M{mid}->{action_name(a)}")
        moved_str = ", ".join(moved_desc) if moved_desc else "all STAY"
        print(f"\n  Applied: {moved_str}")
        print(f"  moved={info['n_moved']} conflicts={info['n_conflicts']} "
              f"invalid={info['n_invalid']} obj={info['objective']:.3f} "
              f"reward={reward:+.3f}")
        if info["rejections"]:
            print(f"  rejections: {info['rejections']}")

        frames.append(snapshot(env))

        if term or trunc:
            print("\n  Episode ended (termination/truncation).")
            break

    print(f"\nFinal objective: {env.objective_value():.3f}")
    print(f"Recorded {len(frames)} frames.")

    # 5. Animate the sequence.
    if len(frames) < 2:
        print("Only the initial frame — nothing to animate.")
        return

    print("\nLaunching animation... (close the window to exit)")
    anim = animate_configurations(frames, interval=args.interval,
                                  save_path=args.save)
    if args.save:
        print(f"Saved animation to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()