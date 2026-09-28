# scripts/replay.py
"""
Replay a saved move-sequence trace and animate the reconfiguration.

A trace (produced by main.py under results/<id>/traces/) contains the initial
configuration and the per-step joint actions of the best run for a given
scenario + controller. This script rebuilds the exact environment, replays the
actions, and animates the sequence.

Usage:
    python -m scripts.replay results/20250101_120000/traces/power_n4_random_seed0__greedy_joint.json
    python -m scripts.replay <trace.json> --save out.gif --interval 500
"""
from __future__ import annotations
import argparse
import json

from environment.spacecraft import SpacecraftEnv, EnvConfig
from environment.missions import MissionConfig
from evaluation.visualization import animate_configurations, snapshot


def build_env_from_trace(trace: dict) -> SpacecraftEnv:
    ic = trace["initial_config"]
    n = ic["n_modules"]
    ids = sorted(int(k) for k in ic["positions"].keys())
    positions = [tuple(ic["positions"][str(i)]) if str(i) in ic["positions"]
                 else tuple(ic["positions"][i]) for i in ids]
    orientations = [ic["orientations"][str(i)] if str(i) in ic["orientations"]
                    else ic["orientations"][i] for i in ids]
    caps = [ic["face_capabilities"][str(i)] if str(i) in ic["face_capabilities"]
            else ic["face_capabilities"][i] for i in ids]

    direction = "minimize" if ic["mission"] == "thermal" else "maximize"
    mission_cfg = MissionConfig(
        name=f"{ic['mission']}_mission",
        primary_objective=ic["mission"],
        direction=direction,
        sun_direction=tuple(ic["sun_direction"]),
        earth_direction=tuple(ic["earth_direction"]),
        max_steps=ic["max_steps"],
    )
    env_cfg = EnvConfig(
        n_modules=n,
        initial_positions=positions,
        initial_orientations=orientations,
        face_capabilities=caps,
        mission_config=mission_cfg,
        seed=0,
    )
    return SpacecraftEnv(env_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=str, help="path to a trace JSON")
    ap.add_argument("--interval", type=int, default=600, help="ms per frame")
    ap.add_argument("--save", type=str, default=None, help="save gif to path")
    args = ap.parse_args()

    with open(args.trace) as f:
        trace = json.load(f)

    print(f"Replaying: {trace['scenario_id']} | {trace['controller']}")
    print(f"  final_objective={trace['final_objective']:.3f} "
          f"total_moves={trace['total_moves']} "
          f"steps={len(trace['action_trace'])}")

    env = build_env_from_trace(trace)
    env.reset()

    frames = [snapshot(env)]
    print(f"  initial objective = {env.objective_value():.3f}")

    for step, action in enumerate(trace["action_trace"]):
        # JSON keys may be strings; normalize to int module ids.
        joint = {int(k): int(v) for k, v in action.items()}
        obs, reward, term, trunc, info = env.step(joint)
        frames.append(snapshot(env))
        if term or trunc:
            break

    sun_direction = env.mission.config.sun_direction
    earth_direction = env.mission.config.earth_direction

    print(f"  final objective   = {env.objective_value():.3f}")
    print(f"  recorded {len(frames)} frames")

    anim = animate_configurations(frames, interval=args.interval,
                                  save_path=args.save, 
                                  sun_direction=sun_direction,
                                  earth_direction=earth_direction)
    if args.save:
        print(f"Saved animation to {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()