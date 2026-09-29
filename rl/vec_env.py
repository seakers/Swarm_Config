# rl/vec_env.py
"""
Subprocess vector environment for parallel rollout collection.

PROTOCOL: both reset() and step() return, per env, a 7-tuple describing that
env's CURRENT state:

    (global_obs, all_local_obs, mask, id_order, reward, done, objective)

reset() returns reward=0.0, done=False. Workers auto-reset a finished env and
return the FRESH state, so the main loop never stalls.

Main-loop invariant:
    state = vec.reset()
    loop:
        act on `state`  ->  joint_actions
        results = vec.step(joint_actions)
        store transition using the state we acted on + results' reward/done
        state = results            # overwrite with next state

Workers compute the action mask from the same env state they return as obs, so
mask and obs are always consistent.
"""

from __future__ import annotations
import multiprocessing as mp
from typing import Callable, List
import numpy as np


# ---------------------------------------------------------------- task factory
class TaskSampler:
    """Picklable env factory: samples a random (N, mission) on each call."""

    def __init__(self, base_seed, n_choices, mission_choices, shape,
                 sun, earth, max_steps, reward_mode, offset=0):
        self.base_seed = base_seed
        self.n_choices = list(n_choices)
        self.mission_choices = list(mission_choices)
        self.shape = shape
        self.sun = tuple(sun)
        self.earth = tuple(earth)
        self.max_steps = max_steps
        self.reward_mode = reward_mode
        self._i = offset

    def __call__(self):
        from environment.make_env import make_env
        self._i += 1
        seed = self.base_seed * 100000 + self._i
        rng = np.random.default_rng(seed)
        n = int(rng.choice(self.n_choices))
        mission = str(rng.choice(self.mission_choices))
        return make_env(
            n_modules=n, shape=self.shape, mission=mission,
            sun_direction=self.sun, earth_direction=self.earth,
            max_steps=self.max_steps, reward_mode=self.reward_mode, seed=seed)


# ---------------------------------------------------------------- state helper
def _env_view(env, reward=0.0, done=False):
    """Build the 7-tuple current-state view for one env."""
    from rl.action_masking import node_action_mask
    id_order = sorted(m.id for m in env.modules)
    mask = node_action_mask(env, id_order)
    return (env.global_observation(),
            env.all_local_observations(),
            mask,
            id_order,
            float(reward),
            bool(done),
            float(env.objective_value()))


# ---------------------------------------------------------------- worker
def _worker(remote, env_fns):
    envs = [fn() for fn in env_fns]
    for e in envs:
        e.reset()
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                out = []
                for e, joint in zip(envs, data):
                    _, reward, term, trunc, info = e.step(joint)
                    done = bool(term or trunc)
                    if not np.isfinite(reward):
                        reward = 0.0
                    if done:
                        e.reset()          # auto-reset: new task sampled
                    out.append(_env_view(e, reward=reward, done=done))
                remote.send(out)
            elif cmd == "reset":
                out = []
                for e in envs:
                    e.reset()
                    out.append(_env_view(e))
                remote.send(out)
            elif cmd == "close":
                remote.close()
                break
    except KeyboardInterrupt:
        pass


# ---------------------------------------------------------------- vec envs
class SubprocVecEnv:
    """Runs env_fns across n_workers subprocesses."""

    def __init__(self, env_fns: List[Callable], n_workers: int = 8):
        self.n_envs = len(env_fns)
        n_workers = max(1, min(n_workers, self.n_envs))
        # Round-robin assignment: worker w owns global indices w, w+nw, w+2nw...
        self._layout = [list(range(w, self.n_envs, n_workers))
                        for w in range(n_workers)]
        chunks = [[env_fns[g] for g in idxs] for idxs in self._layout]

        ctx = mp.get_context("spawn")
        pipes = [ctx.Pipe() for _ in range(n_workers)]
        self.remotes = [p[0] for p in pipes]
        work_remotes = [p[1] for p in pipes]
        self.procs = []
        for wr, chunk in zip(work_remotes, chunks):
            p = ctx.Process(target=_worker, args=(wr, chunk), daemon=True)
            p.start()
            self.procs.append(p)
        for wr in work_remotes:
            wr.close()

    def _flatten(self, per_worker):
        out = [None] * self.n_envs
        for w, idxs in enumerate(self._layout):
            for local_i, global_i in enumerate(idxs):
                out[global_i] = per_worker[w][local_i]
        return out

    def step(self, joint_actions: List[dict]):
        for w, remote in enumerate(self.remotes):
            remote.send(("step", [joint_actions[g] for g in self._layout[w]]))
        return self._flatten([r.recv() for r in self.remotes])

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        return self._flatten([r.recv() for r in self.remotes])

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)


class SerialVecEnv:
    """Fallback with an identical API; all envs step in-process."""

    def __init__(self, env_fns: List[Callable]):
        self.n_envs = len(env_fns)
        self.envs = [fn() for fn in env_fns]
        for e in self.envs:
            e.reset()

    def step(self, joint_actions):
        out = []
        for e, joint in zip(self.envs, joint_actions):
            _, reward, term, trunc, info = e.step(joint)
            done = bool(term or trunc)
            if not np.isfinite(reward):
                reward = 0.0
            if done:
                e.reset()
            out.append(_env_view(e, reward=reward, done=done))
        return out

    def reset(self):
        out = []
        for e in self.envs:
            e.reset()
            out.append(_env_view(e))
        return out

    def close(self):
        pass


def make_vec_env(env_fns, n_workers=8, use_subproc=True):
    """Build a vector env, falling back to serial on any failure."""
    if use_subproc and n_workers > 1:
        try:
            return SubprocVecEnv(env_fns, n_workers=n_workers)
        except Exception as e:
            print(f"[vec_env] Subprocess start failed ({e}); using serial.")
    return SerialVecEnv(env_fns)


def build_task_samplers(cfg, n_envs):
    """Create one TaskSampler per env slot from a trainer config."""
    earth = getattr(cfg, "earth_direction", (0.0, 1.0, 0.0))
    return [TaskSampler(cfg.seed + k, cfg.n_modules_choices,
                        cfg.mission_choices, cfg.shape,
                        cfg.sun_direction, earth,
                        cfg.max_steps, cfg.reward_mode, offset=k)
            for k in range(n_envs)]