# environment/make_env.py
"""
Convenience factory turning high-level parameters into an EnvConfig + env.
Central place so scripts and tests share identical construction logic.
"""

from __future__ import annotations
from typing import Optional

from .spacecraft import SpacecraftEnv, EnvConfig
from .missions import MissionConfig
from .configuration_builder import (
    line_configuration, L_configuration,
    default_face_capabilities, default_orientations,
    random_orientations, rand_configuration
)


def make_env(n_modules: int = 4,
             shape: str = "random",
             mission: str = "power",
             direction: str = "maximize",
             sun_direction=(0.0, 0.0, 1.0),
             earth_direction=(0.0, 1.0, 0.0),
             max_steps: int = 100,
             reward_mode: str = "improvement",
             seed: int = 0) -> SpacecraftEnv:
    if shape == "line":
        positions = line_configuration(n_modules)
        orientations = default_orientations(n_modules)
    elif shape == "L":
        positions = L_configuration(n_modules)
        orientations = default_orientations(n_modules)
    elif shape == "random":
        positions = rand_configuration(n_modules, seed=seed)
        orientations = random_orientations(n_modules, seed=seed)
    else:
        raise ValueError(f"Unknown shape {shape}")

    if mission == "thermal":
        direction = "minimize"

    mission_cfg = MissionConfig(
        name=f"{mission}_mission",
        primary_objective=mission,
        direction=direction,
        sun_direction=sun_direction,
        earth_direction=earth_direction,
        max_steps=max_steps,
    )
    env_cfg = EnvConfig(
        n_modules=n_modules,
        initial_positions=positions,
        initial_orientations=orientations,
        face_capabilities=default_face_capabilities(n_modules),
        mission_config=mission_cfg,
        reward_mode=reward_mode,
        seed=seed,
    )
    return SpacecraftEnv(env_cfg)