# tests/test_missions.py
"""Tests that the mission evaluator responds sensibly to configuration."""

import numpy as np

from environment.cubesat import CubeSat
from environment.geometry import FaceCapability
from environment.missions import PowerMission, MissionConfig


def _solar_module(mid, pos, orient=0):
    caps = [FaceCapability.STRUCTURAL] * 6
    caps[4] = FaceCapability.SOLAR_PANEL  # local +Z
    return CubeSat(id=mid, position=pos, orientation=orient,
                   face_capabilities=caps)


def test_power_rewards_sun_facing_panel():
    cfg = MissionConfig(name="p", primary_objective="power",
                        direction="maximize", sun_direction=(0, 0, 1))
    mission = PowerMission(cfg)
    modules = [_solar_module(0, (0, 0, 0))]
    # +Z panel directly faces +Z sun -> illumination 1.0.
    assert abs(mission.objective(modules) - 1.0) < 1e-6


def test_power_occlusion_reduces_output():
    cfg = MissionConfig(name="p", primary_objective="power",
                        direction="maximize", sun_direction=(0, 0, 1))
    mission = PowerMission(cfg)
    # Second module sits directly above (+Z) the first, occluding its panel.
    m0 = _solar_module(0, (0, 0, 0))
    m1 = _solar_module(1, (0, 0, 1))
    # m0's +Z panel is now occluded; only m1's +Z panel generates.
    val = mission.objective([m0, m1])
    assert abs(val - 1.0) < 1e-6