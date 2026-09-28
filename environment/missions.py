# environment/missions.py
"""
Mission definitions and objective evaluation.

A mission is: optimize ONE primary objective subject to HARD constraints.
Missions do NOT live inside the neural network. The environment queries the
mission object for (a) objective value and (b) constraint satisfaction.

No geometry math beyond querying module world-face directions is done here;
we import from geometry but keep spacecraft state ownership in the environment.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

from .geometry import FaceCapability, world_face_direction
from .cubesat import (
    STEFAN_BOLTZMANN, FACE_AREA, RADIATOR_EMISSIVITY, PASSIVE_EMISSIVITY,
    SOLAR_ABSORPTIVITY, SOLAR_CONSTANT, MIN_POWER_DISSIPATION,
    CONDUCTION_COEFF, SPACE_TEMPERATURE, COMMS_BETA,
)

Pos = Tuple[int, int, int]


@dataclass
class MissionConfig:
    name: str
    primary_objective: str          # e.g. "power"
    direction: str                  # "maximize" or "minimize"
    # Hard constraint limits:
    battery_min: float = 0.0
    temperature_max: float = 120.0
    require_connected: bool = True
    require_no_overlap: bool = True
    max_steps: int = 100
    # Environment info:
    sun_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    earth_direction: Tuple[float, float, float] = (0.0, 1.0, 0.0)


class Mission:
    """Base class. Subclasses implement objective()."""

    def __init__(self, config: MissionConfig):
        self.config = config

    # --- Objective ---
    def objective(self, modules: List) -> float:
        raise NotImplementedError

    # --- Hard constraints ---
    def constraints_satisfied(self, modules, positions_by_id) -> Tuple[bool, Dict[str, bool]]:
        cfg = self.config
        checks: Dict[str, bool] = {}

        # Overlap (should be impossible by construction, but verify).
        if cfg.require_no_overlap:
            positions = list(positions_by_id.values())
            checks["no_overlap"] = len(set(positions)) == len(positions)

        # Connectivity checked by environment/graph; we expect a flag passed in.
        # Battery / thermal:
        checks["battery_ok"] = all(m.battery >= cfg.battery_min for m in modules)
        checks["temperature_ok"] = all(
            m.temperature <= cfg.temperature_max for m in modules
        )
        ok = all(checks.values())
        return ok, checks


class PowerMission(Mission):
    """Maximize solar power generation.

    A solar-panel face generates power proportional to how directly it faces
    the sun (dot product with sun direction, clamped at 0), and reduced if it
    is occluded by an adjacent module on that face.
    """

    def objective(self, modules) -> float:
        sun = np.array(self.config.sun_direction, dtype=float)
        norm = np.linalg.norm(sun)
        if norm > 0:
            sun = sun / norm

        positions = {tuple(m.position) for m in modules}
        total = 0.0
        for m in modules:
            for local_face in range(6):
                if m.face_capabilities[local_face] != FaceCapability.SOLAR_PANEL:
                    continue
                wdir = np.array(world_face_direction(m.orientation, local_face),
                                dtype=float)
                # Occlusion: is the adjacent cell in that direction occupied?
                adj = (m.position[0] + int(wdir[0]),
                       m.position[1] + int(wdir[1]),
                       m.position[2] + int(wdir[2]))
                if adj in positions:
                    continue  # occluded, no generation
                illumination = max(0.0, float(np.dot(wdir, sun)))
                total += illumination
        return total


class ThermalMission(Mission):
    """Minimize the worst-case steady-state temperature in long-term cruise.

    Per-module energy balance solved to steady state:
        Q_dissipation + Q_solar = Q_radiated(T) + Q_conduction(T)
    Radiator faces emit at high emissivity; other exposed faces emit weakly.
    Conduction couples physically connected modules. Objective (minimize) is
    the maximum steady-state temperature (hot-spot).
    """

    def _steady_state_temps(self, modules):
        sun = np.array(self.config.sun_direction, dtype=float)
        n = np.linalg.norm(sun)
        if n > 0:
            sun = sun / n
        positions = {tuple(m.position) for m in modules}
        pos_to_idx = {tuple(m.position): i for i, m in enumerate(modules)}
        N = len(modules)

        # Per-module: absorbed solar power, and effective emissivity*area.
        Q_abs = np.zeros(N)
        emis_area = np.zeros(N)   # sum over exposed faces of (eps * area)
        for i, m in enumerate(modules):
            for lf in range(6):
                wdir = np.array(world_face_direction(m.orientation, lf),
                                dtype=float)
                adj = (m.position[0] + int(wdir[0]),
                       m.position[1] + int(wdir[1]),
                       m.position[2] + int(wdir[2]))
                if adj in positions:
                    continue  # occluded face: no solar in, no radiation out
                # Emissivity depends on capability.
                if m.face_capabilities[lf] == FaceCapability.RADIATOR:
                    eps = RADIATOR_EMISSIVITY
                else:
                    eps = PASSIVE_EMISSIVITY
                emis_area[i] += eps * FACE_AREA
                illum = max(0.0, float(np.dot(wdir, sun)))
                Q_abs[i] += SOLAR_ABSORPTIVITY * SOLAR_CONSTANT * FACE_AREA * illum

        Q_in = Q_abs + MIN_POWER_DISSIPATION  # dissipation + absorbed solar

        # Neighbor conduction adjacency.
        neighbors = [[] for _ in range(N)]
        for i, m in enumerate(modules):
            for d in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
                adj = (m.position[0]+d[0], m.position[1]+d[1], m.position[2]+d[2])
                if adj in pos_to_idx:
                    neighbors[i].append(pos_to_idx[adj])

        # Fixed-point / Newton iteration on radiative + conductive balance.
        T = np.full(N, 250.0)  # K initial guess
        for _ in range(50):
            rad = emis_area * STEFAN_BOLTZMANN * (T**4 - SPACE_TEMPERATURE**4)
            cond = np.array([
                sum(CONDUCTION_COEFF * (T[i] - T[j]) for j in neighbors[i])
                for i in range(N)
            ])
            f = Q_in - rad - cond
            # Diagonal Jacobian approx: d/dT[i] of (-rad - cond)
            drad = emis_area * STEFAN_BOLTZMANN * 4.0 * T**3
            dcond = np.array([CONDUCTION_COEFF * len(neighbors[i])
                              for i in range(N)])
            denom = drad + dcond
            denom = np.where(denom < 1e-9, 1e-9, denom)
            T = T + f / denom
            T = np.clip(T, SPACE_TEMPERATURE, 1e4)
        return T

    def objective(self, modules) -> float:
        T = self._steady_state_temps(modules)
        return float(np.max(T))


class CommsMission(Mission):
    """Maximize data rate to Earth (deep-space downlink).

    Antenna faces combine coherently toward the earth direction. Effective
    transmit gain ~ |sum_i (antenna_i . earth) * unoccluded_i|^2, and
    datarate = log2(1 + beta * gain).
    """

    def objective(self, modules) -> float:
        earth = np.array(self.config.earth_direction, dtype=float)
        n = np.linalg.norm(earth)
        if n > 0:
            earth = earth / n
        positions = {tuple(m.position) for m in modules}

        coherent_sum = 0.0
        for m in modules:
            for lf in range(6):
                if m.face_capabilities[lf] != FaceCapability.ANTENNA:
                    continue
                wdir = np.array(world_face_direction(m.orientation, lf),
                                dtype=float)
                adj = (m.position[0]+int(wdir[0]),
                       m.position[1]+int(wdir[1]),
                       m.position[2]+int(wdir[2]))
                if adj in positions:
                    continue  # occluded
                contribution = max(0.0, float(np.dot(wdir, earth)))
                coherent_sum += contribution
        gain = coherent_sum ** 2
        return float(np.log2(1.0 + COMMS_BETA * gain))


class ApertureMission(Mission):
    """Maximize interferometric baseline (effective aperture).

    Science-instrument faces oriented toward Earth and unoccluded act as
    aperture elements. Objective = longest pairwise baseline projected onto the
    plane perpendicular to the earth (viewing) direction. Longer baseline =>
    finer angular resolution.
    """

    def objective(self, modules) -> float:
        earth = np.array(self.config.earth_direction, dtype=float)
        n = np.linalg.norm(earth)
        if n > 0:
            earth = earth / n
        positions = {tuple(m.position) for m in modules}

        elements = []
        for m in modules:
            for lf in range(6):
                if m.face_capabilities[lf] != FaceCapability.SCIENCE_INSTRUMENT:
                    continue
                wdir = np.array(world_face_direction(m.orientation, lf),
                                dtype=float)
                adj = (m.position[0]+int(wdir[0]),
                       m.position[1]+int(wdir[1]),
                       m.position[2]+int(wdir[2]))
                if adj in positions:
                    continue
                if float(np.dot(wdir, earth)) <= 0.0:
                    continue  # science face must point toward Earth
                elements.append(np.array(m.position, dtype=float))
                break  # one element per module max

        n_elements = len(elements)
        if n_elements < 2:
            return float(n_elements)   # 0 or 1 -> still gives signal to align

        max_baseline = 0.0
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                d = elements[i] - elements[j]
                d_perp = d - np.dot(d, earth) * earth
                b = float(np.linalg.norm(d_perp))
                if b > max_baseline:
                    max_baseline = b
        return float(n_elements) + max_baseline


MISSION_REGISTRY = {
    "power": PowerMission,
    "thermal": ThermalMission,
    "comms": CommsMission,
    "aperture": ApertureMission,
}


def make_mission(config: MissionConfig) -> Mission:
    cls = MISSION_REGISTRY[config.primary_objective]
    return cls(config)