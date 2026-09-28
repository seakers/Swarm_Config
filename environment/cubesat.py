# environment/cubesat.py
"""
A single CubeSat module.

Holds geometric state (position, orientation), fixed local face capabilities,
and internal physical state (battery, temperature, power generation).

This class is a pure data holder + small helpers. All *transition* logic lives
in the environment/transition layer -- the module does not move itself.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np

from .geometry import (
    world_face_direction,
    world_face_capability_map,
    FaceCapability,
)

# --- Physical constants for mission evaluation (normalized, plausible) ---
# Thermal (steady-state cruise energy balance)
STEFAN_BOLTZMANN = 5.67e-8          # W / m^2 K^4
FACE_AREA = 0.01                    # m^2 (10cm x 10cm CubeSat face)
RADIATOR_EMISSIVITY = 0.85          # high-emissivity radiator faces
PASSIVE_EMISSIVITY = 0.10           # other faces radiate weakly
SOLAR_ABSORPTIVITY = 0.30           # fraction of incident solar absorbed
SOLAR_CONSTANT = 1361.0             # W / m^2 (near Earth; deep space adjust later)
MIN_POWER_DISSIPATION = 2.0         # W per module at minimum (cruise) power
CONDUCTION_COEFF = 0.20             # W / K between connected modules
SPACE_TEMPERATURE = 2.7             # K (radiative sink)

# Comms
COMMS_BETA = 1.0                    # link-budget scaling for coherent gain

@dataclass
class CubeSat:
    id: int
    position: Tuple[int, int, int]
    orientation: int  # index into ROTATIONS (0..23)
    # face_capabilities[local_face] = capability id, fixed in the module's frame
    face_capabilities: List[int] = field(
        default_factory=lambda: [FaceCapability.STRUCTURAL] * 6
    )
    battery: float = 1.0        # normalized 0..1
    temperature: float = 20.0   # arbitrary units (deg C conceptually)
    power_generation: float = 0.0  # last-computed generation (mission fills this)

    def copy(self) -> "CubeSat":
        return CubeSat(
            id=self.id,
            position=tuple(int(v) for v in self.position),
            orientation=int(self.orientation),
            face_capabilities=list(self.face_capabilities),
            battery=float(self.battery),
            temperature=float(self.temperature),
            power_generation=float(self.power_generation),
        )

    def world_face_directions(self):
        """Return list of (world_direction, capability) for all six faces."""
        result = []
        for local_face in range(6):
            wdir = world_face_direction(self.orientation, local_face)
            cap = self.face_capabilities[local_face]
            result.append((wdir, cap))
        return result

    def capability_facing(self, world_face_index: int) -> int:
        """Capability of the face currently pointing along the given world dir."""
        mapping = world_face_capability_map(self.orientation)
        local_face = mapping[world_face_index]
        return self.face_capabilities[local_face]