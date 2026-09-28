# environment/spacecraft.py
"""
The Spacecraft environment. This OWNS all state and transition logic.

Gymnasium-style API for the CENTRALIZED view, plus explicit accessors for the
DECENTRALIZED (per-module local + neighbor) observations.

The environment is deliberately independent of PPO/MAPPO and of any specific
controller. Controllers propose actions; the environment validates, resolves
conflicts, applies transitions, evaluates the mission, and returns rewards.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .cubesat import CubeSat
from .configuration import Configuration
from .geometry import (
    DIRECTIONS, world_face_direction, FaceCapability,
)
from .actions import NUM_ACTIONS
from .graph import build_adjacency, is_connected, connection_face
from .transitions import all_legal_single_moves
from .conflict_resolution import resolve
from .missions import Mission, MissionConfig, make_mission

Pos = Tuple[int, int, int]


@dataclass
class EnvConfig:
    n_modules: int
    initial_positions: List[Pos]
    initial_orientations: List[int]
    face_capabilities: List[List[int]]  # per-module local face capabilities
    mission_config: MissionConfig
    movement_cost: float = 0.05
    conflict_penalty: float = 0.02
    invalid_action_penalty: float = 0.02
    reward_mode: str = "improvement"   # "improvement" or "absolute"
    seed: int = 0


class SpacecraftEnv:
    """Multi-module reconfiguration environment."""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.mission: Mission = make_mission(config.mission_config)
        self.rng = np.random.default_rng(config.seed)
        self.modules: List[CubeSat] = []
        self.step_count = 0
        self._prev_objective = 0.0
        self._build_initial_modules()

    # ------------------------------------------------------------------ setup
    def _build_initial_modules(self):
        cfg = self.config
        self.modules = []
        for i in range(cfg.n_modules):
            self.modules.append(CubeSat(
                id=i,
                position=tuple(cfg.initial_positions[i]),
                orientation=int(cfg.initial_orientations[i]),
                face_capabilities=list(cfg.face_capabilities[i]),
                battery=1.0,
                temperature=20.0,
            ))

    # -------------------------------------------------------- state accessors
    def positions_by_id(self) -> Dict[int, Pos]:
        return {m.id: tuple(m.position) for m in self.modules}

    def orientations_by_id(self) -> Dict[int, int]:
        return {m.id: int(m.orientation) for m in self.modules}

    def configuration(self) -> Configuration:
        return Configuration.from_modules(self.modules)

    def adjacency(self) -> Dict[int, List[int]]:
        return build_adjacency(self.positions_by_id())

    def objective_value(self) -> float:
        return self.mission.objective(self.modules)

    # ------------------------------------------------------------- gym API
    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.config.seed = seed
            self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self._build_initial_modules()
        self._prev_objective = self.objective_value()
        return self.global_observation(), self._info()

    def step(self, joint_action: Dict[int, int]):
        """Advance one timestep given a joint action.

        joint_action: {module_id: action_index (0..6)}
        Missing modules are treated as STAY.
        """
        self.step_count += 1
        positions = self.positions_by_id()
        orientations = self.orientations_by_id()
        require_conn = self.mission.config.require_connected

        # 1. Compute per-module legal moves.
        legal = all_legal_single_moves(positions, orientations, require_connected=require_conn)

        # 2. Separate proposals into moving / invalid.
        proposals: Dict[int, Tuple[Pos, int]] = {}
        n_invalid = 0
        n_attempted_moves = 0
        for m in self.modules:
            a = joint_action.get(m.id, 0)
            if a == 0:
                continue  # STAY
            n_attempted_moves += 1
            if a in legal[m.id]:
                proposals[m.id] = legal[m.id][a]
            else:
                n_invalid += 1  # individually illegal action

        # 3. Resolve conflicts among individually-legal proposals.
        accepted, rejections = resolve(
            positions, orientations, proposals,
            require_connected=self.mission.config.require_connected,
        )
        n_conflicts = len(rejections)

        # 4. Apply accepted moves.
        n_moved = 0
        for mid, (dest, dest_orient) in accepted.items():
            mod = self._module(mid)
            mod.position = dest
            mod.orientation = dest_orient
            n_moved += 1

        # 5. Update mission-derived per-module quantities (power gen etc.).
        self._update_module_physics()

        # 6. Evaluate objective and reward.
        new_objective = self.objective_value()
        reward = self._compute_reward(
            new_objective, n_moved, n_conflicts, n_invalid
        )
        self._prev_objective = new_objective

        # 7. Constraints & termination.
        connected = is_connected(self.positions_by_id())
        ok_constraints, checks = self.mission.constraints_satisfied(
            self.modules, self.positions_by_id()
        )
        checks["connected"] = connected or not self.mission.config.require_connected
        terminated = not (checks["connected"] and ok_constraints)
        truncated = self.step_count >= self.mission.config.max_steps

        info = self._info()
        info.update({
            "n_moved": n_moved,
            "n_conflicts": n_conflicts,
            "n_invalid": n_invalid,
            "n_attempted_moves": n_attempted_moves,
            "rejections": rejections,
            "objective": new_objective,
            "constraints": checks,
            "constraints_ok": bool(checks["connected"] and ok_constraints),
        })
        return self.global_observation(), reward, terminated, truncated, info

    # ------------------------------------------------------------- rewards
    def _compute_reward(self, new_obj, n_moved, n_conflicts, n_invalid) -> float:
        cfg = self.config
        direction = 1.0 if self.mission.config.direction == "maximize" else -1.0
        if cfg.reward_mode == "improvement":
            obj_term = direction * (new_obj - self._prev_objective)
        else:  # absolute
            obj_term = direction * new_obj
        reward = obj_term
        reward -= cfg.movement_cost * n_moved
        reward -= cfg.conflict_penalty * n_conflicts
        reward -= cfg.invalid_action_penalty * n_invalid
        return float(reward)

    # ------------------------------------------------------------- physics
    def _update_module_physics(self):
        """Fill per-module power_generation for observation purposes.

        Kept simple: solar faces contribute illumination-based generation.
        Battery/thermal integration hooks live here for future extension.
        """
        sun = np.array(self.mission.config.sun_direction, dtype=float)
        n = np.linalg.norm(sun)
        if n > 0:
            sun = sun / n
        positions = self.positions_by_id_set()
        for m in self.modules:
            gen = 0.0
            for lf in range(6):
                if m.face_capabilities[lf] == FaceCapability.SOLAR_PANEL:
                    wdir = np.array(world_face_direction(m.orientation, lf),
                                    dtype=float)
                    adj = (m.position[0] + int(wdir[0]),
                           m.position[1] + int(wdir[1]),
                           m.position[2] + int(wdir[2]))
                    if adj not in positions:
                        gen += max(0.0, float(np.dot(wdir, sun)))
            m.power_generation = gen

    def positions_by_id_set(self):
        return {tuple(m.position) for m in self.modules}

    # ------------------------------------------------- observations (global)
    def global_observation(self) -> Dict:
        """Full global state for the centralized controller.

        Returns a structured dict; controllers flatten/encode as needed.
        """
        adjacency = self.adjacency()
        modules_state = []
        for m in self.modules:
            modules_state.append({
                "id": m.id,
                "position": np.array(m.position, dtype=np.float32),
                "orientation": m.orientation,
                "face_capabilities": np.array(m.face_capabilities, dtype=np.int64),
                "battery": np.float32(m.battery),
                "temperature": np.float32(m.temperature),
                "power_generation": np.float32(m.power_generation),
            })
        return {
            "modules": modules_state,
            "adjacency": adjacency,
            "sun_direction": np.array(self.mission.config.sun_direction,
                                      dtype=np.float32),
            "earth_direction": np.array(self.mission.config.earth_direction,
                                        dtype=np.float32),
            "mission": self.mission.config.primary_objective,
            "objective_value": np.float32(self.objective_value()),
        }

    # ---------------------------------------------- observations (local/decentr)
    def local_observation(self, module_id: int) -> Dict:
        """Local + neighbor observation for the decentralized controller.

        The module sees ONLY its own state and its physically connected
        neighbors' states (relative positions). No global information.
        Mission mode is provided (allowed per spec).
        """
        m = self._module(module_id)
        adjacency = self.adjacency()
        neighbor_ids = adjacency[module_id]

        own = {
            "position": np.array(m.position, dtype=np.float32),
            "orientation": m.orientation,
            "face_capabilities": np.array(m.face_capabilities, dtype=np.int64),
            "battery": np.float32(m.battery),
            "temperature": np.float32(m.temperature),
            "power_generation": np.float32(m.power_generation),
        }
        neighbors = []
        for nid in neighbor_ids:
            nb = self._module(nid)
            rel = (nb.position[0] - m.position[0],
                   nb.position[1] - m.position[1],
                   nb.position[2] - m.position[2])
            neighbors.append({
                "id": nid,
                "relative_position": np.array(rel, dtype=np.float32),
                "connection_face": connection_face(m.position, nb.position),
                "orientation": nb.orientation,
                "face_capabilities": np.array(nb.face_capabilities, dtype=np.int64),
                "battery": np.float32(nb.battery),
                "temperature": np.float32(nb.temperature),
                "power_generation": np.float32(nb.power_generation),
            })
        return {
            "own": own,
            "neighbors": neighbors,
            "mission": self.mission.config.primary_objective,  # global, allowed
        }

    def all_local_observations(self) -> Dict[int, Dict]:
        return {m.id: self.local_observation(m.id) for m in self.modules}

    # ------------------------------------------------- action-space helpers
    def legal_actions(self) -> Dict[int, List[int]]:
        """Return {module_id: [legal action indices including STAY]}."""
        require_conn = self.mission.config.require_connected
        legal = all_legal_single_moves(
            self.positions_by_id(), self.orientations_by_id(), require_connected=require_conn
        )
        result = {}
        for m in self.modules:
            result[m.id] = [0] + sorted(legal[m.id].keys())
        return result

    # ------------------------------------------------------------- utilities
    def _module(self, module_id: int) -> CubeSat:
        for m in self.modules:
            if m.id == module_id:
                return m
        raise KeyError(f"No module with id {module_id}")

    def _info(self) -> Dict:
        return {
            "step": self.step_count,
            "n_modules": len(self.modules),
            "connected": is_connected(self.positions_by_id()),
            "objective": self.objective_value(),
        }

    def render_state_text(self) -> str:
        lines = [f"Step {self.step_count} | obj={self.objective_value():.3f} "
                 f"| connected={is_connected(self.positions_by_id())}"]
        for m in self.modules:
            lines.append(f"  M{m.id}: pos={m.position} orient={m.orientation} "
                         f"batt={m.battery:.2f} temp={m.temperature:.1f}")
        return "\n".join(lines)