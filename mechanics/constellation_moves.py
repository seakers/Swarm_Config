import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

from core.constellation import Constellation, GroupState


@dataclass(frozen=True)
class SeparationAction:
    """
    Action to separate a subset of cubes from their group.
    """
    cube_ids: frozenset  # Cubes to separate (frozenset for hashability)
    separation_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    separation_velocity: float = 1.0  # m/s
    
    def __post_init__(self):
        # Ensure cube_ids is a frozenset
        if not isinstance(self.cube_ids, frozenset):
            object.__setattr__(self, 'cube_ids', frozenset(self.cube_ids))


@dataclass(frozen=True)
class DockingAction:
    """
    Action to dock two groups together.
    """
    group_a_id: int
    group_b_id: int


@dataclass(frozen=True)
class ManeuverAction:
    """
    Action to perform a delta-v maneuver on a group.
    """
    group_id: int
    delta_v: Tuple[float, float, float]  # m/s


@dataclass 
class ConstellationActionResult:
    """
    Result of a constellation-level action.
    """
    success: bool
    reason: str = ""
    delta_v_used: float = 0.0
    new_group_id: Optional[int] = None
    merged_group_id: Optional[int] = None


class ConstellationController:
    """
    Controller that handles constellation-level actions.
    
    This sits above the MovementSystem and handles:
    - Separation/docking decisions
    - Group-level maneuvers
    - Formation flying commands
    """
    
    def __init__(self, constellation: Constellation):
        self.constellation = constellation
    
    def get_valid_separation_actions(self) -> List[SeparationAction]:
        """
        Get all valid separation actions.
        
        This is a simplified version that considers separating:
        - Individual edge cubes
        - Pre-defined subgroups (halves, quarters)
        """
        valid_actions = []
        
        # For each group, find valid separation options
        for group_id, group in self.constellation._groups.items():
            if len(group.cube_ids) < 2 * self.constellation.separation_reqs.min_group_size:
                continue  # Can't split this group
            
            # Option 1: Find cubes that could separate individually
            if self.constellation.separation_reqs.allow_single_cube_groups:
                for cube_id in group.cube_ids:
                    # Check if this cube is on the edge (not fully surrounded)
                    connections = self.constellation.swarm._connections.get_connected_cubes(cube_id)
                    internal_connections = connections & group.cube_ids
                    
                    if len(internal_connections) < 6:  # Not fully surrounded
                        # Check if remaining cubes stay connected
                        remaining = group.cube_ids - {cube_id}
                        if self.constellation._is_cube_set_connected(remaining):
                            action = SeparationAction(
                                cube_ids=frozenset({cube_id}),
                                separation_direction=(1.0, 0.0, 0.0),
                                separation_velocity=1.0,
                            )
                            if self.constellation.can_separate({cube_id})[0]:
                                valid_actions.append(action)
            
            # Option 2: Find connected subgroups that could separate
            # Use BFS to find natural "chunks" that could break off
            subgroups = self._find_separable_subgroups(group)
            
            for subgroup in subgroups:
                if len(subgroup) >= self.constellation.separation_reqs.min_group_size:
                    remaining = group.cube_ids - subgroup
                    if len(remaining) >= self.constellation.separation_reqs.min_group_size:
                        action = SeparationAction(
                            cube_ids=frozenset(subgroup),
                            separation_direction=(1.0, 0.0, 0.0),
                            separation_velocity=1.0,
                        )
                        if self.constellation.can_separate(subgroup)[0]:
                            valid_actions.append(action)
        
        return valid_actions
    
    def _find_separable_subgroups(self, group: GroupState) -> List[Set[int]]:
        """
        Find natural subgroups that could be separated.
        
        Looks for "articulation points" - cubes whose removal would
        split the group into multiple components.
        """
        subgroups = []
        cube_ids = list(group.cube_ids)
        
        if len(cube_ids) < 4:
            return subgroups
        
        # Find articulation points using simple approach
        # An articulation point is a cube whose removal disconnects the group
        
        for test_cube in cube_ids:
            remaining = set(cube_ids) - {test_cube}
            
            if not self.constellation._is_cube_set_connected(remaining):
                # This cube is an articulation point
                # Find the components that would result
                components = self._find_components_without_cube(group.cube_ids, test_cube)
                
                for component in components:
                    if (len(component) >= self.constellation.separation_reqs.min_group_size and
                        len(component) < len(group.cube_ids)):
                        subgroups.append(component)
        
        # Also try splitting roughly in half along each axis
        positions = {}
        for cid in cube_ids:
            cube = self.constellation.swarm.get_cube(cid)
            if cube:
                positions[cid] = cube.position
        
        if positions:
            for axis in range(3):
                sorted_cubes = sorted(positions.keys(), key=lambda c: positions[c][axis])
                mid = len(sorted_cubes) // 2
                
                first_half = set(sorted_cubes[:mid])
                second_half = set(sorted_cubes[mid:])
                
                # Check if both halves are connected
                if (self.constellation._is_cube_set_connected(first_half) and
                    self.constellation._is_cube_set_connected(second_half)):
                    if len(first_half) >= self.constellation.separation_reqs.min_group_size:
                        subgroups.append(first_half)
                    if len(second_half) >= self.constellation.separation_reqs.min_group_size:
                        subgroups.append(second_half)
        
        # Remove duplicates
        unique_subgroups = []
        seen = set()
        for sg in subgroups:
            sg_frozen = frozenset(sg)
            if sg_frozen not in seen:
                seen.add(sg_frozen)
                unique_subgroups.append(sg)
        
        return unique_subgroups
    
    def _find_components_without_cube(self, cube_ids: Set[int], 
                                    excluded: int) -> List[Set[int]]:
        """Find connected components after removing a cube."""
        remaining = cube_ids - {excluded}
        components = []
        visited = set()
        
        for start in remaining:
            if start in visited:
                continue
            
            # BFS to find component
            component = {start}
            queue = [start]
            visited.add(start)  # Mark as visited immediately
            
            while queue:
                current = queue.pop(0)
                connections = self.constellation.swarm._connections.get_connected_cubes(current)
                
                for neighbor in connections:
                    if neighbor in remaining and neighbor not in visited:
                        visited.add(neighbor)  # Mark visited when adding to queue
                        component.add(neighbor)
                        queue.append(neighbor)
            
            components.append(component)
        
        return components
    
    def get_valid_docking_actions(self) -> List[DockingAction]:
        """
        Get all valid docking actions.
        """
        valid_actions = []
        
        group_ids = list(self.constellation._groups.keys())
        
        for i, ga_id in enumerate(group_ids):
            for gb_id in group_ids[i+1:]:
                can_dock, _ = self.constellation.can_dock(ga_id, gb_id)
                if can_dock:
                    valid_actions.append(DockingAction(ga_id, gb_id))
        
        return valid_actions
    
    def get_valid_maneuver_actions(self, 
                                    max_delta_v: float = 1.0,
                                    directions: Optional[List[Tuple[float, float, float]]] = None
                                    ) -> List[ManeuverAction]:
        """
        Get valid maneuver actions for all groups.
        
        Args:
            max_delta_v: Maximum delta-v per maneuver
            directions: List of directions to consider (default: 6 cardinal + stop)
        """
        if directions is None:
            directions = [
                (1, 0, 0), (-1, 0, 0),
                (0, 1, 0), (0, -1, 0),
                (0, 0, 1), (0, 0, -1),
                (0, 0, 0),  # Zero - no maneuver
            ]
        
        valid_actions = []
        
        for group_id, group in self.constellation._groups.items():
            for direction in directions:
                # Scale direction by max_delta_v
                delta_v = tuple(d * max_delta_v for d in direction)
                delta_v_mag = np.linalg.norm(delta_v)
                
                if delta_v_mag <= group.propulsion.remaining_delta_v:
                    valid_actions.append(ManeuverAction(group_id, delta_v))
        
        return valid_actions
    
    def execute_separation(self, action: SeparationAction) -> ConstellationActionResult:
        """Execute a separation action."""
        cube_ids = set(action.cube_ids)
        direction = np.array(action.separation_direction)
        
        success, reason, new_group_id = self.constellation.separate(
            cube_ids, direction, action.separation_velocity
        )
        
        delta_v_used = action.separation_velocity if success else 0.0
        
        return ConstellationActionResult(
            success=success,
            reason=reason,
            delta_v_used=delta_v_used,
            new_group_id=new_group_id,
        )
    
    def execute_docking(self, action: DockingAction) -> ConstellationActionResult:
        """Execute a docking action."""
        success, reason = self.constellation.dock(action.group_a_id, action.group_b_id)
        
        delta_v_used = self.constellation.docking_reqs.docking_delta_v_cost if success else 0.0
        
        return ConstellationActionResult(
            success=success,
            reason=reason,
            delta_v_used=delta_v_used,
            merged_group_id=action.group_a_id if success else None,
        )
    
    def execute_maneuver(self, action: ManeuverAction) -> ConstellationActionResult:
        """Execute a maneuver action."""
        delta_v = np.array(action.delta_v)
        delta_v_mag = float(np.linalg.norm(delta_v))
        
        success, reason = self.constellation.apply_delta_v(action.group_id, delta_v)
        
        return ConstellationActionResult(
            success=success,
            reason=reason,
            delta_v_used=delta_v_mag if success else 0.0,
        )