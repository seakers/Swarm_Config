import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum, auto

from core.cube import Orientation
from core.swarm import Swarm


class PropulsionType(Enum):
    """Types of propulsion available."""
    COLD_GAS = auto()
    ELECTROSPRAY = auto()
    NONE = auto()


@dataclass
class PropulsionSubsystem:
    """
    Propulsion capability for a cube or group.
    
    Delta-V is the total velocity change capability remaining.
    Once depleted, the cube cannot perform maneuvers.
    """
    propulsion_type: PropulsionType = PropulsionType.COLD_GAS
    
    # Total delta-v capacity (m/s)
    max_delta_v: float = 50.0
    
    # Remaining delta-v (m/s)  
    remaining_delta_v: float = 50.0
    
    # Maximum thrust (N) - determines how fast maneuvers can be
    max_thrust: float = 0.01  # 10 mN typical for cubesat
    
    # Specific impulse (s) - efficiency measure
    specific_impulse: float = 70.0  # Cold gas typical
    
    # Is the system functional?
    is_functional: bool = True
    
    def can_perform_maneuver(self, delta_v_required: float) -> bool:
        """Check if we have enough delta-v for a maneuver."""
        return (self.is_functional and 
                self.remaining_delta_v >= delta_v_required)
    
    def perform_maneuver(self, delta_v_used: float) -> bool:
        """
        Use delta-v for a maneuver.
        
        Returns True if successful, False if insufficient delta-v.
        """
        if not self.can_perform_maneuver(delta_v_used):
            return False
        
        self.remaining_delta_v -= delta_v_used
        return True
    
    def get_remaining_fraction(self) -> float:
        """Get remaining delta-v as fraction of capacity."""
        return self.remaining_delta_v / self.max_delta_v if self.max_delta_v > 0 else 0
    
    def copy(self) -> 'PropulsionSubsystem':
        return PropulsionSubsystem(
            propulsion_type=self.propulsion_type,
            max_delta_v=self.max_delta_v,
            remaining_delta_v=self.remaining_delta_v,
            max_thrust=self.max_thrust,
            specific_impulse=self.specific_impulse,
            is_functional=self.is_functional,
        )


@dataclass
class GroupState:
    """
    State of a separated group of cubes.
    
    Each group has:
    - A set of cube IDs that belong to it
    - A position in continuous 3D space (meters from constellation origin)
    - A velocity vector (m/s)
    - Aggregate propulsion capability
    """
    group_id: int
    cube_ids: Set[int] = field(default_factory=set)
    
    # Position in continuous space (meters)
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Velocity (m/s)
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Group-level propulsion (aggregated from member cubes)
    # Or could track per-cube propulsion
    propulsion: PropulsionSubsystem = field(default_factory=PropulsionSubsystem)
    
    def get_distance_to(self, other: 'GroupState') -> float:
        """Get distance to another group in meters."""
        return float(np.linalg.norm(self.position - other.position))
    
    def get_relative_velocity(self, other: 'GroupState') -> np.ndarray:
        """Get velocity relative to another group."""
        return self.velocity - other.velocity
    
    def get_relative_speed(self, other: 'GroupState') -> float:
        """Get relative speed to another group."""
        return float(np.linalg.norm(self.get_relative_velocity(other)))
    
    def copy(self) -> 'GroupState':
        return GroupState(
            group_id=self.group_id,
            cube_ids=self.cube_ids.copy(),
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            propulsion=self.propulsion.copy(),
        )


@dataclass
class SeparationRequirements:
    """
    Requirements and costs for separation maneuvers.
    """
    # Minimum delta-v needed to separate (m/s)
    min_separation_delta_v: float = 0.5
    
    # Default separation velocity (m/s)
    default_separation_velocity: float = 1.0
    
    # Minimum cubes per group after separation
    min_group_size: int = 1
    
    # Maximum number of separate groups allowed
    max_groups: int = 8
    
    # Can single cubes separate, or must stay in groups?
    allow_single_cube_groups: bool = True


@dataclass
class DockingRequirements:
    """
    Requirements for docking/rejoining maneuvers.
    """
    # Maximum relative velocity for safe docking (m/s)
    max_docking_velocity: float = 0.1
    
    # Maximum distance for docking initiation (m)
    max_docking_range: float = 100.0
    
    # Delta-v cost for final approach and docking (m/s)
    docking_delta_v_cost: float = 0.5
    
    # Alignment tolerance (radians)
    max_alignment_error: float = 0.1  # ~5.7 degrees
    
    # Time required for docking sequence (seconds)
    docking_duration: float = 300.0  # 5 minutes


@dataclass 
class CommunicationRequirements:
    """
    Communication constraints between separated groups.
    """
    # Maximum range for direct inter-satellite link (m)
    max_isl_range: float = 100_000.0  # 100 km
    
    # Data rate vs distance (simplified model)
    # rate = base_rate * (reference_distance / distance)^2
    base_data_rate: float = 10_000.0  # bits/sec at reference distance
    reference_distance: float = 1_000.0  # 1 km
    
    # Minimum data rate for coordination (bits/sec)
    min_coordination_rate: float = 100.0
    
    def get_data_rate(self, distance: float) -> float:
        """Get achievable data rate at a given distance."""
        if distance <= 0:
            return self.base_data_rate
        
        rate = self.base_data_rate * (self.reference_distance / distance) ** 2
        return max(0, rate)
    
    def can_communicate(self, distance: float) -> bool:
        """Check if groups can communicate at this distance."""
        return distance <= self.max_isl_range
    
    def can_coordinate(self, distance: float) -> bool:
        """Check if groups can coordinate maneuvers at this distance."""
        rate = self.get_data_rate(distance)
        return rate >= self.min_coordination_rate


class Constellation:
    """
    Manages a constellation of cube groups that may be separated in space.
    
    This is the top-level container that replaces or wraps Swarm when
    dealing with separated groups.
    
    Key responsibilities:
    - Track multiple groups and their positions
    - Handle separation and docking operations
    - Propagate group positions over time
    - Enforce communication and propulsion constraints
    """
    
    def __init__(self, 
                 swarm: 'Swarm',
                 separation_reqs: Optional[SeparationRequirements] = None,
                 docking_reqs: Optional[DockingRequirements] = None,
                 comm_reqs: Optional[CommunicationRequirements] = None):
        """
        Initialize constellation from a swarm.
        
        Initially all cubes are in a single group at the origin.
        
        Args:
            swarm: The underlying swarm (manages cube properties and grid positions)
            separation_reqs: Requirements for separation maneuvers
            docking_reqs: Requirements for docking maneuvers
            comm_reqs: Communication constraints
        """
        self.swarm = swarm
        self.separation_reqs = separation_reqs or SeparationRequirements()
        self.docking_reqs = docking_reqs or DockingRequirements()
        self.comm_reqs = comm_reqs or CommunicationRequirements()
        
        # Initialize all cubes in a single group
        self._groups: Dict[int, GroupState] = {}
        self._cube_to_group: Dict[int, int] = {}
        self._next_group_id = 0
        
        # Create initial group with all cubes
        initial_cubes = set(range(swarm.num_cubes))
        self._create_group(initial_cubes, np.zeros(3), np.zeros(3))
        
        # Per-cube propulsion (cubes contribute to group propulsion when grouped)
        self._cube_propulsion: Dict[int, PropulsionSubsystem] = {
            i: PropulsionSubsystem() for i in range(swarm.num_cubes)
        }
        
        # Time tracking for propagation
        self._current_time: float = 0.0  # seconds
    
    # -------------------------------------------------------------------------
    # Group management
    # -------------------------------------------------------------------------
    
    def _create_group(self, cube_ids: Set[int], 
                      position: np.ndarray, 
                      velocity: np.ndarray) -> int:
        """
        Create a new group with the specified cubes.
        
        Returns the new group ID.
        """
        group_id = self._next_group_id
        self._next_group_id += 1
        
        # Calculate aggregate propulsion from member cubes
        total_delta_v = sum(
            self._cube_propulsion[cid].remaining_delta_v 
            for cid in cube_ids
            if cid in self._cube_propulsion
        ) if hasattr(self, '_cube_propulsion') else 50.0 * len(cube_ids)
        
        group = GroupState(
            group_id=group_id,
            cube_ids=cube_ids,
            position=position.copy(),
            velocity=velocity.copy(),
            propulsion=PropulsionSubsystem(
                max_delta_v=total_delta_v,
                remaining_delta_v=total_delta_v,
            )
        )
        
        self._groups[group_id] = group
        
        for cube_id in cube_ids:
            self._cube_to_group[cube_id] = group_id
        
        return group_id
    
    def _remove_group(self, group_id: int) -> None:
        """Remove a group (after merging into another)."""
        if group_id in self._groups:
            group = self._groups[group_id]
            for cube_id in group.cube_ids:
                if self._cube_to_group.get(cube_id) == group_id:
                    del self._cube_to_group[cube_id]
            del self._groups[group_id]
    
    def get_group(self, group_id: int) -> Optional[GroupState]:
        """Get a group by ID."""
        return self._groups.get(group_id)
    
    def get_group_for_cube(self, cube_id: int) -> Optional[GroupState]:
        """Get the group containing a cube."""
        group_id = self._cube_to_group.get(cube_id)
        if group_id is not None:
            return self._groups.get(group_id)
        return None
    
    def get_all_groups(self) -> List[GroupState]:
        """Get all groups."""
        return list(self._groups.values())
    
    def get_num_groups(self) -> int:
        """Get number of separate groups."""
        return len(self._groups)
    
    def get_cubes_in_group(self, group_id: int) -> Set[int]:
        """Get the set of cube IDs in a group."""
        group = self._groups.get(group_id)
        if group is None:
            return set()
        return group.cube_ids.copy()
    
    def is_single_group(self) -> bool:
        """Check if all cubes are in a single group (not separated)."""
        return len(self._groups) == 1
    
    # -------------------------------------------------------------------------
    # Separation operations
    # -------------------------------------------------------------------------
    
    def can_separate(self, cube_ids_to_separate: Set[int], 
                     separation_direction: Optional[np.ndarray] = None) -> Tuple[bool, str]:
        """
        Check if a set of cubes can separate from their current group.
        
        Args:
            cube_ids_to_separate: Cubes that would form the new group
            separation_direction: Direction of separation (optional)
            
        Returns:
            (can_separate, reason)
        """
        if not cube_ids_to_separate:
            return False, "No cubes specified for separation"
        
        # Check max groups
        if len(self._groups) >= self.separation_reqs.max_groups:
            return False, f"Maximum number of groups ({self.separation_reqs.max_groups}) reached"
        
        # Check min group size
        if len(cube_ids_to_separate) < self.separation_reqs.min_group_size:
            return False, f"Group too small (min {self.separation_reqs.min_group_size})"
        
        if not self.separation_reqs.allow_single_cube_groups and len(cube_ids_to_separate) == 1:
            return False, "Single-cube groups not allowed"
        
        # Check all cubes are in the same group
        source_group_ids = set(
            self._cube_to_group.get(cid) for cid in cube_ids_to_separate
        )
        source_group_ids.discard(None)
        
        if len(source_group_ids) == 0:
            return False, "Cubes not assigned to any group"
        
        if len(source_group_ids) > 1:
            return False, "Cubes are in different groups (must be in same group to separate)"
        
        source_group_id = list(source_group_ids)[0]
        source_group = self._groups[source_group_id]
        
        # Check the remaining group would meet min size
        remaining = source_group.cube_ids - cube_ids_to_separate
        if len(remaining) == 0:
            return False, "Cannot separate all cubes from a group (would leave empty group)"
        
        if len(remaining) > 0 and len(remaining) < self.separation_reqs.min_group_size:
            return False, f"Remaining group would be too small (min {self.separation_reqs.min_group_size})"
        
        # Check propulsion capability
        # Need enough delta-v for separation maneuver
        separating_cubes_delta_v = sum(
            self._cube_propulsion[cid].remaining_delta_v
            for cid in cube_ids_to_separate
        )
        
        if separating_cubes_delta_v < self.separation_reqs.min_separation_delta_v:
            return False, f"Insufficient delta-v for separation (need {self.separation_reqs.min_separation_delta_v} m/s)"
        
        # Check connectivity in the swarm
        # The cubes being separated must form a connected component
        # and the remaining cubes must also form a connected component
        
        # Get the swarm's connection graph
        if not self._check_separation_connectivity(cube_ids_to_separate, source_group.cube_ids):
            return False, "Separation would create disconnected cubes within a group"
        
        return True, "Separation possible"
    
    def _check_separation_connectivity(self, separating: Set[int], 
                                       full_group: Set[int]) -> bool:
        """
        Check that both the separating cubes and remaining cubes form 
        connected components.
        """
        remaining = full_group - separating
        
        # If either set is empty, that's fine
        if len(separating) == 0 or len(remaining) == 0:
            return True
        
        # Check separating cubes are connected
        if len(separating) > 1:
            if not self._is_cube_set_connected(separating):
                return False
        
        # Check remaining cubes are connected
        if len(remaining) > 1:
            if not self._is_cube_set_connected(remaining):
                return False
        
        return True
    
    def _is_cube_set_connected(self, cube_ids: Set[int]) -> bool:
        """Check if a set of cubes forms a connected component."""
        if len(cube_ids) <= 1:
            return True
        
        # BFS from first cube
        start = next(iter(cube_ids))
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            
            # Get connected neighbors from swarm
            neighbors = self.swarm._connections.get_connected_cubes(current)
            
            for neighbor in neighbors:
                if neighbor in cube_ids and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return visited == cube_ids
    
    def separate(self, cube_ids_to_separate: Set[int],
                 separation_direction: Optional[np.ndarray] = None,
                 separation_velocity: Optional[float] = None) -> Tuple[bool, str, Optional[int]]:
        """
        Separate a set of cubes into a new group.
        
        Args:
            cube_ids_to_separate: Cubes to separate into new group
            separation_direction: Unit vector for separation direction
                                 (defaults to +X if not specified)
            separation_velocity: Separation velocity in m/s
                                (defaults to default_separation_velocity)
        
        Returns:
            (success, message, new_group_id or None)
        """
        # Check if separation is possible
        can_sep, reason = self.can_separate(cube_ids_to_separate, separation_direction)
        if not can_sep:
            return False, reason, None
        
        # Get source group
        source_group_id = self._cube_to_group[next(iter(cube_ids_to_separate))]
        source_group = self._groups[source_group_id]
        
        # Default separation parameters
        if separation_direction is None:
            separation_direction = np.array([1.0, 0.0, 0.0])
        else:
            separation_direction = np.array(separation_direction, dtype=float)
            norm = np.linalg.norm(separation_direction)
            if norm > 1e-10:
                separation_direction = separation_direction / norm
            else:
                separation_direction = np.array([1.0, 0.0, 0.0])
        
        if separation_velocity is None:
            separation_velocity = self.separation_reqs.default_separation_velocity
        
        # Calculate delta-v cost
        # Both groups need to spend delta-v to achieve separation
        # Simplified: assume equal mass, each spends half the separation velocity
        delta_v_each = separation_velocity / 2.0
        
        # Consume delta-v from separating cubes
        for cid in cube_ids_to_separate:
            self._cube_propulsion[cid].perform_maneuver(
                delta_v_each / len(cube_ids_to_separate)
            )
        
        # Consume delta-v from remaining cubes
        remaining_cubes = source_group.cube_ids - cube_ids_to_separate
        if remaining_cubes:
            for cid in remaining_cubes:
                self._cube_propulsion[cid].perform_maneuver(
                    delta_v_each / len(remaining_cubes)
                )
        
        # Break connections between separating and remaining cubes
        self._break_inter_group_connections(cube_ids_to_separate, remaining_cubes)
        
        # Calculate new group position and velocity
        # New group moves in separation direction
        new_position = source_group.position.copy()  # Start at same position
        new_velocity = source_group.velocity + separation_direction * (separation_velocity / 2.0)
        
        # Source group gets opposite velocity change
        source_group.velocity = source_group.velocity - separation_direction * (separation_velocity / 2.0)
        
        # Update source group's cube set
        source_group.cube_ids = remaining_cubes
        
        # Recalculate source group propulsion
        if remaining_cubes:
            source_group.propulsion.remaining_delta_v = sum(
                self._cube_propulsion[cid].remaining_delta_v
                for cid in remaining_cubes
            )
        
        # Create new group if there are cubes to separate
        # (If remaining is empty, we effectively just renamed the group)
        if remaining_cubes:
            new_group_id = self._create_group(
                cube_ids_to_separate, new_position, new_velocity
            )
        else:
            # All cubes moved to new group, remove empty source group
            self._remove_group(source_group_id)
            new_group_id = self._create_group(
                cube_ids_to_separate, new_position, new_velocity
            )
        
        return True, "Separation successful", new_group_id
    
    def _break_inter_group_connections(self, group_a: Set[int], group_b: Set[int]) -> int:
        """
        Break all connections between cubes in different groups.
        
        Returns number of connections broken.
        """
        connections_broken = 0
        
        for cube_a in group_a:
            connected = self.swarm._connections.get_connected_cubes(cube_a)
            # Create a copy to avoid modifying during iteration
            connected_list = list(connected)
            for cube_b in connected_list:
                if cube_b in group_b:
                    success = self.swarm.disconnect_cubes(cube_a, cube_b)
                    if success:
                        connections_broken += 1
        
        return connections_broken
    
    # -------------------------------------------------------------------------
    # Docking/Rejoining operations
    # -------------------------------------------------------------------------
    
    def can_dock(self, group_a_id: int, group_b_id: int) -> Tuple[bool, str]:
        """
        Check if two groups can dock together.
        
        Args:
            group_a_id: First group
            group_b_id: Second group
            
        Returns:
            (can_dock, reason)
        """
        group_a = self._groups.get(group_a_id)
        group_b = self._groups.get(group_b_id)
        
        if group_a is None:
            return False, f"Group {group_a_id} does not exist"
        if group_b is None:
            return False, f"Group {group_b_id} does not exist"
        if group_a_id == group_b_id:
            return False, "Cannot dock a group with itself"
        
        # Check distance
        distance = group_a.get_distance_to(group_b)
        if distance > self.docking_reqs.max_docking_range:
            return False, f"Groups too far apart ({distance:.1f}m > {self.docking_reqs.max_docking_range}m)"
        
        # Check relative velocity
        rel_speed = group_a.get_relative_speed(group_b)
        if rel_speed > self.docking_reqs.max_docking_velocity:
            return False, f"Relative velocity too high ({rel_speed:.3f} m/s > {self.docking_reqs.max_docking_velocity} m/s)"
        
        # Check propulsion for docking maneuver
        total_delta_v = (group_a.propulsion.remaining_delta_v + 
                        group_b.propulsion.remaining_delta_v)
        
        if total_delta_v < self.docking_reqs.docking_delta_v_cost:
            return False, f"Insufficient delta-v for docking maneuver"
        
        # Check communication
        if not self.comm_reqs.can_coordinate(distance):
            return False, "Groups cannot coordinate at this distance"
        
        return True, "Docking possible"
    
    def initiate_docking_approach(self, group_a_id: int, group_b_id: int,
                                approach_velocity: float = 0.5) -> Tuple[bool, str]:
        """
        Initiate a docking approach between two groups.
        
        This sets up the relative velocity for groups to approach each other.
        Actual docking happens when they're close enough and slow enough.
        
        Args:
            group_a_id: First group (will maneuver to approach group_b)
            group_b_id: Second group (target, stays on current trajectory)
            approach_velocity: Desired closing velocity (m/s), positive means closing
            
        Returns:
            (success, message)
        """
        group_a = self._groups.get(group_a_id)
        group_b = self._groups.get(group_b_id)
        
        if group_a is None or group_b is None:
            return False, "One or both groups do not exist"
        
        # Calculate direction from A to B (unit vector)
        relative_position = group_b.position - group_a.position
        distance = np.linalg.norm(relative_position)
        
        if distance < 1e-6:
            return False, "Groups are already at same position"
        
        direction_a_to_b = relative_position / distance
        
        # Calculate current relative velocity of A with respect to B
        # Positive component along direction_a_to_b means A is moving toward B
        relative_velocity = group_a.velocity - group_b.velocity
        current_closing_rate = np.dot(relative_velocity, direction_a_to_b)
        
        # current_closing_rate > 0 means A is moving toward B
        # current_closing_rate < 0 means A is moving away from B
        
        # We want the closing rate to be +approach_velocity (moving toward B)
        # Required change in velocity component along direction
        delta_v_needed = approach_velocity - current_closing_rate
        
        # If already closing faster than desired, no maneuver needed
        if delta_v_needed <= 0:
            return True, f"Already approaching at {current_closing_rate:.3f} m/s (>= {approach_velocity} m/s)"
        
        # The delta-v vector to apply to group A
        delta_v_vector = direction_a_to_b * delta_v_needed
        delta_v_magnitude = abs(delta_v_needed)
        
        # Check propulsion
        if not group_a.propulsion.can_perform_maneuver(delta_v_magnitude):
            return False, f"Insufficient delta-v (need {delta_v_magnitude:.3f} m/s, have {group_a.propulsion.remaining_delta_v:.3f} m/s)"
        
        # Perform maneuver on group A
        group_a.propulsion.perform_maneuver(delta_v_magnitude)
        
        # Update propulsion in member cubes
        delta_v_per_cube = delta_v_magnitude / len(group_a.cube_ids)
        for cid in group_a.cube_ids:
            self._cube_propulsion[cid].perform_maneuver(delta_v_per_cube)
        
        # Update velocity - add the delta-v vector
        group_a.velocity = group_a.velocity + delta_v_vector
        
        # Verify the new closing rate
        new_relative_velocity = group_a.velocity - group_b.velocity
        new_closing_rate = np.dot(new_relative_velocity, direction_a_to_b)
        
        return True, f"Approach initiated: closing at {new_closing_rate:.3f} m/s, distance {distance:.1f}m"
    
    def dock(self, group_a_id: int, group_b_id: int,
             docking_configuration: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Dock two groups together, merging them into one.
        
        Args:
            group_a_id: First group
            group_b_id: Second group
            docking_configuration: Optional dict specifying how cubes should
                                  be positioned relative to each other
        
        Returns:
            (success, message)
        """
        # Check if docking is possible
        can_dock_result, reason = self.can_dock(group_a_id, group_b_id)
        if not can_dock_result:
            return False, reason
        
        group_a = self._groups[group_a_id]
        group_b = self._groups[group_b_id]
        
        # Consume delta-v for docking
        docking_cost = self.docking_reqs.docking_delta_v_cost
        cost_per_group = docking_cost / 2.0
        
        group_a.propulsion.perform_maneuver(cost_per_group)
        group_b.propulsion.perform_maneuver(cost_per_group)
        
        # Update per-cube propulsion
        for cid in group_a.cube_ids:
            self._cube_propulsion[cid].perform_maneuver(
                cost_per_group / len(group_a.cube_ids)
            )
        for cid in group_b.cube_ids:
            self._cube_propulsion[cid].perform_maneuver(
                cost_per_group / len(group_b.cube_ids)
            )
        
        # Merge groups
        # Group B cubes will be repositioned adjacent to Group A
        merged_cubes = group_a.cube_ids | group_b.cube_ids
        
        # Calculate merged velocity (momentum conservation)
        total_mass = len(merged_cubes)  # Assuming equal mass per cube
        merged_velocity = (
            group_a.velocity * len(group_a.cube_ids) +
            group_b.velocity * len(group_b.cube_ids)
        ) / total_mass
        
        # Use Group A's position
        merged_position = group_a.position.copy()
        
        # Reposition Group B cubes in the swarm grid
        success = self._reposition_cubes_for_docking(group_a, group_b, docking_configuration)
        
        if not success:
            return False, "Failed to find valid docking configuration"
        
        # Update group A with merged data
        group_a.cube_ids = merged_cubes
        group_a.velocity = merged_velocity
        group_a.propulsion.remaining_delta_v = sum(
            self._cube_propulsion[cid].remaining_delta_v
            for cid in merged_cubes
        )
        
        # Update cube-to-group mapping
        for cid in group_b.cube_ids:
            self._cube_to_group[cid] = group_a_id
        
        # Remove group B
        self._remove_group(group_b_id)
        
        # Auto-connect adjacent cubes
        self.swarm.auto_connect_all()
        
        return True, "Docking successful"
    
    def _reposition_cubes_for_docking(self, group_a: GroupState, 
                                    group_b: GroupState,
                                    configuration: Optional[Dict],
                                    preserve_orientations: bool = True) -> bool:
        """
        Reposition cubes from group B to dock with group A.
        
        Args:
            group_a: Target group
            group_b: Group being docked
            configuration: Optional docking configuration
            preserve_orientations: If True, try to maintain cube orientations
            
        Returns True if successful.
        """
        # Get boundary positions of group A (positions with empty neighbors)
        group_a_positions = set()
        for cid in group_a.cube_ids:
            cube = self.swarm.get_cube(cid)
            if cube:
                group_a_positions.add(cube.position)
        
        # Find empty positions adjacent to group A
        available_positions = []
        for pos in group_a_positions:
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                adj_pos = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                if self.swarm._grid.is_empty(adj_pos):
                    available_positions.append(adj_pos)
        
        # Remove duplicates and sort for determinism
        available_positions = sorted(set(available_positions))
        
        if len(available_positions) < len(group_b.cube_ids):
            return False  # Not enough space
        
        group_b_cubes = list(group_b.cube_ids)
        
        # Store original state for rollback
        old_positions = {}
        old_orientations = {}
        
        for cid in group_b_cubes:
            cube = self.swarm.get_cube(cid)
            if cube:
                old_positions[cid] = cube.position
                old_orientations[cid] = cube.orientation.copy()
                self.swarm._grid.remove_cube(cid)
        
        # Calculate centroid and relative positions of group B
        if old_positions:
            group_b_positions_list = list(old_positions.values())
            centroid_b = np.mean(group_b_positions_list, axis=0)
            
            relative_positions = {
                cid: np.array(pos) - centroid_b 
                for cid, pos in old_positions.items()
            }
        else:
            relative_positions = {}
        
        # Find centroid of available positions
        if not available_positions:
            self._restore_cube_positions(group_b_cubes, old_positions, old_orientations)
            return False
        
        available_centroid = np.mean(available_positions, axis=0)
        
        # Try multiple placement strategies
        placement_map = self._try_shape_preserving_placement(
            group_b_cubes, relative_positions, available_positions, 
            group_a_positions, available_centroid
        )
        
        if not placement_map:
            # Fallback to greedy placement
            placement_map = self._try_greedy_placement(
                group_b_cubes, available_positions, group_a_positions
            )
        
        if not placement_map:
            # Restore and fail
            self._restore_cube_positions(group_b_cubes, old_positions, old_orientations)
            return False
        
        # Apply the placement
        for cid, new_pos in placement_map.items():
            self.swarm._grid.place_cube(cid, new_pos)
            cube = self.swarm.get_cube(cid)
            if cube:
                cube.position = new_pos
                if preserve_orientations and cid in old_orientations:
                    cube.orientation = old_orientations[cid]
        
        return True

    def _restore_cube_positions(self, cube_ids: List[int], 
                                positions: Dict[int, tuple],
                                orientations: Dict[int, 'Orientation']) -> None:
        """Restore cubes to their original positions."""
        for cid in cube_ids:
            if cid in positions:
                self.swarm._grid.place_cube(cid, positions[cid])
                cube = self.swarm.get_cube(cid)
                if cube and cid in orientations:
                    cube.orientation = orientations[cid]

    def _try_shape_preserving_placement(self, cube_ids: List[int],
                                        relative_positions: Dict[int, np.ndarray],
                                        available_positions: List[tuple],
                                        occupied_positions: Set[tuple],
                                        target_centroid: np.ndarray) -> Optional[Dict[int, tuple]]:
        """Try to place cubes while preserving their relative shape."""
        for base_pos in available_positions[:10]:
            base_pos = np.array(base_pos)
            
            test_positions = {}
            valid = True
            
            for cid in cube_ids:
                if cid not in relative_positions:
                    valid = False
                    break
                    
                target = base_pos + relative_positions[cid]
                target_int = tuple(int(round(x)) for x in target)
                
                if (target_int in occupied_positions or 
                    target_int in test_positions.values() or
                    self.swarm._grid.is_occupied(target_int)):
                    valid = False
                    break
                
                test_positions[cid] = target_int
            
            if valid and len(test_positions) == len(cube_ids):
                # Verify adjacency to group A
                if self._has_adjacency_to_group(test_positions, occupied_positions):
                    return test_positions
        
        return None

    def _try_greedy_placement(self, cube_ids: List[int],
                            available_positions: List[tuple],
                            occupied_positions: Set[tuple]) -> Optional[Dict[int, tuple]]:
        """Greedy placement fallback."""
        placement_map = {}
        used_positions = set()
        
        for cid in cube_ids:
            placed = False
            for pos in available_positions:
                if pos not in used_positions:
                    if self.swarm._grid.is_empty(pos):
                        placement_map[cid] = pos
                        used_positions.add(pos)
                        placed = True
                        break
            
            if not placed:
                return None
        
        if self._has_adjacency_to_group(placement_map, occupied_positions):
            return placement_map
        return None

    def _has_adjacency_to_group(self, placement: Dict[int, tuple], 
                                group_positions: Set[tuple]) -> bool:
        """Check if at least one placed cube is adjacent to the group."""
        for pos in placement.values():
            for dx, dy, dz in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                adj = (pos[0] + dx, pos[1] + dy, pos[2] + dz)
                if adj in group_positions:
                    return True
        return False
    
    # -------------------------------------------------------------------------
    # Time propagation and orbital mechanics (simplified)
    # -------------------------------------------------------------------------
    
    def propagate(self, time_step: float) -> None:
        """
        Propagate all group positions forward in time.
        
        This is a simplified linear propagation. For more realistic
        orbital mechanics, you would integrate the equations of motion
        considering gravitational forces.
        
        Args:
            time_step: Time to propagate in seconds
        """
        self._current_time += time_step
        
        for group in self._groups.values():
            # Simple linear propagation: position += velocity * time
            group.position = group.position + group.velocity * time_step
    
    def propagate_to_time(self, target_time: float) -> None:
        """
        Propagate to a specific time.
        
        Args:
            target_time: Target time in seconds
        """
        if target_time > self._current_time:
            self.propagate(target_time - self._current_time)
    
    def get_time(self) -> float:
        """Get current simulation time in seconds."""
        return self._current_time
    
    def apply_delta_v(self, group_id: int, 
                      delta_v: np.ndarray) -> Tuple[bool, str]:
        """
        Apply a delta-v maneuver to a group.
        
        Args:
            group_id: The group to maneuver
            delta_v: Velocity change vector (m/s)
            
        Returns:
            (success, message)
        """
        group = self._groups.get(group_id)
        if group is None:
            return False, f"Group {group_id} does not exist"
        
        delta_v = np.array(delta_v, dtype=float)
        delta_v_magnitude = float(np.linalg.norm(delta_v))
        
        if delta_v_magnitude < 1e-10:
            return True, "No maneuver needed"
        
        # Check propulsion
        if not group.propulsion.can_perform_maneuver(delta_v_magnitude):
            return False, f"Insufficient delta-v (need {delta_v_magnitude:.3f} m/s, have {group.propulsion.remaining_delta_v:.3f} m/s)"
        
        # Perform maneuver
        group.propulsion.perform_maneuver(delta_v_magnitude)
        
        # Update per-cube propulsion
        dv_per_cube = delta_v_magnitude / len(group.cube_ids)
        for cid in group.cube_ids:
            self._cube_propulsion[cid].perform_maneuver(dv_per_cube)
        
        # Update velocity
        group.velocity = group.velocity + delta_v
        
        return True, f"Maneuver applied: delta-v = {delta_v_magnitude:.3f} m/s"
    
    def station_keep(self, group_id: int, 
                     target_position: np.ndarray,
                     tolerance: float = 10.0) -> Tuple[bool, str]:
        """
        Perform station-keeping to maintain position near a target.
        
        Args:
            group_id: The group to station-keep
            target_position: Desired position (m)
            tolerance: Acceptable position error (m)
            
        Returns:
            (success, message)
        """
        group = self._groups.get(group_id)
        if group is None:
            return False, f"Group {group_id} does not exist"
        
        target_position = np.array(target_position, dtype=float)
        
        # Calculate position error
        error = target_position - group.position
        error_magnitude = np.linalg.norm(error)
        
        if error_magnitude <= tolerance:
            return True, "Already within tolerance"
        
        # Calculate required velocity change
        # Simplified: we'll do a two-impulse transfer
        # First impulse: start moving toward target
        # Second impulse: stop at target (would need to call again)
        
        # For now, calculate delta-v to cancel current velocity 
        # and add velocity toward target
        
        # Time to reach target at a reasonable speed
        approach_speed = 1.0  # m/s
        time_to_target = error_magnitude / approach_speed
        
        # Desired velocity
        desired_velocity = error / max(time_to_target, 1.0)
        
        # Delta-v needed
        delta_v = desired_velocity - group.velocity
        
        return self.apply_delta_v(group_id, delta_v)
    
    # -------------------------------------------------------------------------
    # Communication and coordination
    # -------------------------------------------------------------------------
    
    def get_inter_group_distances(self) -> Dict[Tuple[int, int], float]:
        """
        Get distances between all pairs of groups.
        
        Returns:
            Dict mapping (group_a_id, group_b_id) -> distance in meters
        """
        distances = {}
        group_ids = list(self._groups.keys())
        
        for i, ga_id in enumerate(group_ids):
            for gb_id in group_ids[i+1:]:
                ga = self._groups[ga_id]
                gb = self._groups[gb_id]
                dist = ga.get_distance_to(gb)
                distances[(ga_id, gb_id)] = dist
                distances[(gb_id, ga_id)] = dist
        
        return distances
    
    def get_communication_links(self) -> Dict[Tuple[int, int], Dict]:
        """
        Get communication status between all group pairs.
        
        Returns:
            Dict mapping (group_a_id, group_b_id) -> link info dict
        """
        links = {}
        distances = self.get_inter_group_distances()
        
        for (ga_id, gb_id), distance in distances.items():
            if ga_id >= gb_id:
                continue  # Only store each pair once
            
            can_comm = self.comm_reqs.can_communicate(distance)
            can_coord = self.comm_reqs.can_coordinate(distance)
            data_rate = self.comm_reqs.get_data_rate(distance)
            
            links[(ga_id, gb_id)] = {
                'distance': distance,
                'can_communicate': can_comm,
                'can_coordinate': can_coord,
                'data_rate': data_rate,
            }
        
        return links
    
    def get_communication_graph(self) -> Dict[int, Set[int]]:
        """
        Get the communication topology as an adjacency graph.
        
        Returns:
            Dict mapping group_id -> set of group_ids it can communicate with
        """
        graph = {gid: set() for gid in self._groups.keys()}
        
        links = self.get_communication_links()
        for (ga_id, gb_id), info in links.items():
            if info['can_communicate']:
                graph[ga_id].add(gb_id)
                graph[gb_id].add(ga_id)
        
        return graph
    
    def is_constellation_connected(self) -> bool:
        """
        Check if all groups can communicate (directly or through relay).
        """
        if len(self._groups) <= 1:
            return True
        
        graph = self.get_communication_graph()
        
        # BFS from first group
        start = next(iter(self._groups.keys()))
        visited = {start}
        queue = [start]
        
        while queue:
            current = queue.pop(0)
            for neighbor in graph.get(current, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        
        return len(visited) == len(self._groups)
    
    def get_max_baseline(self) -> float:
        """
        Get the maximum distance between any two groups.
        
        This is useful for sparse aperture / interferometry calculations.
        """
        if len(self._groups) <= 1:
            return 0.0
        
        max_dist = 0.0
        distances = self.get_inter_group_distances()
        
        for dist in distances.values():
            max_dist = max(max_dist, dist)
        
        return max_dist
    
    def get_total_delta_v_remaining(self) -> float:
        """Get total delta-v remaining across all cubes."""
        return sum(p.remaining_delta_v for p in self._cube_propulsion.values())
    
    def get_delta_v_by_group(self) -> Dict[int, float]:
        """Get remaining delta-v for each group."""
        return {
            gid: group.propulsion.remaining_delta_v
            for gid, group in self._groups.items()
        }
    
    # -------------------------------------------------------------------------
    # State and metrics
    # -------------------------------------------------------------------------
    
    def get_constellation_state(self) -> Dict:
        """
        Get complete state of the constellation.
        
        Useful for observation in RL environment.
        """
        state = {
            'time': self._current_time,
            'num_groups': len(self._groups),
            'total_cubes': self.swarm.num_cubes,
            'is_connected': self.is_constellation_connected(),
            'max_baseline': self.get_max_baseline(),
            'total_delta_v': self.get_total_delta_v_remaining(),
            'groups': {},
        }
        
        for gid, group in self._groups.items():
            state['groups'][gid] = {
                'num_cubes': len(group.cube_ids),
                'cube_ids': list(group.cube_ids),
                'position': group.position.tolist(),
                'velocity': group.velocity.tolist(),
                'delta_v_remaining': group.propulsion.remaining_delta_v,
            }
        
        return state
    
    def get_observation_vector(self) -> np.ndarray:
        """
        Get a fixed-size observation vector for RL.
        
        Returns array containing:
        - Number of groups (1)
        - Per-group info: position (3), velocity (3), num_cubes (1), delta_v (1) = 8 per group
        - Padded to max_groups
        """
        max_groups = self.separation_reqs.max_groups
        obs_per_group = 8  # pos(3) + vel(3) + num_cubes(1) + delta_v(1)
        
        obs = np.zeros(1 + max_groups * obs_per_group, dtype=np.float32)
        
        obs[0] = len(self._groups)
        
        for i, (gid, group) in enumerate(self._groups.items()):
            if i >= max_groups:
                break
            
            offset = 1 + i * obs_per_group
            obs[offset:offset+3] = group.position / 1000.0  # Normalize to km
            obs[offset+3:offset+6] = group.velocity  # m/s
            obs[offset+6] = len(group.cube_ids) / self.swarm.num_cubes  # Fraction
            obs[offset+7] = group.propulsion.remaining_delta_v / 50.0  # Normalized
        
        return obs
    
    def copy(self) -> 'Constellation':
        """Create a deep copy of the constellation."""
        new_constellation = Constellation(
            self.swarm.copy(),
            SeparationRequirements(
                min_separation_delta_v=self.separation_reqs.min_separation_delta_v,
                default_separation_velocity=self.separation_reqs.default_separation_velocity,
                min_group_size=self.separation_reqs.min_group_size,
                max_groups=self.separation_reqs.max_groups,
                allow_single_cube_groups=self.separation_reqs.allow_single_cube_groups,
            ),
            DockingRequirements(
                max_docking_velocity=self.docking_reqs.max_docking_velocity,
                max_docking_range=self.docking_reqs.max_docking_range,
                docking_delta_v_cost=self.docking_reqs.docking_delta_v_cost,
                max_alignment_error=self.docking_reqs.max_alignment_error,
                docking_duration=self.docking_reqs.docking_duration,
            ),
            CommunicationRequirements(
                max_isl_range=self.comm_reqs.max_isl_range,
                base_data_rate=self.comm_reqs.base_data_rate,
                reference_distance=self.comm_reqs.reference_distance,
                min_coordination_rate=self.comm_reqs.min_coordination_rate,
            ),
        )
        
        # Copy groups
        new_constellation._groups = {
            gid: group.copy() for gid, group in self._groups.items()
        }
        
        # Copy cube-to-group mapping
        new_constellation._cube_to_group = self._cube_to_group.copy()
        
        # Copy next group ID
        new_constellation._next_group_id = self._next_group_id
        
        # Copy per-cube propulsion
        new_constellation._cube_propulsion = {
            cid: prop.copy() for cid, prop in self._cube_propulsion.items()
        }
        
        # Copy time
        new_constellation._current_time = self._current_time
        
        return new_constellation
    
    def __repr__(self) -> str:
        return (f"Constellation({self.swarm.num_cubes} cubes, "
                f"{len(self._groups)} groups, "
                f"max_baseline={self.get_max_baseline():.1f}m)")