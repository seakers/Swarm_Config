import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Set

from core.constellation import Constellation
from rewards.constellation_metrics import ConstellationMetrics


class ConstellationTask(ABC):
    """
    Abstract base class for constellation-level tasks.
    
    These tasks involve separation, formation flying, and docking.
    """
    
    @abstractmethod
    def compute_reward(self, constellation: Constellation, 
                       action_taken: Optional[any] = None) -> float:
        """Compute reward for current constellation state."""
        pass
    
    @abstractmethod
    def is_complete(self, constellation: Constellation) -> bool:
        """Check if task is complete."""
        pass
    
    @abstractmethod
    def is_failed(self, constellation: Constellation) -> bool:
        """Check if task has failed."""
        pass
    
    @abstractmethod
    def get_task_info(self) -> Dict[str, any]:
        """Get task information."""
        pass
    
    def get_progress(self, constellation: Constellation) -> float:
        """Get progress toward completion (0 to 1)."""
        return 1.0 if self.is_complete(constellation) else 0.0


class FormConstellationTask(ConstellationTask):
    """
    Task: Separate into a specific number of groups with target baseline.
    
    Useful for sparse aperture imaging or distributed sensing.
    """
    
    def __init__(self,
                 target_num_groups: int = 2,
                 target_baseline: float = 10000.0,  # meters
                 baseline_tolerance: float = 1000.0,
                 min_group_size: int = 8,
                 maintain_communication: bool = True):
        """
        Args:
            target_num_groups: Desired number of separate groups
            target_baseline: Desired maximum distance between groups (m)
            baseline_tolerance: Acceptable deviation from target (m)
            min_group_size: Minimum cubes per group
            maintain_communication: Require all groups stay in communication
        """
        self.target_num_groups = target_num_groups
        self.target_baseline = target_baseline
        self.baseline_tolerance = baseline_tolerance
        self.min_group_size = min_group_size
        self.maintain_communication = maintain_communication
    
    def compute_reward(self, constellation: Constellation,
                       action_taken: Optional[any] = None) -> float:
        metrics = ConstellationMetrics(constellation)
        
        # Component 1: Number of groups
        num_groups = constellation.get_num_groups()
        if num_groups == self.target_num_groups:
            group_score = 1.0
        elif num_groups < self.target_num_groups:
            group_score = num_groups / self.target_num_groups
        else:
            # Penalty for too many groups
            group_score = self.target_num_groups / num_groups
        
        # Component 2: Baseline distance
        baseline = constellation.get_max_baseline()
        baseline_error = abs(baseline - self.target_baseline)
        
        if baseline_error <= self.baseline_tolerance:
            baseline_score = 1.0
        else:
            baseline_score = max(0, 1.0 - baseline_error / self.target_baseline)
        
        # Component 3: Group sizes
        groups = constellation.get_all_groups()
        if groups:
            min_size = min(len(g.cube_ids) for g in groups)
            if min_size >= self.min_group_size:
                size_score = 1.0
            else:
                size_score = min_size / self.min_group_size
        else:
            size_score = 0.0
        
        # Component 4: Communication
        if self.maintain_communication:
            if constellation.is_constellation_connected():
                comm_score = 1.0
            else:
                comm_score = 0.0
        else:
            comm_score = 1.0
        
        # Component 5: Propulsion efficiency (don't waste delta-v)
        prop_score = metrics.get_propulsion_efficiency()
        
        # Combine scores
        reward = (
            0.3 * group_score +
            0.3 * baseline_score +
            0.15 * size_score +
            0.15 * comm_score +
            0.1 * prop_score
        )
        
        # Bonus for completion
        if self.is_complete(constellation):
            reward += 1.0
        
        return reward
    
    def is_complete(self, constellation: Constellation) -> bool:
        # Check number of groups
        if constellation.get_num_groups() != self.target_num_groups:
            return False
        
        # Check baseline
        baseline = constellation.get_max_baseline()
        if abs(baseline - self.target_baseline) > self.baseline_tolerance:
            return False
        
        # Check group sizes
        for group in constellation.get_all_groups():
            if len(group.cube_ids) < self.min_group_size:
                return False
        
        # Check communication
        if self.maintain_communication and not constellation.is_constellation_connected():
            return False
        
        return True
    
    def is_failed(self, constellation: Constellation) -> bool:
        # Failed if we can't possibly reach target due to propulsion limits
        total_delta_v = constellation.get_total_delta_v_remaining()
        
        # Rough estimate: need at least some delta-v per group for formation
        min_required = self.target_num_groups * 1.0  # 1 m/s per group minimum
        
        if total_delta_v < min_required and constellation.get_num_groups() < self.target_num_groups:
            return True
        
        # Failed if communication is required but permanently lost
        if self.maintain_communication:
            if not constellation.is_constellation_connected():
                # Check if any group is out of communication range with no propulsion
                for group in constellation.get_all_groups():
                    if group.propulsion.remaining_delta_v < 0.1:
                        # This group can't maneuver
                        # Check if it can communicate
                        can_comm_with_any = False
                        for other in constellation.get_all_groups():
                            if other.group_id != group.group_id:
                                dist = group.get_distance_to(other)
                                if constellation.comm_reqs.can_communicate(dist):
                                    can_comm_with_any = True
                                    break
                        
                        if not can_comm_with_any:
                            return True  # Isolated group with no propulsion
        
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'form_constellation',
            'target_num_groups': self.target_num_groups,
            'target_baseline': self.target_baseline,
            'baseline_tolerance': self.baseline_tolerance,
            'min_group_size': self.min_group_size,
            'maintain_communication': self.maintain_communication,
        }
    
    def get_progress(self, constellation: Constellation) -> float:
        # Progress based on how close we are to target configuration
        num_groups = constellation.get_num_groups()
        group_progress = min(1.0, num_groups / self.target_num_groups)
        
        baseline = constellation.get_max_baseline()
        if self.target_baseline > 0:
            baseline_progress = min(1.0, baseline / self.target_baseline)
        else:
            baseline_progress = 1.0
        
        return 0.5 * group_progress + 0.5 * baseline_progress


class RendezvousAndDockTask(ConstellationTask):
    """
    Task: Bring separated groups back together and dock.
    
    This is the reverse of FormConstellationTask.
    """
    
    def __init__(self,
                 target_num_groups: int = 1,
                 max_time: float = 3600.0,  # seconds
                 efficiency_weight: float = 0.3):
        """
        Args:
            target_num_groups: Desired final number of groups (usually 1)
            max_time: Maximum time allowed for rendezvous
            efficiency_weight: Weight for propulsion efficiency in reward
        """
        self.target_num_groups = target_num_groups
        self.max_time = max_time
        self.efficiency_weight = efficiency_weight
    
    def compute_reward(self, constellation: Constellation,
                       action_taken: Optional[any] = None) -> float:
        metrics = ConstellationMetrics(constellation)
        
        num_groups = constellation.get_num_groups()
        
        # Component 1: Number of groups (fewer is better)
        if num_groups <= self.target_num_groups:
            group_score = 1.0
        else:
            group_score = self.target_num_groups / num_groups
        
        # Component 2: Proximity of groups (closer is better for docking)
        if num_groups > 1:
            distances = constellation.get_inter_group_distances()
            if distances:
                min_dist = min(distances.values())
                max_docking_range = constellation.docking_reqs.max_docking_range
                
                if min_dist <= max_docking_range:
                    proximity_score = 1.0
                else:
                    # Score based on how close to docking range
                    proximity_score = max_docking_range / min_dist
            else:
                proximity_score = 1.0
        else:
            proximity_score = 1.0
        
        # Component 3: Relative velocities (lower is better for docking)
        if num_groups > 1:
            groups = constellation.get_all_groups()
            min_rel_speed = float('inf')
            
            for i, ga in enumerate(groups):
                for gb in groups[i+1:]:
                    rel_speed = ga.get_relative_speed(gb)
                    min_rel_speed = min(min_rel_speed, rel_speed)
            
            max_docking_vel = constellation.docking_reqs.max_docking_velocity
            if min_rel_speed <= max_docking_vel:
                velocity_score = 1.0
            else:
                velocity_score = max_docking_vel / min_rel_speed
        else:
            velocity_score = 1.0
        
        # Component 4: Propulsion efficiency
        prop_score = metrics.get_propulsion_efficiency()
        
        # Combine scores
        reward = (
            0.3 * group_score +
            0.25 * proximity_score +
            0.25 * velocity_score +
            self.efficiency_weight * prop_score
        )
        
        # Normalize weights
        total_weight = 0.3 + 0.25 + 0.25 + self.efficiency_weight
        reward = reward / total_weight
        
        # Bonus for completion
        if self.is_complete(constellation):
            reward += 1.0
        
        return reward
    
    def is_complete(self, constellation: Constellation) -> bool:
        return constellation.get_num_groups() <= self.target_num_groups
    
    def is_failed(self, constellation: Constellation) -> bool:
        # Failed if groups are too far apart and no propulsion to close
        if constellation.get_num_groups() <= self.target_num_groups:
            return False
        
        # Check if any pair of groups can potentially dock
        groups = constellation.get_all_groups()
        
        for i, ga in enumerate(groups):
            for gb in groups[i+1:]:
                dist = ga.get_distance_to(gb)
                rel_speed = ga.get_relative_speed(gb)
                
                # Estimate delta-v needed to rendezvous
                # Simplified: need to cancel relative velocity and then approach
                delta_v_needed = rel_speed + 0.5  # Plus some margin for approach
                
                total_dv = ga.propulsion.remaining_delta_v + gb.propulsion.remaining_delta_v
                
                if total_dv >= delta_v_needed:
                    return False  # At least this pair could potentially dock
        
        return True  # No pairs can dock
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'rendezvous_and_dock',
            'target_num_groups': self.target_num_groups,
            'max_time': self.max_time,
            'efficiency_weight': self.efficiency_weight,
        }
    
    def get_progress(self, constellation: Constellation) -> float:
        num_groups = constellation.get_num_groups()
        
        if num_groups <= self.target_num_groups:
            return 1.0
        
        # Progress based on number of groups reduced
        initial_max_groups = constellation.separation_reqs.max_groups
        groups_to_merge = num_groups - self.target_num_groups
        max_merges_needed = initial_max_groups - self.target_num_groups
        
        if max_merges_needed <= 0:
            return 1.0
        
        merges_done = initial_max_groups - num_groups
        return merges_done / max_merges_needed


class StereoImagingTask(ConstellationTask):
    """
    Task: Position groups for stereo imaging of a target.
    
    For stereo imaging, we want:
    - Two or more groups separated by a baseline
    - All groups' cameras pointing at the target
    - Specific angular separation for stereo effect
    """
    
    def __init__(self,
                 target_direction: Tuple[float, float, float] = (0, 1, 0),
                 num_viewpoints: int = 2,
                 min_angular_separation: float = 0.1,  # radians (~5.7 degrees)
                 max_angular_separation: float = 0.5,  # radians (~28.6 degrees)
                 target_distance: float = 1000.0):  # meters to target (for baseline calc)
        """
        Args:
            target_direction: Direction to the imaging target
            num_viewpoints: Number of separate viewing positions
            min_angular_separation: Minimum angle between viewpoints
            max_angular_separation: Maximum angle between viewpoints
            target_distance: Assumed distance to target for geometry
        """
        self.target_direction = np.array(target_direction, dtype=float)
        self.target_direction = self.target_direction / np.linalg.norm(self.target_direction)
        self.num_viewpoints = num_viewpoints
        self.min_angular_separation = min_angular_separation
        self.max_angular_separation = max_angular_separation
        self.target_distance = target_distance
        
        # Calculate required baseline from angular separation
        # baseline = 2 * distance * tan(angle/2)
        self.min_baseline = 2 * target_distance * np.tan(min_angular_separation / 2)
        self.max_baseline = 2 * target_distance * np.tan(max_angular_separation / 2)
    
    def compute_reward(self, constellation: Constellation,
                       action_taken: Optional[any] = None) -> float:
        num_groups = constellation.get_num_groups()
        
        # Component 1: Number of viewpoints
        if num_groups >= self.num_viewpoints:
            viewpoint_score = 1.0
        else:
            viewpoint_score = num_groups / self.num_viewpoints
        
        # Component 2: Baseline (angular separation)
        baseline = constellation.get_max_baseline()
        
        if self.min_baseline <= baseline <= self.max_baseline:
            baseline_score = 1.0
        elif baseline < self.min_baseline:
            baseline_score = baseline / self.min_baseline
        else:
            # Too far apart
            baseline_score = self.max_baseline / baseline
        
        # Component 3: Camera alignment with target
        # Check that cameras in each group can see the target
        alignment_scores = []
        
        from rewards.metrics import SwarmFaceAnalyzer
        from core.cube_faces import FaceFunction
        
        # for group in constellation.get_all_groups():
        #     # Create a temporary view of just this group's cubes
        #     # For simplicity, we'll check overall camera exposure
        #     analyzer = SwarmFaceAnalyzer(constellation.swarm)
            
        #     # Get cubes in this group with exposed cameras
        #     camera_alignment = analyzer.compute_function_alignment(
        #         FaceFunction.CAMERA,
        #         tuple(self.target_direction),
        #         only_exposed=True
        #     )
            
        #     alignment_scores.append(max(0, camera_alignment['mean']))
        
        # if alignment_scores:
        #     camera_score = np.mean(alignment_scores)
        # else:
        #     camera_score = 0.0
        analyzer = SwarmFaceAnalyzer(constellation.swarm)
        camera_alignment = analyzer.compute_function_alignment(
            FaceFunction.CAMERA,
            tuple(self.target_direction),
            only_exposed=True
        )
        camera_score = max(0, camera_alignment['mean'])


        # Component 4: Communication (groups need to share data)
        if constellation.is_constellation_connected():
            comm_score = 1.0
        else:
            comm_score = 0.0
        
        # Combine scores
        reward = (
            0.25 * viewpoint_score +
            0.3 * baseline_score +
            0.3 * camera_score +
            0.15 * comm_score
        )
        
        if self.is_complete(constellation):
            reward += 1.0
        
        return reward
    
    def is_complete(self, constellation: Constellation) -> bool:
        # Check viewpoints
        if constellation.get_num_groups() < self.num_viewpoints:
            return False
        
        # Check baseline
        baseline = constellation.get_max_baseline()
        if not (self.min_baseline <= baseline <= self.max_baseline):
            return False
        
        # Check communication
        if not constellation.is_constellation_connected():
            return False
        
        return True
    
    def is_failed(self, constellation: Constellation) -> bool:
        # Similar failure conditions to FormConstellationTask
        total_delta_v = constellation.get_total_delta_v_remaining()
        
        if total_delta_v < 1.0 and constellation.get_num_groups() < self.num_viewpoints:
            return True
        
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'stereo_imaging',
            'target_direction': tuple(self.target_direction),
            'num_viewpoints': self.num_viewpoints,
            'min_angular_separation_deg': np.degrees(self.min_angular_separation),
            'max_angular_separation_deg': np.degrees(self.max_angular_separation),
            'min_baseline': self.min_baseline,
            'max_baseline': self.max_baseline,
        }
    
    def get_progress(self, constellation: Constellation) -> float:
        num_groups = constellation.get_num_groups()
        viewpoint_progress = min(1.0, num_groups / self.num_viewpoints)
        
        baseline = constellation.get_max_baseline()
        target_baseline = (self.min_baseline + self.max_baseline) / 2
        baseline_progress = min(1.0, baseline / target_baseline) if target_baseline > 0 else 1.0
        
        return 0.5 * viewpoint_progress + 0.5 * baseline_progress


class MultiPointSensingTask(ConstellationTask):
    """
    Task: Distribute groups to sample multiple points in space.
    
    Useful for measuring spatial gradients in magnetic fields,
    particle densities, etc.
    """
    
    def __init__(self,
                 target_num_groups: int = 4,
                 target_volume: float = 1e9,  # cubic meters
                 min_group_size: int = 4,
                 formation_type: str = 'tetrahedron'):
        """
        Args:
            target_num_groups: Number of sampling points
            target_volume: Target volume to cover (m^3)
            min_group_size: Minimum cubes per group
            formation_type: Desired formation ('tetrahedron', 'line', 'plane', 'cube')
        """
        self.target_num_groups = target_num_groups
        self.target_volume = target_volume
        self.min_group_size = min_group_size
        self.formation_type = formation_type
    
    def compute_reward(self, constellation: Constellation,
                       action_taken: Optional[any] = None) -> float:
        metrics = ConstellationMetrics(constellation)
        
        num_groups = constellation.get_num_groups()
        
        # Component 1: Number of groups
        if num_groups == self.target_num_groups:
            group_score = 1.0
        elif num_groups < self.target_num_groups:
            group_score = num_groups / self.target_num_groups
        else:
            group_score = self.target_num_groups / num_groups
        
        # Component 2: Coverage volume
        volume = metrics.get_coverage_volume()
        if volume >= self.target_volume:
            volume_score = 1.0
        else:
            volume_score = volume / self.target_volume if self.target_volume > 0 else 0.0
        
        # Component 3: Formation quality
        formation_score = metrics.get_formation_quality(self.formation_type)
        
        # Component 4: Group sizes
        groups = constellation.get_all_groups()
        if groups:
            sizes = [len(g.cube_ids) for g in groups]
            min_size = min(sizes)
            
            if min_size >= self.min_group_size:
                size_score = 1.0
            else:
                size_score = min_size / self.min_group_size
            
            # Bonus for balanced group sizes
            mean_size = np.mean(sizes)
            if mean_size > 0:
                size_variance = np.var(sizes) / (mean_size ** 2)
                balance_bonus = max(0, 0.1 * (1.0 - size_variance))
            else:
                balance_bonus = 0.0
        else:
            size_score = 0.0
            balance_bonus = 0.0
        
        # Component 5: Science instrument exposure
        from rewards.metrics import SwarmFaceAnalyzer
        from core.cube_faces import FaceFunction
        
        analyzer = SwarmFaceAnalyzer(constellation.swarm)
        science_exposure = analyzer.get_exposure_fraction(FaceFunction.SCIENCE_INSTRUMENTS)
        
        # Component 6: Communication
        comm_score = 1.0 if constellation.is_constellation_connected() else 0.0
        
        # Combine scores
        reward = (
            0.2 * group_score +
            0.2 * volume_score +
            0.15 * formation_score +
            0.15 * size_score +
            balance_bonus +
            0.1 * science_exposure +
            0.1 * comm_score
        )
        
        if self.is_complete(constellation):
            reward += 1.0
        
        return reward
    
    def is_complete(self, constellation: Constellation) -> bool:
        # Check number of groups
        if constellation.get_num_groups() != self.target_num_groups:
            return False
        
        # Check volume
        metrics = ConstellationMetrics(constellation)
        volume = metrics.get_coverage_volume()
        if volume < self.target_volume * 0.8:  # 80% tolerance
            return False
        
        # Check group sizes
        for group in constellation.get_all_groups():
            if len(group.cube_ids) < self.min_group_size:
                return False
        
        # Check communication
        if not constellation.is_constellation_connected():
            return False
        
        return True
    
    def is_failed(self, constellation: Constellation) -> bool:
        total_delta_v = constellation.get_total_delta_v_remaining()
        
        if total_delta_v < 2.0 and constellation.get_num_groups() < self.target_num_groups:
            return True
        
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'multi_point_sensing',
            'target_num_groups': self.target_num_groups,
            'target_volume': self.target_volume,
            'min_group_size': self.min_group_size,
            'formation_type': self.formation_type,
        }
    
    def get_progress(self, constellation: Constellation) -> float:
        metrics = ConstellationMetrics(constellation)
        
        num_groups = constellation.get_num_groups()
        group_progress = min(1.0, num_groups / self.target_num_groups)
        
        volume = metrics.get_coverage_volume()
        volume_progress = min(1.0, volume / self.target_volume) if self.target_volume > 0 else 1.0
        
        return 0.5 * group_progress + 0.5 * volume_progress
    

# =============================================================================
# Communication tasks
# =============================================================================

class EarthDownlinkTask(ConstellationTask):
    """
    Maximize effective antenna aperture area pointed at Earth.

    Physical metric: effective coherent array area (m²), which is directly
    proportional to downlink data rate at any given Earth distance.

    Ideal configuration: flat plane perpendicular to Earth direction, all
    high-gain antenna faces exposed and co-aligned.
    """

    # Reference: 64 × 10cm × 10cm faces fully exposed + aligned = 0.64 m²
    REFERENCE_APERTURE_M2 = 0.64

    def __init__(
        self,
        earth_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        min_aperture_m2: float = 0.3,
        require_connected: bool = True,
    ):
        self.earth_direction = np.array(earth_direction, dtype=float)
        self.earth_direction /= np.linalg.norm(self.earth_direction)
        self.min_aperture_m2 = min_aperture_m2
        self.require_connected = require_connected

    def _aperture(self, constellation: Constellation) -> float:
        return ConstellationMetrics(constellation).get_earth_array_gain(
            tuple(self.earth_direction)
        )

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        aperture = self._aperture(constellation)
        # Normalised against reference; smoothly approaches 1
        aperture_score = min(1.0, aperture / self.REFERENCE_APERTURE_M2)

        conn_score = 1.0 if (
            not self.require_connected or constellation.is_constellation_connected()
        ) else 0.0

        prop_score = ConstellationMetrics(constellation).get_propulsion_efficiency()

        reward = 0.6 * aperture_score + 0.25 * conn_score + 0.15 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        return (
            self._aperture(constellation) >= self.min_aperture_m2
            and (not self.require_connected or constellation.is_constellation_connected())
        )

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'earth_downlink',
            'earth_direction': tuple(self.earth_direction),
            'min_aperture_m2': self.min_aperture_m2,
        }

    def get_progress(self, constellation: Constellation) -> float:
        return min(1.0, self._aperture(constellation) / self.min_aperture_m2)


class LocalRelayTask(ConstellationTask):
    """
    Position relay units so that detached explorer units can communicate
    back to the main group via a chain.

    Physical metric: minimum hop-count data-rate from each explorer group
    to the main group, approximated as the product of per-link data rates
    along the shortest communication path.

    This models deep-space relay scenarios (e.g. a probe behind Titan with
    the main swarm on the near side).
    """

    def __init__(
        self,
        explorer_group_ids: Optional[List[int]] = None,
        main_group_id: Optional[int] = None,
        target_relay_bandwidth: float = 1e6,  # bits/s
    ):
        self.explorer_group_ids = explorer_group_ids or []
        self.main_group_id = main_group_id
        self.target_relay_bandwidth = target_relay_bandwidth

    def _relay_bandwidth(self, constellation: Constellation) -> float:
        """
        Find best-effort bottleneck bandwidth from each explorer to main.
        Returns mean across explorers.
        """
        if not self.explorer_group_ids:
            return constellation.comm_reqs.base_data_rate

        links = constellation.get_communication_links()
        all_groups = list(constellation._groups.keys())

        # Build adjacency: group_id -> {neighbor_id: data_rate}
        graph: Dict[int, Dict[int, float]] = {g: {} for g in all_groups}
        for (a, b), info in links.items():
            graph[a][b] = info.get('data_rate', 0.0)
            graph[b][a] = info.get('data_rate', 0.0)

        def max_bottleneck_bw(src: int, dst: int) -> float:
            """Dijkstra-like max-bottleneck path."""
            import heapq
            best = {src: float('inf')}
            heap = [(-float('inf'), src)]
            while heap:
                neg_bw, node = heapq.heappop(heap)
                cur_bw = -neg_bw
                if node == dst:
                    return cur_bw
                if cur_bw < best.get(node, 0):
                    continue
                for nb, rate in graph.get(node, {}).items():
                    new_bw = min(cur_bw, rate)
                    if new_bw > best.get(nb, 0):
                        best[nb] = new_bw
                        heapq.heappush(heap, (-new_bw, nb))
            return 0.0

        main = self.main_group_id or all_groups[0]
        bws = [max_bottleneck_bw(eid, main) for eid in self.explorer_group_ids
               if eid in graph]
        return float(np.mean(bws)) if bws else 0.0

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        bw = self._relay_bandwidth(constellation)
        bw_score = min(1.0, bw / max(self.target_relay_bandwidth, 1.0))
        prop_score = ConstellationMetrics(constellation).get_propulsion_efficiency()

        reward = 0.75 * bw_score + 0.25 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        return self._relay_bandwidth(constellation) >= self.target_relay_bandwidth

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'local_relay',
            'explorer_group_ids': self.explorer_group_ids,
            'main_group_id': self.main_group_id,
            'target_relay_bandwidth': self.target_relay_bandwidth,
        }

    def get_progress(self, constellation: Constellation) -> float:
        bw = self._relay_bandwidth(constellation)
        return min(1.0, bw / max(self.target_relay_bandwidth, 1.0))


# =============================================================================
# Science observation tasks
# =============================================================================

class SparseApertureTask(ConstellationTask):
    """
    Maximize angular resolution for observing distant objects.

    Physical metric: angular resolution θ = λ/B (µrad), where B is the
    maximum baseline in meters.  Smaller θ is better science, but the
    reward is framed as maximising baseline to keep it positive.

    Analogy: Event Horizon Telescope, VLBI radio arrays.
    """

    def __init__(
        self,
        target_baseline_m: float = 1000.0,
        wavelength_m: float = 1e-6,
        maintain_communication: bool = True,
        min_group_size: int = 4,
    ):
        self.target_baseline_m = target_baseline_m
        self.wavelength_m = wavelength_m
        self.maintain_communication = maintain_communication
        self.min_group_size = min_group_size

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        metrics = ConstellationMetrics(constellation)
        baseline = constellation.get_max_baseline()

        # Reward grows logarithmically — large baselines are valuable but
        # the marginal gain diminishes once well past target
        if self.target_baseline_m > 0:
            baseline_score = min(1.0, np.log1p(baseline) / np.log1p(self.target_baseline_m))
        else:
            baseline_score = 0.0

        conn_score = 1.0 if (
            not self.maintain_communication or constellation.is_constellation_connected()
        ) else 0.5  # partial penalty (not total failure — science still possible)

        groups = constellation.get_all_groups()
        min_size = min((len(g.cube_ids) for g in groups), default=0)
        size_score = min(1.0, min_size / max(self.min_group_size, 1))

        prop_score = metrics.get_propulsion_efficiency()

        reward = 0.5 * baseline_score + 0.2 * conn_score + 0.15 * size_score + 0.15 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        if constellation.get_max_baseline() < self.target_baseline_m:
            return False
        if self.maintain_communication and not constellation.is_constellation_connected():
            return False
        for g in constellation.get_all_groups():
            if len(g.cube_ids) < self.min_group_size:
                return False
        return True

    def is_failed(self, constellation: Constellation) -> bool:
        dv = constellation.get_total_delta_v_remaining()
        return dv < 0.5 and constellation.get_max_baseline() < self.target_baseline_m * 0.1

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'sparse_aperture',
            'target_baseline_m': self.target_baseline_m,
            'wavelength_m': self.wavelength_m,
            'angular_resolution_urad': self.wavelength_m / max(self.target_baseline_m, 1e-3) * 1e6,
        }

    def get_progress(self, constellation: Constellation) -> float:
        baseline = constellation.get_max_baseline()
        return min(1.0, baseline / max(self.target_baseline_m, 1.0))


class OccultationArrayTask(ConstellationTask):
    """
    Form a linear array perpendicular to a shadow/occultation track.

    Physical metric: occultation baseline B_perp (m), which determines
    the spatial sampling resolution across an occulting body's limb.
    A longer perpendicular baseline lets different cubes see the event
    at slightly different impact parameters → richer ring/atmosphere profile.

    Ideal config: long thin line (e.g. 1×64) with line axis ⊥ shadow_direction.
    """

    def __init__(
        self,
        shadow_direction: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        target_perp_baseline_m: float = 500.0,
        require_connected: bool = True,
    ):
        self.shadow_direction = np.array(shadow_direction, dtype=float)
        self.shadow_direction /= np.linalg.norm(self.shadow_direction)
        self.target_perp_baseline_m = target_perp_baseline_m
        self.require_connected = require_connected

    def _perp_baseline(self, constellation: Constellation) -> float:
        return ConstellationMetrics(constellation).get_occultation_baseline_m(
            tuple(self.shadow_direction)
        )

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        perp = self._perp_baseline(constellation)
        perp_score = min(1.0, perp / max(self.target_perp_baseline_m, 1.0))

        conn_score = 1.0 if (
            not self.require_connected or constellation.is_constellation_connected()
        ) else 0.0

        prop_score = ConstellationMetrics(constellation).get_propulsion_efficiency()

        reward = 0.65 * perp_score + 0.2 * conn_score + 0.15 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        return (
            self._perp_baseline(constellation) >= self.target_perp_baseline_m
            and (not self.require_connected or constellation.is_constellation_connected())
        )

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'occultation_array',
            'shadow_direction': tuple(self.shadow_direction),
            'target_perp_baseline_m': self.target_perp_baseline_m,
        }

    def get_progress(self, constellation: Constellation) -> float:
        return min(1.0, self._perp_baseline(constellation) / max(self.target_perp_baseline_m, 1.0))


class InSituFieldTask(ConstellationTask):
    """
    Distribute groups in 3D to measure spatial gradients of a field
    (magnetic, plasma density, particle flux, etc.).

    Physical metrics:
    - Convex hull volume of group positions (m³) — spatial coverage
    - Sampling uniformity — how evenly spread the measurement points are
    - Science instrument exposure fraction

    Ideal config: groups at vertices of a regular tetrahedron or larger polyhedron.
    """

    def __init__(
        self,
        target_volume_m3: float = 1e9,
        target_num_groups: int = 4,
        min_uniformity: float = 0.5,
        sun_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        sun_distance_au: float = 10.0,
    ):
        self.target_volume_m3 = target_volume_m3
        self.target_num_groups = target_num_groups
        self.min_uniformity = min_uniformity
        self.sun_direction = sun_direction
        self.sun_distance_au = sun_distance_au

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        metrics = ConstellationMetrics(constellation)

        volume = metrics.get_coverage_volume()
        volume_score = min(1.0, volume / max(self.target_volume_m3, 1.0))

        uniformity = metrics.get_sampling_uniformity()

        num_groups = constellation.get_num_groups()
        if num_groups == self.target_num_groups:
            group_score = 1.0
        elif num_groups < self.target_num_groups:
            group_score = num_groups / self.target_num_groups
        else:
            group_score = self.target_num_groups / num_groups

        # Science exposure (want instruments unobstructed)
        from rewards.metrics import SwarmFaceAnalyzer
        from core.cube_faces import FaceFunction
        # science_scores = []
        # for group in constellation._groups.values():
        #     analyzer = SwarmFaceAnalyzer(group.swarm)
        #     science_scores.append(analyzer.get_exposure_fraction(FaceFunction.SCIENCE_INSTRUMENTS))
        # science_score = float(np.mean(science_scores)) if science_scores else 0.0
        analyzer = SwarmFaceAnalyzer(constellation.swarm)
        science_score = analyzer.get_exposure_fraction(FaceFunction.SCIENCE_INSTRUMENTS)

        prop_score = metrics.get_propulsion_efficiency()

        reward = (
            0.30 * volume_score
            + 0.25 * uniformity
            + 0.20 * group_score
            + 0.15 * science_score
            + 0.10 * prop_score
        )

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        metrics = ConstellationMetrics(constellation)
        return (
            constellation.get_num_groups() == self.target_num_groups
            and metrics.get_coverage_volume() >= self.target_volume_m3 * 0.8
            and metrics.get_sampling_uniformity() >= self.min_uniformity
        )

    def is_failed(self, constellation: Constellation) -> bool:
        dv = constellation.get_total_delta_v_remaining()
        return dv < 0.5 and constellation.get_num_groups() < 2

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'in_situ_field',
            'target_volume_m3': self.target_volume_m3,
            'target_num_groups': self.target_num_groups,
            'min_uniformity': self.min_uniformity,
        }

    def get_progress(self, constellation: Constellation) -> float:
        metrics = ConstellationMetrics(constellation)
        volume_progress = min(1.0, metrics.get_coverage_volume() / max(self.target_volume_m3, 1.0))
        group_progress = min(1.0, constellation.get_num_groups() / max(self.target_num_groups, 1))
        return 0.5 * volume_progress + 0.5 * group_progress


# =============================================================================
# Survival / operational tasks
# =============================================================================

class CruiseModeTask(ConstellationTask):
    """
    Minimise heat loss and power consumption during long interplanetary transit.

    Physical metrics:
    - Radiative heat loss in Watts (from exposed surface area) — lower is better
    - Solar power collected (should exceed baseline consumption)
    - Power-sharing efficiency (compact structures share power more easily)

    Ideal config: all groups docked as a single compact 4×4×4 cube.
    """

    # Rough baseline power draw: ~0.5 W per cube (heaters, clocks, housekeeping)
    POWER_DRAW_PER_CUBE_W = 0.5

    def __init__(
        self,
        sun_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        sun_distance_au: float = 10.0,
        max_heat_loss_w: Optional[float] = None,
    ):
        self.sun_direction = sun_direction
        self.sun_distance_au = sun_distance_au
        self.max_heat_loss_w = max_heat_loss_w  # None = auto-set from n_cubes

    def _power_budget(self, constellation: Constellation) -> Tuple[float, float]:
        """Returns (heat_loss_W, solar_power_W)."""
        metrics = ConstellationMetrics(constellation)
        return (
            metrics.get_heat_loss_watts(),
            metrics.get_solar_power_watts(self.sun_direction, self.sun_distance_au),
        )

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        metrics = ConstellationMetrics(constellation)
        heat_loss, solar_power = self._power_budget(constellation)

        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        consumption = n * self.POWER_DRAW_PER_CUBE_W

        # Reference heat loss: all cubes separate (max surface)
        max_heat = n * 6 * ConstellationMetrics.CUBE_FACE_AREA_M2 * (
            ConstellationMetrics.CUBE_EMISSIVITY
            * ConstellationMetrics.STEFAN_BOLTZMANN
            * (ConstellationMetrics.OPERATING_TEMP_K ** 4 - ConstellationMetrics.SPACE_TEMP_K ** 4)
        )
        # Min heat: compact cube (6 × side² faces)
        side = max(1, round(n ** (1 / 3)))
        min_heat = 6 * (side ** 2) * ConstellationMetrics.CUBE_FACE_AREA_M2 * (
            ConstellationMetrics.CUBE_EMISSIVITY
            * ConstellationMetrics.STEFAN_BOLTZMANN
            * (ConstellationMetrics.OPERATING_TEMP_K ** 4 - ConstellationMetrics.SPACE_TEMP_K ** 4)
        )

        heat_score = 1.0 - np.clip(
            (heat_loss - min_heat) / max(max_heat - min_heat, 1e-9), 0.0, 1.0
        )

        # Power balance: positive surplus is good, deficit is penalised
        surplus = solar_power - consumption
        power_score = np.clip(0.5 + surplus / max(consumption, 1e-3), 0.0, 1.0)

        prop_score = metrics.get_propulsion_efficiency()

        reward = 0.55 * heat_score + 0.30 * power_score + 0.15 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        heat_loss, solar_power = self._power_budget(constellation)
        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        consumption = n * self.POWER_DRAW_PER_CUBE_W
        ref = self.max_heat_loss_w or (consumption * 1.5)
        return heat_loss <= ref and solar_power >= consumption * 0.8

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'cruise_mode',
            'sun_direction': self.sun_direction,
            'sun_distance_au': self.sun_distance_au,
        }

    def get_progress(self, constellation: Constellation) -> float:
        heat_loss, solar_power = self._power_budget(constellation)
        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        consumption = n * self.POWER_DRAW_PER_CUBE_W
        ref = self.max_heat_loss_w or (consumption * 1.5)
        heat_progress = 1.0 - np.clip(heat_loss / max(ref * 2, 1e-9), 0.0, 1.0)
        power_progress = np.clip(solar_power / max(consumption, 1e-9), 0.0, 1.0)
        return 0.5 * heat_progress + 0.5 * power_progress


class SolarCollectionTask(ConstellationTask):
    """
    Maximise total solar power collection — critical when power is scarce
    (e.g. around Saturn where flux is ~1% of Earth's).

    Physical metric: total collected power in Watts = solar_flux(AU) × effective_area.
    Ideal config: flat 8×8 plane with solar faces normal to sun vector.
    """

    def __init__(
        self,
        sun_direction: Tuple[float, float, float] = (0.0, 0.0, -1.0),
        sun_distance_au: float = 10.0,
        target_power_w: Optional[float] = None,  # None = auto (50% of theoretical max)
    ):
        self.sun_direction = sun_direction
        self.sun_distance_au = sun_distance_au
        self.target_power_w = target_power_w

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        metrics = ConstellationMetrics(constellation)
        solar_power = metrics.get_solar_power_watts(self.sun_direction, self.sun_distance_au)

        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        solar_flux = ConstellationMetrics.SOLAR_CONSTANT_W_PER_M2 / (self.sun_distance_au ** 2)
        theoretical_max = n * ConstellationMetrics.CUBE_FACE_AREA_M2 * solar_flux
        target = self.target_power_w or (0.5 * theoretical_max)

        power_score = min(1.0, solar_power / max(target, 1e-9))
        prop_score = metrics.get_propulsion_efficiency()

        reward = 0.80 * power_score + 0.20 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        metrics = ConstellationMetrics(constellation)
        solar_power = metrics.get_solar_power_watts(self.sun_direction, self.sun_distance_au)
        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        solar_flux = ConstellationMetrics.SOLAR_CONSTANT_W_PER_M2 / (self.sun_distance_au ** 2)
        theoretical_max = n * ConstellationMetrics.CUBE_FACE_AREA_M2 * solar_flux
        target = self.target_power_w or (0.5 * theoretical_max)
        return solar_power >= target

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'solar_collection',
            'sun_direction': self.sun_direction,
            'sun_distance_au': self.sun_distance_au,
        }

    def get_progress(self, constellation: Constellation) -> float:
        metrics = ConstellationMetrics(constellation)
        solar_power = metrics.get_solar_power_watts(self.sun_direction, self.sun_distance_au)
        n = sum(len(g.cube_ids) for g in constellation._groups.values())
        solar_flux = ConstellationMetrics.SOLAR_CONSTANT_W_PER_M2 / (self.sun_distance_au ** 2)
        theoretical_max = n * ConstellationMetrics.CUBE_FACE_AREA_M2 * solar_flux
        target = self.target_power_w or (0.5 * theoretical_max)
        return min(1.0, solar_power / max(target, 1e-9))


class ThermalShieldTask(ConstellationTask):
    """
    Protect sensitive/critical units by interposing non-critical units
    between them and a thermal/radiation threat (e.g. Jupiter's radiation belts,
    a close solar approach, or re-entry heating).

    Physical metric:
    - Shielded fraction of critical units (0–1)
    - Mean shielding depth (layers of intervening cubes)
    - Estimated flux reduction on critical units

    Ideal config: sacrificial outer layer around critical core.
    """

    def __init__(
        self,
        threat_direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
        critical_cube_ids: Optional[Set[int]] = None,
        target_shielded_fraction: float = 0.8,
        target_shield_depth: float = 2.0,
    ):
        self.threat_direction = np.array(threat_direction, dtype=float)
        self.threat_direction /= np.linalg.norm(self.threat_direction)
        self.critical_cube_ids = critical_cube_ids
        self.target_shielded_fraction = target_shielded_fraction
        self.target_shield_depth = target_shield_depth

    def _shielding(self, constellation: Constellation) -> Dict[str, float]:
        return ConstellationMetrics(constellation).get_critical_unit_shielding(
            tuple(self.threat_direction), self.critical_cube_ids
        )

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        s = self._shielding(constellation)

        shield_frac_score = min(1.0, s['shielded_fraction'] / max(self.target_shielded_fraction, 1e-3))
        depth_score = min(1.0, s['mean_shield_depth'] / max(self.target_shield_depth, 1e-3))
        # Flux reduction: lower is better (0 = full block, 1 = no shielding)
        flux_score = 1.0 - s['threat_flux_reduction']

        prop_score = ConstellationMetrics(constellation).get_propulsion_efficiency()

        reward = 0.35 * shield_frac_score + 0.30 * flux_score + 0.20 * depth_score + 0.15 * prop_score

        if self.is_complete(constellation):
            reward += 1.0

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        s = self._shielding(constellation)
        return (
            s['shielded_fraction'] >= self.target_shielded_fraction
            and s['mean_shield_depth'] >= self.target_shield_depth
        )

    def is_failed(self, constellation: Constellation) -> bool:
        return False

    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'thermal_shield',
            'threat_direction': tuple(self.threat_direction),
            'target_shielded_fraction': self.target_shielded_fraction,
            'target_shield_depth': self.target_shield_depth,
        }

    def get_progress(self, constellation: Constellation) -> float:
        s = self._shielding(constellation)
        frac_progress = s['shielded_fraction'] / max(self.target_shielded_fraction, 1e-3)
        depth_progress = s['mean_shield_depth'] / max(self.target_shield_depth, 1e-3)
        return min(1.0, 0.5 * frac_progress + 0.5 * depth_progress)


class DamagedReconfigTask(ConstellationTask):
    """
    After losing some units, gracefully reconfigure to maintain mission capability.

    The task wraps another 'primary_task' and measures how close the degraded
    swarm can get to the primary objective, normalised by what would be
    achievable with the full complement of cubes.

    This teaches the agent to maintain partial functionality under unit loss —
    a key requirement for deep-space resilience.

    Usage:
        base_task = SparseApertureTask(target_baseline_m=1000)
        task = DamagedReconfigTask(primary_task=base_task, damage_fraction=0.25)
    """

    def __init__(
        self,
        primary_task: ConstellationTask,
        damage_fraction: float = 0.25,
        graceful_threshold: float = 0.7,
    ):
        """
        Args:
            primary_task: The underlying mission objective.
            damage_fraction: Fraction of cubes that were lost (informational; the
                             environment handles actual removal before reset).
            graceful_threshold: Fraction of full-capability score to accept as 'complete'.
        """
        self.primary_task = primary_task
        self.damage_fraction = damage_fraction
        self.graceful_threshold = graceful_threshold

        # Baseline score is set on first call; represents undamaged capability
        self._baseline_score: Optional[float] = None

    def compute_reward(self, constellation: Constellation,
                       action_taken=None) -> float:
        raw_reward = self.primary_task.compute_reward(constellation, action_taken)

        # Grace bonus: reward partial success more than the underlying task alone
        progress = self.primary_task.get_progress(constellation)
        grace_bonus = 0.3 * min(1.0, progress / max(self.graceful_threshold, 1e-3))

        reward = raw_reward + grace_bonus

        if self.is_complete(constellation):
            reward += 0.5  # Smaller bonus — partial success is expected

        return reward

    def is_complete(self, constellation: Constellation) -> bool:
        return self.primary_task.get_progress(constellation) >= self.graceful_threshold

    def is_failed(self, constellation: Constellation) -> bool:
        return self.primary_task.is_failed(constellation)

    def get_task_info(self) -> Dict[str, any]:
        info = self.primary_task.get_task_info()
        info['task_type'] = 'damaged_reconfig'
        info['wrapped_task'] = info.get('task_type', 'unknown')
        info['damage_fraction'] = self.damage_fraction
        info['graceful_threshold'] = self.graceful_threshold
        return info

    def get_progress(self, constellation: Constellation) -> float:
        return self.primary_task.get_progress(constellation)