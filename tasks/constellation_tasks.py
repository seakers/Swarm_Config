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
        
        for group in constellation.get_all_groups():
            # Create a temporary view of just this group's cubes
            # For simplicity, we'll check overall camera exposure
            analyzer = SwarmFaceAnalyzer(constellation.swarm)
            
            # Get cubes in this group with exposed cameras
            camera_alignment = analyzer.compute_function_alignment(
                FaceFunction.CAMERA,
                tuple(self.target_direction),
                only_exposed=True
            )
            
            alignment_scores.append(max(0, camera_alignment['mean']))
        
        if alignment_scores:
            camera_score = np.mean(alignment_scores)
        else:
            camera_score = 0.0
        
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