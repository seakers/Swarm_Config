import numpy as np
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple, Optional, Dict, Set

from core.swarm import Swarm
from core.cube_faces import FaceFunction
from mechanics.moves import HingeMove
from rewards.metrics import SwarmMetrics, MissionModeScorer, SwarmFaceAnalyzer


class TaskType(Enum):
    """Types of reconfiguration tasks."""
    FORM_CUBE = auto()        # Form compact cube (4x4x4)
    FORM_PLANE = auto()       # Form flat plane (8x8) with target normal
    FORM_LINE = auto()        # Form line in target direction
    MAXIMIZE_SPREAD = auto()  # Maximize spatial extent
    MINIMIZE_SURFACE = auto() # Minimize exposed surface area
    SPLIT_GROUPS = auto()     # Split into N separate groups
    CUSTOM_SHAPE = auto()     # Match a custom target configuration


class Task(ABC):
    """
    Abstract base class for reconfiguration tasks.
    
    A task defines:
    - A goal configuration or property
    - A reward function measuring progress toward the goal
    - Success/failure criteria
    """
    
    @abstractmethod
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        """
        Compute reward for the current swarm state.
        
        Args:
            swarm: Current swarm configuration
            action_taken: The action that led to this state (optional)
            
        Returns:
            Reward value (higher is better)
        """
        pass
    
    @abstractmethod
    def is_complete(self, swarm: Swarm) -> bool:
        """Check if the task has been successfully completed."""
        pass
    
    @abstractmethod
    def is_failed(self, swarm: Swarm) -> bool:
        """Check if the task has failed (unrecoverable)."""
        pass
    
    @abstractmethod
    def get_task_info(self) -> Dict[str, any]:
        """Get task-specific information for observation."""
        pass
    
    def get_progress(self, swarm: Swarm) -> float:
        """
        Get progress toward completion (0.0 to 1.0).
        
        Default implementation returns 0 or 1 based on is_complete.
        Subclasses should override for continuous progress tracking.
        """
        return 1.0 if self.is_complete(swarm) else 0.0


class FormCubeTask(Task):
    """
    Task: Arrange all cubes into a compact cube formation.
    
    For 64 cubes, the target is a 4x4x4 cube.
    """
    
    def __init__(self, target_size: int = 4):
        self.target_size = target_size
        self.target_volume = target_size ** 3
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        # Primary reward: compactness (more connections = more compact)
        compactness = metrics.compactness()
        
        # Secondary: minimize surface area
        surface_area = metrics.surface_area()
        # For 4x4x4 cube: minimum surface area is 6 * 16 = 96
        # For 64 separate cubes: maximum is 64 * 6 = 384
        min_surface = 6 * (self.target_size ** 2)
        max_surface = swarm.num_cubes * 6
        surface_score = 1.0 - (surface_area - min_surface) / (max_surface - min_surface)
        
        # Tertiary: bounding box should approach target size
        bounds = swarm.get_bounds()
        dims = [bounds[1][i] - bounds[0][i] + 1 for i in range(3)]
        dims.sort()
        target_dims = [self.target_size, self.target_size, self.target_size]
        
        # Penalize dimensions that are too large or too small
        dim_penalty = sum(abs(d - t) for d, t in zip(dims, target_dims)) / (3 * self.target_size)
        dim_score = max(0, 1.0 - dim_penalty)
        
        # Combine rewards
        reward = 0.4 * compactness + 0.4 * surface_score + 0.2 * dim_score
        
        # Bonus for completion
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        bounds = swarm.get_bounds()
        dims = sorted([bounds[1][i] - bounds[0][i] + 1 for i in range(3)])
        target = [self.target_size, self.target_size, self.target_size]
        
        return dims == target and swarm.is_connected()
    
    def is_failed(self, swarm: Swarm) -> bool:
        # This task can't really "fail" - we can always keep trying
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'form_cube',
            'target_size': self.target_size,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        return metrics.compactness()


class FormPlaneTask(Task):
    """
    Task: Arrange all cubes into a flat plane perpendicular to a target normal.
    
    For 64 cubes, the target is an 8x8 plane.
    """
    
    def __init__(self, 
                 normal: Tuple[float, float, float] = (0, 0, 1),
                 width: int = 8, 
                 height: int = 8):
        self.normal = np.array(normal, dtype=float)
        self.normal = self.normal / np.linalg.norm(self.normal)
        self.width = width
        self.height = height
        self.target_area = width * height
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        # Primary: planarity (all cubes on same plane)
        planarity = metrics.planarity(tuple(self.normal), tolerance=0.5)
        
        # Secondary: projected area (should be maximized)
        projected_area = metrics.planar_coverage(tuple(self.normal))
        area_score = min(1.0, projected_area / self.target_area)
        
        # Tertiary: connectivity
        connectivity = metrics.connectivity_ratio()
        
        # Combine
        reward = 0.4 * planarity + 0.3 * area_score + 0.3 * connectivity
        
        # Bonus for completion
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        metrics = SwarmMetrics(swarm)
        
        # All cubes must be on the same plane (within tolerance)
        planarity = metrics.planarity(tuple(self.normal), tolerance=0.5)
        if planarity < 0.99:  # Allow tiny error
            return False
        
        # Must be connected
        if not swarm.is_connected():
            return False
        
        # Check dimensions
        bounds = swarm.get_bounds()
        dims = sorted([bounds[1][i] - bounds[0][i] + 1 for i in range(3)])
        
        # Smallest dimension should be 1 (it's a plane)
        # Other two should be width and height
        target_dims = sorted([1, self.width, self.height])
        
        return dims == target_dims
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'form_plane',
            'normal': tuple(self.normal),
            'width': self.width,
            'height': self.height,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        return metrics.planarity(tuple(self.normal), tolerance=0.5)


class FormLineTask(Task):
    """
    Task: Arrange all cubes into a straight line along a target axis.
    """
    
    def __init__(self, 
                 axis: Tuple[float, float, float] = (1, 0, 0),
                 length: int = 64):
        self.axis = np.array(axis, dtype=float)
        self.axis = self.axis / np.linalg.norm(self.axis)
        self.length = length
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        # Primary: linearity
        linearity = metrics.linearity(tuple(self.axis), tolerance=0.5)
        
        # Secondary: extent along axis (should be length - 1)
        extent = metrics.maximum_baseline()
        target_extent = self.length - 1
        extent_score = min(1.0, extent / target_extent) if target_extent > 0 else 1.0
        
        # Tertiary: connectivity
        connectivity = metrics.connectivity_ratio()
        
        # Combine
        reward = 0.4 * linearity + 0.3 * extent_score + 0.3 * connectivity
        
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        metrics = SwarmMetrics(swarm)
        
        linearity = metrics.linearity(tuple(self.axis), tolerance=0.5)
        if linearity < 0.99:
            return False
        
        if not swarm.is_connected():
            return False
        
        bounds = swarm.get_bounds()
        dims = sorted([bounds[1][i] - bounds[0][i] + 1 for i in range(3)])
        
        # Two dimensions should be 1, one should be length
        target_dims = sorted([1, 1, self.length])
        
        return dims == target_dims
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'form_line',
            'axis': tuple(self.axis),
            'length': self.length,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        return metrics.linearity(tuple(self.axis), tolerance=0.5)


class MaximizeSpreadTask(Task):
    """
    Task: Maximize the spatial extent of the swarm while maintaining connectivity.
    
    Useful for sparse aperture imaging.
    """
    
    def __init__(self, target_baseline: float = 20.0, maintain_connectivity: bool = True):
        self.target_baseline = target_baseline
        self.maintain_connectivity = maintain_connectivity
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        # Primary: maximum baseline distance
        baseline = metrics.maximum_baseline()
        baseline_score = min(1.0, baseline / self.target_baseline)
        
        # Connectivity penalty/requirement
        if self.maintain_connectivity:
            connectivity = metrics.connectivity_ratio()
            if connectivity < 1.0:
                # Heavy penalty for disconnection
                return baseline_score * connectivity * 0.5
        
        reward = baseline_score
        
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        metrics = SwarmMetrics(swarm)
        baseline = metrics.maximum_baseline()
        
        if baseline < self.target_baseline:
            return False
        
        if self.maintain_connectivity and not swarm.is_connected():
            return False
        
        return True
    
    def is_failed(self, swarm: Swarm) -> bool:
        if self.maintain_connectivity:
            return not swarm.is_connected()
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'maximize_spread',
            'target_baseline': self.target_baseline,
            'maintain_connectivity': self.maintain_connectivity,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        baseline = metrics.maximum_baseline()
        return min(1.0, baseline / self.target_baseline)


class MinimizeSurfaceTask(Task):
    """
    Task: Minimize exposed surface area (cruise mode / thermal conservation).
    
    The optimal configuration is a compact cube.
    """
    
    def __init__(self):
        pass
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        surface_area = metrics.surface_area()
        
        # Calculate theoretical minimum (for a cube)
        n = swarm.num_cubes
        side = round(n ** (1/3))
        min_surface = 6 * (side ** 2)
        max_surface = 6 * n
        
        # Reward is how close we are to minimum
        if max_surface == min_surface:
            surface_score = 1.0
        else:
            surface_score = 1.0 - (surface_area - min_surface) / (max_surface - min_surface)
        
        # Must stay connected
        if not swarm.is_connected():
            surface_score *= 0.5
        
        reward = surface_score
        
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        n = swarm.num_cubes
        side = round(n ** (1/3))
        
        # Check if it's a perfect cube
        if side ** 3 != n:
            # Not a perfect cube number, can't achieve minimum
            return False
        
        min_surface = 6 * (side ** 2)
        actual_surface = swarm.get_surface_area()
        
        return actual_surface == min_surface and swarm.is_connected()
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'minimize_surface',
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        surface_area = metrics.surface_area()
        
        n = self.swarm.num_cubes if hasattr(self, 'swarm') else 64
        side = round(n ** (1/3))
        min_surface = 6 * (side ** 2)
        max_surface = 6 * n
        
        if max_surface == min_surface:
            return 1.0
        
        return 1.0 - (surface_area - min_surface) / (max_surface - min_surface)


class SplitGroupsTask(Task):
    """
    Task: Split the swarm into N separate connected groups.
    
    Useful for multi-point observations or distributed sensing.
    """
    
    def __init__(self, num_groups: int = 2, min_group_size: int = 1):
        self.num_groups = num_groups
        self.min_group_size = min_group_size
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        components = swarm.get_connected_components()
        num_components = len(components)
        
        # Reward for getting closer to target number of groups
        if num_components <= self.num_groups:
            group_score = num_components / self.num_groups
        else:
            # Penalty for too many groups
            group_score = self.num_groups / num_components
        
        # Check if all groups meet minimum size
        valid_groups = sum(1 for c in components if len(c) >= self.min_group_size)
        size_score = valid_groups / max(num_components, 1)
        
        # Balance score - groups should be roughly equal size
        if components:
            sizes = [len(c) for c in components]
            mean_size = sum(sizes) / len(sizes)
            variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            std_dev = variance ** 0.5
            # Lower std_dev is better
            balance_score = 1.0 / (1.0 + std_dev / mean_size) if mean_size > 0 else 0
        else:
            balance_score = 0
        
        reward = 0.5 * group_score + 0.3 * size_score + 0.2 * balance_score
        
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        components = swarm.get_connected_components()
        
        if len(components) != self.num_groups:
            return False
        
        # Check all groups meet minimum size
        for component in components:
            if len(component) < self.min_group_size:
                return False
        
        return True
    
    def is_failed(self, swarm: Swarm) -> bool:
        # Failed if any cube becomes completely isolated (size 1) 
        # when min_group_size > 1
        if self.min_group_size > 1:
            components = swarm.get_connected_components()
            isolated = sum(1 for c in components if len(c) < self.min_group_size)
            # Allow some tolerance - only fail if too many isolated
            return isolated > (swarm.num_cubes - self.num_groups * self.min_group_size)
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'split_groups',
            'num_groups': self.num_groups,
            'min_group_size': self.min_group_size,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        components = swarm.get_connected_components()
        return min(1.0, len(components) / self.num_groups)


class TargetConfigurationTask(Task):
    """
    Task: Match a specific target configuration of positions.
    
    The most general task type - can represent any desired shape.
    """
    
    def __init__(self, target_positions: Set[Tuple[int, int, int]],
                 position_tolerance: int = 0):
        """
        Args:
            target_positions: Set of (x, y, z) positions that should be occupied
            position_tolerance: Allow positions to be off by this many cells
        """
        self.target_positions = target_positions
        self.position_tolerance = position_tolerance
        
        # Compute centroid of target for translation-invariant matching
        if target_positions:
            xs = [p[0] for p in target_positions]
            ys = [p[1] for p in target_positions]
            zs = [p[2] for p in target_positions]
            self.target_centroid = (
                sum(xs) / len(xs),
                sum(ys) / len(ys),
                sum(zs) / len(zs)
            )
        else:
            self.target_centroid = (0, 0, 0)
    
    def _get_translated_positions(self, swarm: Swarm) -> Set[Tuple[int, int, int]]:
        """Get swarm positions translated so centroid matches target centroid."""
        current_centroid = swarm.get_center_of_mass()
        
        # Translation to align centroids
        dx = round(self.target_centroid[0] - current_centroid[0])
        dy = round(self.target_centroid[1] - current_centroid[1])
        dz = round(self.target_centroid[2] - current_centroid[2])
        
        translated = set()
        for cube in swarm.get_all_cubes():
            new_pos = (
                cube.position[0] + dx,
                cube.position[1] + dy,
                cube.position[2] + dz
            )
            translated.add(new_pos)
        
        return translated
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        translated = self._get_translated_positions(swarm)
        
        # Count positions that match target
        if self.position_tolerance == 0:
            matches = len(translated & self.target_positions)
        else:
            matches = 0
            for pos in translated:
                for target in self.target_positions:
                    dist = sum(abs(pos[i] - target[i]) for i in range(3))
                    if dist <= self.position_tolerance:
                        matches += 1
                        break
        
        match_score = matches / len(self.target_positions) if self.target_positions else 1.0
        
        # Penalty for extra positions outside target
        extra = len(translated - self.target_positions)
        extra_penalty = extra / swarm.num_cubes
        
        reward = match_score - 0.5 * extra_penalty
        
        if self.is_complete(swarm):
            reward += 1.0
        
        return max(0, reward)
    
    def is_complete(self, swarm: Swarm) -> bool:
        translated = self._get_translated_positions(swarm)
        
        if self.position_tolerance == 0:
            return translated == self.target_positions
        else:
            # Check each target has a matching position within tolerance
            for target in self.target_positions:
                found = False
                for pos in translated:
                    dist = sum(abs(pos[i] - target[i]) for i in range(3))
                    if dist <= self.position_tolerance:
                        found = True
                        break
                if not found:
                    return False
            return True
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'target_configuration',
            'num_target_positions': len(self.target_positions),
            'tolerance': self.position_tolerance,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        translated = self._get_translated_positions(swarm)
        matches = len(translated & self.target_positions)
        return matches / len(self.target_positions) if self.target_positions else 1.0
    

# =============================================================================
# Updated tasks with face function awareness
# =============================================================================

class FaceAwareFormPlaneTask(Task):
    """
    Form a plane with specific face alignment requirements.
    
    Not just any plane - the plane should have:
    - Solar arrays facing the sun (if in solar charging mode)
    - Antennas facing Earth (if in communication mode)
    - Cameras facing target (if in observation mode)
    """
    
    def __init__(self,
                 plane_normal: Tuple[float, float, float] = (0, 0, 1),
                 required_function_alignment: Optional[Dict[FaceFunction, Tuple[float, float, float]]] = None,
                 alignment_weight: float = 0.3,
                 width: int = 8,
                 height: int = 8):
        """
        Args:
            plane_normal: Normal direction of the plane to form
            required_function_alignment: Dict mapping face functions to directions they should face
            alignment_weight: How much to weight alignment vs. just forming the plane
            width: Target width of plane
            height: Target height of plane
        """
        self.plane_normal = np.array(plane_normal, dtype=float)
        self.plane_normal = self.plane_normal / np.linalg.norm(self.plane_normal)
        
        self.required_function_alignment = required_function_alignment or {}
        self.alignment_weight = alignment_weight
        self.width = width
        self.height = height
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        metrics = SwarmMetrics(swarm)
        
        # Planarity score
        planarity = metrics.planarity(tuple(self.plane_normal), tolerance=0.5)
        
        # Shape score (should be width x height)
        bounds = swarm.get_bounds()
        dims = sorted([bounds[1][i] - bounds[0][i] + 1 for i in range(3)])
        target_dims = sorted([1, self.width, self.height])
        dim_error = sum(abs(d - t) for d, t in zip(dims, target_dims))
        shape_score = max(0, 1.0 - dim_error / (self.width + self.height))
        
        # Connectivity
        connectivity = metrics.connectivity_ratio()
        
        # Base formation score
        formation_score = 0.4 * planarity + 0.3 * shape_score + 0.3 * connectivity
        
        # Face alignment score
        if self.required_function_alignment:
            analyzer = SwarmFaceAnalyzer(swarm)
            alignment_scores = []
            
            for function, direction in self.required_function_alignment.items():
                alignment_data = analyzer.compute_function_alignment(
                    function, direction, only_exposed=True
                )
                # Convert mean alignment from [-1, 1] to [0, 1]
                normalized = (alignment_data['mean'] + 1.0) / 2.0
                alignment_scores.append(normalized)
            
            if alignment_scores:
                alignment_score = np.mean(alignment_scores)
            else:
                alignment_score = 1.0
        else:
            alignment_score = 1.0
        
        # Combine scores
        reward = ((1.0 - self.alignment_weight) * formation_score + 
                  self.alignment_weight * alignment_score)
        
        # Bonus for completion
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        metrics = SwarmMetrics(swarm)
        
        # Check planarity
        planarity = metrics.planarity(tuple(self.plane_normal), tolerance=0.5)
        if planarity < 0.99:
            return False
        
        # Check connectivity
        if not swarm.is_connected():
            return False
        
        # Check dimensions
        bounds = swarm.get_bounds()
        dims = sorted([bounds[1][i] - bounds[0][i] + 1 for i in range(3)])
        target_dims = sorted([1, self.width, self.height])
        
        if dims != target_dims:
            return False
        
        # Check face alignment requirements
        if self.required_function_alignment:
            analyzer = SwarmFaceAnalyzer(swarm)
            
            for function, direction in self.required_function_alignment.items():
                alignment_data = analyzer.compute_function_alignment(
                    function, direction, only_exposed=True
                )
                # Require mean alignment > 0.8 (roughly within ~37 degrees)
                if alignment_data['mean'] < 0.8:
                    return False
        
        return True
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'face_aware_form_plane',
            'plane_normal': tuple(self.plane_normal),
            'required_alignments': {
                func.name: dir for func, dir in self.required_function_alignment.items()
            },
            'alignment_weight': self.alignment_weight,
            'width': self.width,
            'height': self.height,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        metrics = SwarmMetrics(swarm)
        return metrics.planarity(tuple(self.plane_normal), tolerance=0.5)


class MissionModeTask(Task):
    """
    A task that optimizes for a specific mission mode.
    
    Instead of targeting a specific shape, this task optimizes for
    the mission mode score computed by MissionModeScorer.
    """
    
    def __init__(self,
                 mode: str,
                 sun_direction: Tuple[float, float, float] = (0, 0, -1),
                 earth_direction: Tuple[float, float, float] = (1, 0, 0),
                 target_direction: Tuple[float, float, float] = (0, 1, 0),
                 sun_distance_au: float = 10.0,
                 target_score: float = 0.9):
        """
        Args:
            mode: Mission mode ('communication', 'observation', 'cruise', etc.)
            sun_direction: Direction to sun
            earth_direction: Direction to Earth
            target_direction: Direction to science target
            sun_distance_au: Distance from sun in AU
            target_score: Score threshold to consider task complete
        """
        self.mode = mode
        self.sun_direction = sun_direction
        self.earth_direction = earth_direction
        self.target_direction = target_direction
        self.sun_distance_au = sun_distance_au
        self.target_score = target_score
        
        self._scorer: Optional[MissionModeScorer] = None
    
    def _get_scorer(self, swarm: Swarm) -> MissionModeScorer:
        """Get or create scorer for the swarm."""
        if self._scorer is None or self._scorer.swarm is not swarm:
            self._scorer = MissionModeScorer(swarm)
        else:
            self._scorer.refresh()
        return self._scorer
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        scorer = self._get_scorer(swarm)
        
        result = scorer.score_configuration(
            self.mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au
        )
        
        reward = result['total_score']
        
        # Bonus for completion
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        scorer = self._get_scorer(swarm)
        
        result = scorer.score_configuration(
            self.mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au
        )
        
        return result['total_score'] >= self.target_score
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'mission_mode',
            'mode': self.mode,
            'sun_direction': self.sun_direction,
            'earth_direction': self.earth_direction,
            'target_direction': self.target_direction,
            'sun_distance_au': self.sun_distance_au,
            'target_score': self.target_score,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        scorer = self._get_scorer(swarm)
        
        result = scorer.score_configuration(
            self.mode,
            self.sun_direction,
            self.earth_direction,
            self.target_direction,
            self.sun_distance_au
        )
        
        return min(1.0, result['total_score'] / self.target_score)


class MultiObjectiveMissionTask(Task):
    """
    A task that balances multiple mission objectives simultaneously.
    
    This is useful for configurations that need to do multiple things
    at once, e.g., observe a target while maintaining Earth communication
    and keeping power-positive.
    """
    
    def __init__(self,
                 objectives: Dict[str, float],
                 sun_direction: Tuple[float, float, float] = (0, 0, -1),
                 earth_direction: Tuple[float, float, float] = (1, 0, 0),
                 target_direction: Tuple[float, float, float] = (0, 1, 0),
                 sun_distance_au: float = 10.0,
                 target_score: float = 0.8):
        """
        Args:
            objectives: Dict mapping mode names to their weights (should sum to 1)
            sun_direction: Direction to sun
            earth_direction: Direction to Earth
            target_direction: Direction to science target
            sun_distance_au: Distance from sun in AU
            target_score: Combined score threshold for completion
        """
        self.objectives = objectives
        
        # Normalize weights
        total_weight = sum(objectives.values())
        self.objectives = {k: v / total_weight for k, v in objectives.items()}
        
        self.sun_direction = sun_direction
        self.earth_direction = earth_direction
        self.target_direction = target_direction
        self.sun_distance_au = sun_distance_au
        self.target_score = target_score
        
        self._scorer: Optional[MissionModeScorer] = None
    
    def _get_scorer(self, swarm: Swarm) -> MissionModeScorer:
        """Get or create scorer for the swarm."""
        if self._scorer is None or self._scorer.swarm is not swarm:
            self._scorer = MissionModeScorer(swarm)
        else:
            self._scorer.refresh()
        return self._scorer
    
    def compute_reward(self, swarm: Swarm, action_taken: Optional[HingeMove] = None) -> float:
        scorer = self._get_scorer(swarm)
        
        weighted_score = 0.0
        
        for mode, weight in self.objectives.items():
            result = scorer.score_configuration(
                mode,
                self.sun_direction,
                self.earth_direction,
                self.target_direction,
                self.sun_distance_au
            )
            weighted_score += weight * result['total_score']
        
        reward = weighted_score
        
        # Bonus for completion
        if self.is_complete(swarm):
            reward += 1.0
        
        return reward
    
    def is_complete(self, swarm: Swarm) -> bool:
        scorer = self._get_scorer(swarm)
        
        weighted_score = 0.0
        
        for mode, weight in self.objectives.items():
            result = scorer.score_configuration(
                mode,
                self.sun_direction,
                self.earth_direction,
                self.target_direction,
                self.sun_distance_au
            )
            weighted_score += weight * result['total_score']
        
        return weighted_score >= self.target_score
    
    def is_failed(self, swarm: Swarm) -> bool:
        return False
    
    def get_task_info(self) -> Dict[str, any]:
        return {
            'task_type': 'multi_objective_mission',
            'objectives': self.objectives,
            'sun_direction': self.sun_direction,
            'earth_direction': self.earth_direction,
            'target_direction': self.target_direction,
            'sun_distance_au': self.sun_distance_au,
            'target_score': self.target_score,
        }
    
    def get_progress(self, swarm: Swarm) -> float:
        scorer = self._get_scorer(swarm)
        
        weighted_score = 0.0
        
        for mode, weight in self.objectives.items():
            result = scorer.score_configuration(
                mode,
                self.sun_direction,
                self.earth_direction,
                self.target_direction,
                self.sun_distance_au
            )
            weighted_score += weight * result['total_score']
        
        return min(1.0, weighted_score / self.target_score)