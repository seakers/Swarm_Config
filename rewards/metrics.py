import numpy as np
from typing import Tuple, Dict, List

from core.cube import Face
from core.cube_faces import FaceFunction, FUNCTION_TO_FACE, FACE_FUNCTION_PROPERTIES
from core.swarm import Swarm


class SwarmMetrics:
    """
    Computes various metrics about swarm configuration.
    
    These metrics are used for reward computation in RL tasks.
    """
    
    def __init__(self, swarm: Swarm):
        self.swarm = swarm
    
    def surface_area(self) -> int:
        """
        Total exposed surface area (number of unbonded faces).
        
        Lower is better for thermal conservation.
        """
        return self.swarm.get_surface_area()
    
    def compactness(self) -> float:
        """
        Ratio of actual connections to maximum possible connections.
        
        For a fully compact cube, this approaches the theoretical maximum.
        Range: 0.0 to 1.0 (higher = more compact)
        """
        n = self.swarm.num_cubes
        actual_connections = len(self.swarm._connections)
        
        # Maximum connections for n cubes depends on shape
        # For a perfect cube, it's approximately 3n - 3*n^(2/3)
        # We'll use a simpler upper bound: 3n (each cube has up to 6 faces, shared)
        max_theoretical = 3 * n
        
        return actual_connections / max_theoretical if max_theoretical > 0 else 0.0
    
    def maximum_baseline(self) -> float:
        """
        Maximum distance between any two cubes.
        
        Higher is better for sparse aperture imaging.
        """
        return self.swarm.get_maximum_extent()
    
    def planar_coverage(self, normal: Tuple[float, float, float]) -> float:
        """
        Projected area onto a plane with given normal.
        
        Higher is better for antenna arrays pointed in that direction.
        """
        return self.swarm.get_planar_area(normal)
    
    def planarity(self, normal: Tuple[float, float, float], tolerance: float = 0.5) -> float:
        """
        How close the swarm is to being a flat plane perpendicular to normal.
        
        Returns fraction of cubes within tolerance of the mean plane.
        Range: 0.0 to 1.0 (higher = more planar)
        """
        normal = np.array(normal, dtype=float)
        normal = normal / np.linalg.norm(normal)
        
        # Project all positions onto normal
        projections = []
        for cube in self.swarm.get_all_cubes():
            pos = np.array(cube.position)
            proj = np.dot(pos, normal)
            projections.append(proj)
        
        if not projections:
            return 1.0
        
        mean_proj = np.mean(projections)
        within_tolerance = sum(1 for p in projections if abs(p - mean_proj) <= tolerance)
        
        return within_tolerance / len(projections)
    
    def linearity(self, axis: Tuple[float, float, float], tolerance: float = 0.5) -> float:
        """
        How close the swarm is to being a straight line along the given axis.
        
        Returns fraction of cubes within tolerance of the central axis.
        Range: 0.0 to 1.0 (higher = more linear)
        """
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        
        # Find centroid
        positions = [np.array(cube.position) for cube in self.swarm.get_all_cubes()]
        if not positions:
            return 1.0
        
        centroid = np.mean(positions, axis=0)
        
        # Calculate distance from each point to the line through centroid
        within_tolerance = 0
        for pos in positions:
            # Vector from centroid to point
            v = pos - centroid
            # Component along axis
            parallel = np.dot(v, axis) * axis
            # Perpendicular component
            perp = v - parallel
            dist = np.linalg.norm(perp)
            
            if dist <= tolerance:
                within_tolerance += 1
        
        return within_tolerance / len(positions)
    
    def connectivity_ratio(self) -> float:
        """
        Fraction of cubes in the largest connected component.
        
        1.0 means fully connected, <1.0 means swarm is split.
        """
        components = self.swarm.get_connected_components()
        if not components:
            return 0.0
        
        largest = max(len(c) for c in components)
        return largest / self.swarm.num_cubes
    
    def num_components(self) -> int:
        """Number of separate connected components."""
        return len(self.swarm.get_connected_components())
    
    def bounding_box_volume(self) -> int:
        """Volume of the axis-aligned bounding box."""
        bounds = self.swarm.get_bounds()
        dims = tuple(bounds[1][i] - bounds[0][i] + 1 for i in range(3))
        return dims[0] * dims[1] * dims[2]
    
    def aspect_ratios(self) -> Tuple[float, float, float]:
        """
        Aspect ratios of the bounding box (sorted largest to smallest).
        
        For a cube: (1.0, 1.0, 1.0)
        For a flat plane: (1.0, 1.0, ~0.0)
        For a line: (1.0, ~0.0, ~0.0)
        """
        bounds = self.swarm.get_bounds()
        dims = [bounds[1][i] - bounds[0][i] + 1 for i in range(3)]
        dims.sort(reverse=True)
        
        max_dim = max(dims) if max(dims) > 0 else 1
        ratios = tuple(d / max_dim for d in dims)
        
        return ratios
    
    def center_of_mass(self) -> Tuple[float, float, float]:
        """Geometric center of all cubes."""
        return self.swarm.get_center_of_mass()
    
    def alignment_score(self, target_direction: Tuple[float, float, float],
                        reference_face: Face = Face.POS_Z) -> float:
        """
        How well a reference face of each cube aligns with a target direction.
        
        Useful for checking if all cameras/antennas are pointed the same way.
        
        Args:
            target_direction: The direction we want cubes to face
            reference_face: Which face of each cube should point at target
            
        Returns:
            Average dot product of face normals with target (range -1 to 1)
        """
        target = np.array(target_direction, dtype=float)
        target = target / np.linalg.norm(target)
        
        total_alignment = 0.0
        count = 0
        
        for cube in self.swarm.get_all_cubes():
            face_normal = cube.orientation.get_global_face_normal(reference_face)
            face_normal = np.array(face_normal, dtype=float)
            alignment = np.dot(face_normal, target)
            total_alignment += alignment
            count += 1
        
        return total_alignment / count if count > 0 else 0.0
    
    def power_distribution_efficiency(self) -> float:
        """
        Measure of how well power can be distributed through the swarm.
        
        Based on average path length between cubes (shorter = better).
        Range: 0.0 to 1.0 (higher = more efficient)
        """
        # Use Floyd-Warshall or BFS to compute all-pairs shortest paths
        # Then compute average path length
        
        cube_ids = list(self.swarm._cubes.keys())
        n = len(cube_ids)
        
        if n <= 1:
            return 1.0
        
        # Build adjacency for BFS
        adjacency = {}
        for cid in cube_ids:
            adjacency[cid] = self.swarm._connections.get_connected_cubes(cid)
        
        # Compute shortest paths from each cube using BFS
        total_path_length = 0
        num_pairs = 0
        
        for start in cube_ids:
            distances = {start: 0}
            queue = [start]
            
            while queue:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            
            for end in cube_ids:
                if end != start and end in distances:
                    total_path_length += distances[end]
                    num_pairs += 1
        
        if num_pairs == 0:
            return 0.0
        
        avg_path_length = total_path_length / num_pairs
        
        # Normalize: for a perfect line of n cubes, avg path is ~n/3
        # For a compact cube, avg path is much smaller
        # We'll use 1 / (1 + avg_path_length/n) as efficiency
        efficiency = 1.0 / (1.0 + avg_path_length / n)
        
        return efficiency
    
    def get_all_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics and return as a dictionary.
        """
        return {
            'surface_area': float(self.surface_area()),
            'compactness': self.compactness(),
            'maximum_baseline': self.maximum_baseline(),
            'connectivity_ratio': self.connectivity_ratio(),
            'num_components': float(self.num_components()),
            'bounding_box_volume': float(self.bounding_box_volume()),
            'power_efficiency': self.power_distribution_efficiency(),
        }
    
# =============================================================================
# Swarm-level face analysis
# =============================================================================

class SwarmFaceAnalyzer:
    """
    Analyzes face exposure and alignment across the entire swarm.
    
    This is critical for evaluating configurations:
    - Are enough solar panels facing the sun?
    - Are antennas exposed and aligned with Earth?
    - Are cameras pointing at the target?
    - Are radiators facing cold space?
    """
    
    def __init__(self, swarm: 'Swarm'):
        """
        Args:
            swarm: The swarm to analyze (should contain EnhancedCube objects)
        """
        self.swarm = swarm
        
        # Cache occupied positions for efficiency
        self._update_occupied_positions()
    
    def _update_occupied_positions(self) -> None:
        """Update the set of occupied positions."""
        self._occupied = set()
        for cube in self.swarm.get_all_cubes():
            self._occupied.add(cube.position)
    
    def refresh(self) -> None:
        """Refresh cached data after swarm changes."""
        self._update_occupied_positions()
    
    # -------------------------------------------------------------------------
    # Exposure analysis
    # -------------------------------------------------------------------------
    
    def count_exposed_function(self, function: FaceFunction) -> int:
        """Count how many cubes have a given function exposed."""
        count = 0
        for cube in self.swarm.get_all_cubes():
            if hasattr(cube, 'is_function_exposed'):
                if cube.is_function_exposed(function, self._occupied):
                    count += 1
            else:
                # Fallback for basic Cube objects - assume standard face assignment
                local_face = FUNCTION_TO_FACE.get(function)
                if local_face:
                    global_dir = cube.orientation.get_global_face_normal(local_face)
                    adj_pos = (
                        cube.position[0] + global_dir[0],
                        cube.position[1] + global_dir[1],
                        cube.position[2] + global_dir[2]
                    )
                    if adj_pos not in self._occupied:
                        count += 1
        return count
    
    def get_exposure_fraction(self, function: FaceFunction) -> float:
        """Get fraction of cubes that have a given function exposed."""
        count = self.count_exposed_function(function)
        total = len(list(self.swarm.get_all_cubes()))
        return count / total if total > 0 else 0.0
    
    def get_all_exposure_fractions(self) -> Dict[FaceFunction, float]:
        """Get exposure fractions for all face functions."""
        return {
            func: self.get_exposure_fraction(func)
            for func in FaceFunction
        }
    
    def get_cubes_with_function_exposed(self, function: FaceFunction) -> List[int]:
        """Get list of cube IDs that have a given function exposed."""
        cube_ids = []
        for cube in self.swarm.get_all_cubes():
            if hasattr(cube, 'is_function_exposed'):
                if cube.is_function_exposed(function, self._occupied):
                    cube_ids.append(cube.cube_id)
            else:
                local_face = FUNCTION_TO_FACE.get(function)
                if local_face:
                    global_dir = cube.orientation.get_global_face_normal(local_face)
                    adj_pos = (
                        cube.position[0] + global_dir[0],
                        cube.position[1] + global_dir[1],
                        cube.position[2] + global_dir[2]
                    )
                    if adj_pos not in self._occupied:
                        cube_ids.append(cube.cube_id)
        return cube_ids
    
    # -------------------------------------------------------------------------
    # Alignment analysis
    # -------------------------------------------------------------------------
    
    def compute_function_alignment(self, function: FaceFunction,
                                   target_direction: Tuple[float, float, float],
                                   only_exposed: bool = True) -> Dict[str, float]:
        """
        Compute alignment statistics for a function across all cubes.
        
        Args:
            function: The function to analyze
            target_direction: Direction the function should point toward
            only_exposed: If True, only consider cubes where the function is exposed
            
        Returns:
            Dictionary with alignment statistics
        """
        alignments = []
        local_face = FUNCTION_TO_FACE.get(function)
        
        if local_face is None:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        target = np.array(target_direction, dtype=float)
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-10:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        target = target / target_norm
        
        for cube in self.swarm.get_all_cubes():
            # Check exposure if required
            if only_exposed:
                global_dir = cube.orientation.get_global_face_normal(local_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                if adj_pos in self._occupied:
                    continue  # Skip blocked faces
            
            # Compute alignment
            global_dir = cube.orientation.get_global_face_normal(local_face)
            face_vec = np.array(global_dir, dtype=float)
            alignment = float(np.dot(face_vec, target))
            alignments.append(alignment)
        
        if not alignments:
            return {'mean': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        
        return {
            'mean': float(np.mean(alignments)),
            'min': float(np.min(alignments)),
            'max': float(np.max(alignments)),
            'std': float(np.std(alignments)),
            'count': len(alignments),
            'well_aligned_count': sum(1 for a in alignments if a > 0.9),
            'poorly_aligned_count': sum(1 for a in alignments if a < 0.1),
        }
    
    def compute_solar_array_efficiency(self, sun_direction: Tuple[float, float, float]) -> float:
        """
        Compute total solar array efficiency.
        
        Efficiency = sum of (alignment * exposure) for all solar faces.
        Alignment of 1.0 means face is directly pointed at sun.
        
        Args:
            sun_direction: Direction from swarm to sun
            
        Returns:
            Total effective solar array area as fraction of maximum
        """
        total_efficiency = 0.0
        num_cubes = 0
        
        local_face = FUNCTION_TO_FACE.get(FaceFunction.SOLAR_ARRAY)
        if local_face is None:
            return 0.0
        
        sun = np.array(sun_direction, dtype=float)
        sun_norm = np.linalg.norm(sun)
        if sun_norm < 1e-10:
            return 0.0
        sun = sun / sun_norm
        
        for cube in self.swarm.get_all_cubes():
            num_cubes += 1
            
            # Check if solar face is exposed
            global_dir = cube.orientation.get_global_face_normal(local_face)
            adj_pos = (
                cube.position[0] + global_dir[0],
                cube.position[1] + global_dir[1],
                cube.position[2] + global_dir[2]
            )
            
            if adj_pos in self._occupied:
                continue  # Blocked
            
            # Compute alignment (only positive values count - can't generate power from back)
            face_vec = np.array(global_dir, dtype=float)
            alignment = float(np.dot(face_vec, sun))
            
            if alignment > 0:
                total_efficiency += alignment
        
        return total_efficiency / num_cubes if num_cubes > 0 else 0.0
    
    def compute_antenna_effectiveness(self, earth_direction: Tuple[float, float, float],
                                      function: FaceFunction = FaceFunction.ANTENNA_HIGH_GAIN) -> Dict[str, float]:
        """
        Compute antenna effectiveness for Earth communication.
        
        For phased array operation, we want:
        - Many antennas exposed
        - All pointing roughly toward Earth
        - Good spatial distribution for interference
        
        Args:
            earth_direction: Direction from swarm to Earth
            function: Which antenna function to analyze
            
        Returns:
            Dictionary with effectiveness metrics
        """
        exposed_cubes = self.get_cubes_with_function_exposed(function)
        
        if not exposed_cubes:
            return {
                'effective_aperture': 0.0,
                'num_active': 0,
                'mean_alignment': 0.0,
                'coherent_gain': 0.0,
            }
        
        earth = np.array(earth_direction, dtype=float)
        earth_norm = np.linalg.norm(earth)
        if earth_norm < 1e-10:
            return {
                'effective_aperture': 0.0,
                'num_active': len(exposed_cubes),
                'mean_alignment': 0.0,
                'coherent_gain': 0.0,
            }
        earth = earth / earth_norm
        
        local_face = FUNCTION_TO_FACE.get(function)
        
        alignments = []
        positions = []
        
        for cube_id in exposed_cubes:
            cube = self.swarm.get_cube(cube_id)
            if cube is None:
                continue
            
            global_dir = cube.orientation.get_global_face_normal(local_face)
            face_vec = np.array(global_dir, dtype=float)
            alignment = float(np.dot(face_vec, earth))
            
            # Only count antennas pointing roughly toward Earth
            if alignment > 0:
                alignments.append(alignment)
                positions.append(np.array(cube.position))
        
        if not alignments:
            return {
                'effective_aperture': 0.0,
                'num_active': 0,
                'mean_alignment': 0.0,
                'coherent_gain': 0.0,
            }
        
        mean_alignment = np.mean(alignments)
        
        # Effective aperture: sum of alignment-weighted faces
        effective_aperture = sum(alignments)
        
        # Coherent gain: for phased array, gain scales with N^2 for N elements
        # but only if they're all aligned
        # Simplified model: gain = (sum of alignments)^2 / N
        coherent_gain = (sum(alignments) ** 2) / len(alignments)
        
        # Spatial extent (for beam forming)
        if len(positions) > 1:
            positions_arr = np.array(positions)
            spatial_extent = np.max(np.ptp(positions_arr, axis=0))
        else:
            spatial_extent = 0.0
        
        return {
            'effective_aperture': effective_aperture,
            'num_active': len(alignments),
            'mean_alignment': mean_alignment,
            'coherent_gain': coherent_gain,
            'spatial_extent': spatial_extent,
        }
    
    def compute_camera_coverage(self, target_direction: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Compute camera coverage for observation tasks.
        
        For sparse aperture synthesis, we want:
        - Many cameras exposed and pointing at target
        - Maximum spatial separation (baseline)
        
        Args:
            target_direction: Direction from swarm to observation target
            
        Returns:
            Dictionary with coverage metrics
        """
        exposed_cubes = self.get_cubes_with_function_exposed(FaceFunction.CAMERA)
        
        if not exposed_cubes:
            return {
                'num_active': 0,
                'mean_alignment': 0.0,
                'max_baseline': 0.0,
                'effective_aperture': 0.0,
            }
        
        target = np.array(target_direction, dtype=float)
        target_norm = np.linalg.norm(target)
        if target_norm < 1e-10:
            return {
                'num_active': len(exposed_cubes),
                'mean_alignment': 0.0,
                'max_baseline': 0.0,
                'effective_aperture': 0.0,
            }
        target = target / target_norm
        
        local_face = FUNCTION_TO_FACE.get(FaceFunction.CAMERA)
        
        aligned_positions = []
        alignments = []
        
        for cube_id in exposed_cubes:
            cube = self.swarm.get_cube(cube_id)
            if cube is None:
                continue
            
            global_dir = cube.orientation.get_global_face_normal(local_face)
            face_vec = np.array(global_dir, dtype=float)
            alignment = float(np.dot(face_vec, target))
            
            if alignment > 0.5:  # Camera needs to be reasonably aligned
                alignments.append(alignment)
                aligned_positions.append(np.array(cube.position))
        
        if not alignments:
            return {
                'num_active': 0,
                'mean_alignment': 0.0,
                'max_baseline': 0.0,
                'effective_aperture': 0.0,
            }
        
        mean_alignment = np.mean(alignments)
        
        # Maximum baseline for aperture synthesis
        max_baseline = 0.0
        for i, p1 in enumerate(aligned_positions):
            for p2 in aligned_positions[i+1:]:
                baseline = np.linalg.norm(p1 - p2)
                max_baseline = max(max_baseline, baseline)
        
        # Effective aperture (for sparse aperture, this is about coverage, not fill)
        effective_aperture = len(alignments) * mean_alignment
        
        return {
            'num_active': len(alignments),
            'mean_alignment': mean_alignment,
            'max_baseline': max_baseline,
            'effective_aperture': effective_aperture,
        }
    
    def compute_thermal_balance(self, sun_direction: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Compute thermal balance metrics for the swarm.
        
        Thermal balance depends on:
        - Solar input (faces exposed to sun)
        - Radiator effectiveness (radiators facing away from sun)
        - Internal heat generation
        - Heat sharing between connected cubes
        
        Args:
            sun_direction: Direction from swarm to sun
            
        Returns:
            Dictionary with thermal metrics
        """
        sun = np.array(sun_direction, dtype=float)
        sun_norm = np.linalg.norm(sun)
        if sun_norm < 1e-10:
            sun = np.array([0, 0, 1])  # Default
        else:
            sun = sun / sun_norm
        
        # Cold space is opposite to sun (simplified)
        cold_space = -sun
        
        solar_face = FUNCTION_TO_FACE.get(FaceFunction.SOLAR_ARRAY)
        radiator_face = FUNCTION_TO_FACE.get(FaceFunction.RADIATOR)
        
        total_solar_input = 0.0
        total_radiator_output = 0.0
        total_internal_heat = 0.0
        
        cubes_too_hot = 0
        cubes_too_cold = 0
        cubes_ok = 0
        
        for cube in self.swarm.get_all_cubes():
            # --- Solar input ---
            if solar_face:
                global_dir = cube.orientation.get_global_face_normal(solar_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                
                if adj_pos not in self._occupied:
                    # Face is exposed
                    face_vec = np.array(global_dir, dtype=float)
                    sun_alignment = float(np.dot(face_vec, sun))
                    
                    if sun_alignment > 0:
                        # Solar input proportional to alignment
                        # Assume ~2W per face at 1 AU when directly facing sun
                        solar_input = sun_alignment * 2.0
                        total_solar_input += solar_input
            
            # --- Radiator output ---
            if radiator_face:
                global_dir = cube.orientation.get_global_face_normal(radiator_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                
                if adj_pos not in self._occupied:
                    # Face is exposed
                    face_vec = np.array(global_dir, dtype=float)
                    
                    # Radiator works best when facing cold space (away from sun)
                    cold_alignment = float(np.dot(face_vec, cold_space))
                    
                    if cold_alignment > 0:
                        # Heat rejection proportional to alignment with cold space
                        # Assume ~5W max rejection per radiator face
                        radiator_output = cold_alignment * 5.0
                        total_radiator_output += radiator_output
                    elif cold_alignment < 0:
                        # Radiator facing sun - actually absorbs heat!
                        sun_alignment = -cold_alignment
                        total_solar_input += sun_alignment * 1.0  # Less than solar panel
            
            # --- Internal heat generation ---
            if hasattr(cube, 'subsystems'):
                internal = cube.subsystems.thermal.base_heat_generation
            else:
                internal = 2.0  # Default assumption
            total_internal_heat += internal
            
            # --- Check thermal state ---
            if hasattr(cube, 'subsystems'):
                temp = cube.subsystems.thermal.temperature
                if temp > cube.subsystems.thermal.max_operating_temp:
                    cubes_too_hot += 1
                elif temp < cube.subsystems.thermal.min_operating_temp:
                    cubes_too_cold += 1
                else:
                    cubes_ok += 1
            else:
                cubes_ok += 1  # Assume OK if no subsystem data
        
        # Net heat balance
        net_heat = total_solar_input + total_internal_heat - total_radiator_output
        
        num_cubes = len(list(self.swarm.get_all_cubes()))
        
        return {
            'total_solar_input_watts': total_solar_input,
            'total_radiator_output_watts': total_radiator_output,
            'total_internal_heat_watts': total_internal_heat,
            'net_heat_watts': net_heat,
            'net_heat_per_cube_watts': net_heat / num_cubes if num_cubes > 0 else 0,
            'cubes_too_hot': cubes_too_hot,
            'cubes_too_cold': cubes_too_cold,
            'cubes_thermal_ok': cubes_ok,
            'thermal_ok_fraction': cubes_ok / num_cubes if num_cubes > 0 else 0,
        }
    
    def compute_power_balance(self, sun_direction: Tuple[float, float, float],
                              sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Compute power balance for the swarm.
        
        Power generation from solar panels decreases with square of distance from sun.
        
        Args:
            sun_direction: Direction from swarm to sun
            sun_distance_au: Distance from sun in AU (1 AU = Earth's distance)
            
        Returns:
            Dictionary with power metrics
        """
        # Solar power scales with 1/r^2
        solar_intensity_factor = 1.0 / (sun_distance_au ** 2)
        
        # Base solar array output at 1 AU
        base_power_per_face = FACE_FUNCTION_PROPERTIES[FaceFunction.SOLAR_ARRAY].power_generation
        
        sun = np.array(sun_direction, dtype=float)
        sun_norm = np.linalg.norm(sun)
        if sun_norm > 1e-10:
            sun = sun / sun_norm
        
        solar_face = FUNCTION_TO_FACE.get(FaceFunction.SOLAR_ARRAY)
        
        total_power_generation = 0.0
        total_power_consumption = 0.0
        num_solar_exposed = 0
        num_solar_well_aligned = 0
        
        for cube in self.swarm.get_all_cubes():
            # --- Power generation from solar arrays ---
            if solar_face:
                global_dir = cube.orientation.get_global_face_normal(solar_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                
                if adj_pos not in self._occupied:
                    num_solar_exposed += 1
                    face_vec = np.array(global_dir, dtype=float)
                    sun_alignment = float(np.dot(face_vec, sun))
                    
                    if sun_alignment > 0:
                        power = base_power_per_face * sun_alignment * solar_intensity_factor
                        total_power_generation += power
                        
                        if sun_alignment > 0.9:
                            num_solar_well_aligned += 1
            
            # --- Power consumption ---
            if hasattr(cube, 'subsystems'):
                # Base consumption from all systems
                # In a real implementation, this would depend on what's active
                base_consumption = 2.0  # Watts for basic systems
                total_power_consumption += base_consumption
            else:
                total_power_consumption += 2.0  # Default
        
        net_power = total_power_generation - total_power_consumption
        num_cubes = len(list(self.swarm.get_all_cubes()))
        
        return {
            'total_generation_watts': total_power_generation,
            'total_consumption_watts': total_power_consumption,
            'net_power_watts': net_power,
            'net_power_per_cube_watts': net_power / num_cubes if num_cubes > 0 else 0,
            'solar_intensity_factor': solar_intensity_factor,
            'num_solar_exposed': num_solar_exposed,
            'num_solar_well_aligned': num_solar_well_aligned,
            'solar_exposure_fraction': num_solar_exposed / num_cubes if num_cubes > 0 else 0,
            'power_positive': net_power > 0,
        }
    
    def get_configuration_summary(self, 
                                   sun_direction: Tuple[float, float, float] = (0, 0, -1),
                                   earth_direction: Tuple[float, float, float] = (1, 0, 0),
                                   target_direction: Tuple[float, float, float] = (0, 1, 0),
                                   sun_distance_au: float = 10.0) -> Dict[str, any]:
        """
        Get a comprehensive summary of the swarm configuration.
        
        This combines all face analysis metrics into one report.
        
        Args:
            sun_direction: Direction to the sun
            earth_direction: Direction to Earth
            target_direction: Direction to science target
            sun_distance_au: Distance from sun in AU
            
        Returns:
            Comprehensive configuration summary
        """
        self.refresh()  # Ensure cached data is current
        
        summary = {
            # Exposure fractions
            'exposure': self.get_all_exposure_fractions(),
            
            # Power analysis
            'power': self.compute_power_balance(sun_direction, sun_distance_au),
            
            # Thermal analysis
            'thermal': self.compute_thermal_balance(sun_direction),
            
            # Communication analysis
            'communication': self.compute_antenna_effectiveness(earth_direction),
            
            # Science/camera analysis
            'observation': self.compute_camera_coverage(target_direction),
            
            # Alignment scores for key functions
            'alignments': {
                'solar_to_sun': self.compute_function_alignment(
                    FaceFunction.SOLAR_ARRAY, sun_direction
                ),
                'antenna_to_earth': self.compute_function_alignment(
                    FaceFunction.ANTENNA_HIGH_GAIN, earth_direction
                ),
                'camera_to_target': self.compute_function_alignment(
                    FaceFunction.CAMERA, target_direction
                ),
                'radiator_to_cold_space': self.compute_function_alignment(
                    FaceFunction.RADIATOR, tuple(-np.array(sun_direction))
                ),
            },
        }
        
        return summary
    
# =============================================================================
# Configuration scoring for different mission modes
# =============================================================================

class MissionModeScorer:
    """
    Scores swarm configurations for different mission modes.
    
    Each mission mode has different priorities for face exposure and alignment.
    """
    
    def __init__(self, swarm: 'Swarm'):
        self.swarm = swarm
        self.analyzer = SwarmFaceAnalyzer(swarm)
    
    def refresh(self) -> None:
        """Refresh after swarm changes."""
        self.analyzer.refresh()
    
    def score_earth_communication_mode(self,
                                        earth_direction: Tuple[float, float, float],
                                        sun_direction: Tuple[float, float, float],
                                        sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Score configuration for Earth communication.
        
        Priorities:
        1. Maximum antenna exposure and alignment with Earth
        2. Sufficient power generation
        3. Thermal balance maintained
        
        Args:
            earth_direction: Direction to Earth
            sun_direction: Direction to sun
            sun_distance_au: Distance from sun
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        # Antenna effectiveness (primary)
        antenna = self.analyzer.compute_antenna_effectiveness(earth_direction)
        antenna_score = min(1.0, antenna['effective_aperture'] / self.swarm.num_cubes)
        
        # Coherent gain bonus (phased array operation)
        coherent_bonus = min(0.2, antenna['coherent_gain'] / (self.swarm.num_cubes ** 2) * 0.2)
        
        # Power balance (secondary)
        power = self.analyzer.compute_power_balance(sun_direction, sun_distance_au)
        if power['net_power_watts'] >= 0:
            power_score = 1.0
        else:
            # Penalty for negative power balance
            power_score = max(0, 1.0 + power['net_power_watts'] / 100.0)
        
        # Thermal balance (secondary)
        thermal = self.analyzer.compute_thermal_balance(sun_direction)
        thermal_score = thermal['thermal_ok_fraction']
        
        # Combined score
        total_score = (
            0.5 * antenna_score +
            0.1 * coherent_bonus +
            0.25 * power_score +
            0.15 * thermal_score
        )
        
        return {
            'total_score': total_score,
            'antenna_score': antenna_score,
            'coherent_bonus': coherent_bonus,
            'power_score': power_score,
            'thermal_score': thermal_score,
            'antenna_details': antenna,
            'power_details': power,
            'thermal_details': thermal,
        }
    
    def score_science_observation_mode(self,
                                       target_direction: Tuple[float, float, float],
                                       sun_direction: Tuple[float, float, float],
                                       sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Score configuration for science observation.
        
        Priorities:
        1. Maximum camera exposure and alignment with target
        2. Maximum baseline for aperture synthesis
        3. Sufficient power
        4. Science instruments exposed
        
        Args:
            target_direction: Direction to observation target
            sun_direction: Direction to sun
            sun_distance_au: Distance from sun
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        # Camera coverage (primary)
        camera = self.analyzer.compute_camera_coverage(target_direction)
        camera_score = min(1.0, camera['num_active'] / self.swarm.num_cubes)
        
        # Baseline bonus (for aperture synthesis)
        max_possible_baseline = np.sqrt(3) * (self.swarm.num_cubes ** (1/3))  # Rough estimate
        baseline_score = min(0.2, camera['max_baseline'] / max_possible_baseline * 0.2)
        
        # Science instruments exposure
        science_exposure = self.analyzer.get_exposure_fraction(FaceFunction.SCIENCE_INSTRUMENTS)
        science_score = science_exposure
        
        # Power balance
        power = self.analyzer.compute_power_balance(sun_direction, sun_distance_au)
        if power['net_power_watts'] >= 0:
            power_score = 1.0
        else:
            power_score = max(0, 1.0 + power['net_power_watts'] / 100.0)
        
        # Combined score
        total_score = (
            0.4 * camera_score +
            0.15 * baseline_score +
            0.2 * science_score +
            0.25 * power_score
        )
        
        return {
            'total_score': total_score,
            'camera_score': camera_score,
            'baseline_score': baseline_score,
            'science_score': science_score,
            'power_score': power_score,
            'camera_details': camera,
            'power_details': power,
        }
    
    def score_cruise_mode(self,
                          sun_direction: Tuple[float, float, float],
                          sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Score configuration for cruise mode (power conservation).
        
        Priorities:
        1. Minimum surface area (thermal efficiency)
        2. Sufficient power generation
        3. Thermal balance (avoid getting too cold)
        4. Compactness
        
        Args:
            sun_direction: Direction to sun
            sun_distance_au: Distance from sun
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        # Surface area (lower is better)
        surface_area = self.swarm.get_surface_area()
        # Minimum possible for n cubes in a cube shape
        n = self.swarm.num_cubes
        side = round(n ** (1/3))
        min_surface = 6 * (side ** 2)
        max_surface = 6 * n  # All separate
        
        surface_score = 1.0 - (surface_area - min_surface) / (max_surface - min_surface)
        
        # Power balance (just need to break even)
        power = self.analyzer.compute_power_balance(sun_direction, sun_distance_au)
        if power['net_power_watts'] >= 0:
            power_score = 1.0
        else:
            # More tolerance for slightly negative power in cruise mode
            power_score = max(0, 1.0 + power['net_power_watts'] / 50.0)
        
        # Thermal balance
        thermal = self.analyzer.compute_thermal_balance(sun_direction)
        thermal_score = thermal['thermal_ok_fraction']
        
        # Extra bonus for compact shape (bounding box)
        bounds = self.swarm.get_bounds()
        dims = [bounds[1][i] - bounds[0][i] + 1 for i in range(3)]
        volume = dims[0] * dims[1] * dims[2]
        compactness = n / volume  # Packing efficiency
        compactness_score = compactness
        
        # Combined score
        total_score = (
            0.35 * surface_score +
            0.25 * power_score +
            0.2 * thermal_score +
            0.2 * compactness_score
        )
        
        return {
            'total_score': total_score,
            'surface_score': surface_score,
            'power_score': power_score,
            'thermal_score': thermal_score,
            'compactness_score': compactness_score,
            'surface_area': surface_area,
            'min_surface_area': min_surface,
            'power_details': power,
            'thermal_details': thermal,
        }
    
    def score_solar_charging_mode(self,
                                   sun_direction: Tuple[float, float, float],
                                   sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Score configuration for maximum solar power collection.
        
        Priorities:
        1. Maximum solar array exposure and alignment
        2. Thermal management (don't overheat)
        
        Args:
            sun_direction: Direction to sun
            sun_distance_au: Distance from sun
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        # Solar array effectiveness
        solar_efficiency = self.analyzer.compute_solar_array_efficiency(sun_direction)
        solar_score = solar_efficiency
        
        # Power generation
        power = self.analyzer.compute_power_balance(sun_direction, sun_distance_au)
        generation_score = min(1.0, power['total_generation_watts'] / 
                               (self.swarm.num_cubes * 2.0))  # 2W max per cube
        
        # Fraction of solar panels well-aligned
        well_aligned_fraction = (power['num_solar_well_aligned'] / 
                                 self.swarm.num_cubes if self.swarm.num_cubes > 0 else 0)
        
        # Thermal management (don't want to overheat from all that sun exposure)
        thermal = self.analyzer.compute_thermal_balance(sun_direction)
        
        # Penalize if too much heat buildup
        if thermal['net_heat_watts'] > 0:
            # Some heat buildup is OK, but too much is bad
            heat_penalty = min(0.3, thermal['net_heat_per_cube_watts'] / 10.0 * 0.3)
            thermal_score = 1.0 - heat_penalty
        else:
            thermal_score = 1.0
        
        # Also check radiator exposure (helps with thermal)
        radiator_exposure = self.analyzer.get_exposure_fraction(FaceFunction.RADIATOR)
        
        # Combined score
        total_score = (
            0.5 * solar_score +
            0.2 * generation_score +
            0.15 * thermal_score +
            0.15 * well_aligned_fraction
        )
        
        return {
            'total_score': total_score,
            'solar_score': solar_score,
            'generation_score': generation_score,
            'thermal_score': thermal_score,
            'well_aligned_fraction': well_aligned_fraction,
            'radiator_exposure': radiator_exposure,
            'power_details': power,
            'thermal_details': thermal,
        }
    
    def score_thermal_emergency_mode(self,
                                      sun_direction: Tuple[float, float, float],
                                      threat_direction: Tuple[float, float, float]) -> Dict[str, float]:
        """
        Score configuration for thermal emergency (e.g., close solar approach).
        
        Priorities:
        1. Minimize sun-facing surface area
        2. Maximize radiator exposure to cold space
        3. Shield sensitive instruments
        
        Args:
            sun_direction: Direction to sun (heat source)
            threat_direction: Direction of thermal threat (may be same as sun)
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        sun = np.array(sun_direction, dtype=float)
        sun_norm = np.linalg.norm(sun)
        if sun_norm > 1e-10:
            sun = sun / sun_norm
        
        threat = np.array(threat_direction, dtype=float)
        threat_norm = np.linalg.norm(threat)
        if threat_norm > 1e-10:
            threat = threat / threat_norm
        
        cold_space = -sun  # Opposite to sun
        
        # Count how many faces of each type are exposed to threat
        num_cubes = len(list(self.swarm.get_all_cubes()))
        
        solar_facing_threat = 0
        radiator_facing_cold = 0
        camera_exposed_to_threat = 0
        science_exposed_to_threat = 0
        
        solar_face = FUNCTION_TO_FACE.get(FaceFunction.SOLAR_ARRAY)
        radiator_face = FUNCTION_TO_FACE.get(FaceFunction.RADIATOR)
        camera_face = FUNCTION_TO_FACE.get(FaceFunction.CAMERA)
        science_face = FUNCTION_TO_FACE.get(FaceFunction.SCIENCE_INSTRUMENTS)
        
        for cube in self.swarm.get_all_cubes():
            # Check solar panel alignment with threat (want this - use panels as shield)
            if solar_face:
                global_dir = cube.orientation.get_global_face_normal(solar_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                if adj_pos not in self.analyzer._occupied:
                    face_vec = np.array(global_dir, dtype=float)
                    if np.dot(face_vec, threat) > 0.5:
                        solar_facing_threat += 1
            
            # Check radiator facing cold space (good for heat rejection)
            if radiator_face:
                global_dir = cube.orientation.get_global_face_normal(radiator_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                if adj_pos not in self.analyzer._occupied:
                    face_vec = np.array(global_dir, dtype=float)
                    if np.dot(face_vec, cold_space) > 0.5:
                        radiator_facing_cold += 1
            
            # Check camera exposure to threat (bad - cameras are sensitive)
            if camera_face:
                global_dir = cube.orientation.get_global_face_normal(camera_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                if adj_pos not in self.analyzer._occupied:
                    face_vec = np.array(global_dir, dtype=float)
                    if np.dot(face_vec, threat) > 0.5:
                        camera_exposed_to_threat += 1
            
            # Check science instruments exposure (also sensitive)
            if science_face:
                global_dir = cube.orientation.get_global_face_normal(science_face)
                adj_pos = (
                    cube.position[0] + global_dir[0],
                    cube.position[1] + global_dir[1],
                    cube.position[2] + global_dir[2]
                )
                if adj_pos not in self.analyzer._occupied:
                    face_vec = np.array(global_dir, dtype=float)
                    if np.dot(face_vec, threat) > 0.5:
                        science_exposed_to_threat += 1
        
        # Score components
        # Solar panels as shields (higher is better)
        shield_score = solar_facing_threat / num_cubes if num_cubes > 0 else 0
        
        # Radiators facing cold space (higher is better)
        radiator_score = radiator_facing_cold / num_cubes if num_cubes > 0 else 0
        
        # Sensitive equipment protected (lower exposure is better)
        camera_protection = 1.0 - (camera_exposed_to_threat / num_cubes if num_cubes > 0 else 0)
        science_protection = 1.0 - (science_exposed_to_threat / num_cubes if num_cubes > 0 else 0)
        
        # Compactness helps (less total surface area exposed)
        surface_area = self.swarm.get_surface_area()
        side = round(num_cubes ** (1/3))
        min_surface = 6 * (side ** 2)
        max_surface = 6 * num_cubes
        compactness_score = 1.0 - (surface_area - min_surface) / (max_surface - min_surface)
        
        # Combined score
        total_score = (
            0.25 * shield_score +
            0.25 * radiator_score +
            0.2 * camera_protection +
            0.15 * science_protection +
            0.15 * compactness_score
        )
        
        return {
            'total_score': total_score,
            'shield_score': shield_score,
            'radiator_score': radiator_score,
            'camera_protection': camera_protection,
            'science_protection': science_protection,
            'compactness_score': compactness_score,
            'solar_facing_threat': solar_facing_threat,
            'radiator_facing_cold': radiator_facing_cold,
            'camera_exposed': camera_exposed_to_threat,
            'science_exposed': science_exposed_to_threat,
        }
    
    def score_distributed_sensing_mode(self,
                                        sun_direction: Tuple[float, float, float],
                                        sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Score configuration for distributed in-situ sensing.
        
        Priorities:
        1. Maximum science instrument exposure
        2. Maximum spatial spread (for field measurements)
        3. Inter-satellite communication maintained
        4. Sufficient power
        
        Args:
            sun_direction: Direction to sun
            sun_distance_au: Distance from sun
            
        Returns:
            Score breakdown
        """
        self.refresh()
        
        # Science instrument exposure
        science_exposure = self.analyzer.get_exposure_fraction(FaceFunction.SCIENCE_INSTRUMENTS)
        
        # Spatial spread (convex hull volume)
        try:
            hull_volume = self.swarm.get_convex_hull_volume()
            # Normalize by maximum possible (rough estimate)
            max_volume = self.swarm.num_cubes ** 1.5  # Rough scaling
            spread_score = min(1.0, hull_volume / max_volume)
        except:
            # Fallback to maximum extent
            extent = self.swarm.get_maximum_extent()
            spread_score = min(1.0, extent / (self.swarm.num_cubes ** 0.5))
        
        # Inter-satellite link exposure
        isl_exposure = self.analyzer.get_exposure_fraction(FaceFunction.ANTENNA_INTER_SAT)
        
        # Check connectivity (need to be able to share data)
        components = self.swarm.get_connected_components()
        connectivity_score = 1.0 / len(components) if components else 0  # 1.0 if fully connected
        
        # Power balance
        power = self.analyzer.compute_power_balance(sun_direction, sun_distance_au)
        if power['net_power_watts'] >= 0:
            power_score = 1.0
        else:
            power_score = max(0, 1.0 + power['net_power_watts'] / 100.0)
        
        # Combined score
        total_score = (
            0.3 * science_exposure +
            0.25 * spread_score +
            0.2 * isl_exposure +
            0.1 * connectivity_score +
            0.15 * power_score
        )
        
        return {
            'total_score': total_score,
            'science_exposure': science_exposure,
            'spread_score': spread_score,
            'isl_exposure': isl_exposure,
            'connectivity_score': connectivity_score,
            'power_score': power_score,
            'num_components': len(components),
            'power_details': power,
        }
    
    def score_configuration(self, 
                            mode: str,
                            sun_direction: Tuple[float, float, float] = (0, 0, -1),
                            earth_direction: Tuple[float, float, float] = (1, 0, 0),
                            target_direction: Tuple[float, float, float] = (0, 1, 0),
                            sun_distance_au: float = 10.0) -> Dict[str, any]:
        """
        Score configuration for a given mission mode.
        
        Args:
            mode: One of 'communication', 'observation', 'cruise', 
                  'charging', 'thermal_emergency', 'distributed_sensing'
            sun_direction: Direction to sun
            earth_direction: Direction to Earth  
            target_direction: Direction to science target
            sun_distance_au: Distance from sun in AU
            
        Returns:
            Score breakdown for the specified mode
        """
        mode = mode.lower().replace(' ', '_')
        
        if mode in ['communication', 'earth_comm', 'downlink', 'uplink']:
            return self.score_earth_communication_mode(
                earth_direction, sun_direction, sun_distance_au
            )
        
        elif mode in ['observation', 'science', 'imaging', 'camera']:
            return self.score_science_observation_mode(
                target_direction, sun_direction, sun_distance_au
            )
        
        elif mode in ['cruise', 'coast', 'sleep', 'hibernate']:
            return self.score_cruise_mode(sun_direction, sun_distance_au)
        
        elif mode in ['charging', 'solar', 'power']:
            return self.score_solar_charging_mode(sun_direction, sun_distance_au)
        
        elif mode in ['thermal_emergency', 'thermal', 'emergency', 'shield']:
            return self.score_thermal_emergency_mode(sun_direction, sun_direction)
        
        elif mode in ['distributed_sensing', 'sensing', 'field', 'in_situ']:
            return self.score_distributed_sensing_mode(sun_direction, sun_distance_au)
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Valid modes are: "
                           f"communication, observation, cruise, charging, "
                           f"thermal_emergency, distributed_sensing")
    
    def get_all_mode_scores(self,
                            sun_direction: Tuple[float, float, float] = (0, 0, -1),
                            earth_direction: Tuple[float, float, float] = (1, 0, 0),
                            target_direction: Tuple[float, float, float] = (0, 1, 0),
                            sun_distance_au: float = 10.0) -> Dict[str, float]:
        """
        Get scores for all mission modes.
        
        Useful for understanding how well-suited the current configuration
        is for different mission phases.
        
        Returns:
            Dictionary mapping mode name to total score
        """
        modes = [
            'communication',
            'observation', 
            'cruise',
            'charging',
            'thermal_emergency',
            'distributed_sensing'
        ]
        
        scores = {}
        for mode in modes:
            result = self.score_configuration(
                mode, sun_direction, earth_direction, 
                target_direction, sun_distance_au
            )
            scores[mode] = result['total_score']
        
        # Also identify best mode for current configuration
        best_mode = max(scores, key=scores.get)
        scores['best_mode'] = best_mode
        scores['best_score'] = scores[best_mode]
        
        return scores