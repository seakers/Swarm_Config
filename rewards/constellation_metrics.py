import numpy as np
from typing import Dict, List, Tuple, Optional, Set

from core.constellation import Constellation


class ConstellationMetrics:
    """
    Metrics for evaluating constellation configurations.
    
    Extends SwarmMetrics to account for separated groups.
    """
    
    def __init__(self, constellation: Constellation):
        self.constellation = constellation
    
    def get_effective_aperture(self) -> float:
        """
        Get effective aperture for interferometry/sparse aperture imaging.
        
        This is based on the maximum baseline between groups.
        """
        return self.constellation.get_max_baseline()
    
    def get_coverage_volume(self) -> float:
        """
        Get the volume covered by the constellation for in-situ measurements.
        
        Uses convex hull of group positions.
        """
        from scipy.spatial import ConvexHull
        
        positions = [g.position for g in self.constellation._groups.values()]
        
        if len(positions) < 4:
            # Can't form a 3D hull, estimate from max distances
            if len(positions) < 2:
                return 0.0
            
            max_dist = 0.0
            for i, p1 in enumerate(positions):
                for p2 in positions[i+1:]:
                    max_dist = max(max_dist, np.linalg.norm(p1 - p2))
            
            return max_dist ** 3 / 6.0  # Rough sphere approximation
        
        try:
            points = np.array(positions)
            hull = ConvexHull(points)
            return hull.volume
        except:
            return 0.0
    
    def get_formation_quality(self, target_formation: str = 'line') -> float:
        """
        Evaluate how well the constellation matches a target formation.
        
        Args:
            target_formation: 'line', 'plane', 'tetrahedron', etc.
            
        Returns:
            Quality score from 0 to 1
        """
        positions = np.array([g.position for g in self.constellation._groups.values()])
        
        if len(positions) < 2:
            return 1.0 if target_formation == 'point' else 0.0
        
        if target_formation == 'line':
            # Check how well positions fit a line (PCA)
            centered = positions - np.mean(positions, axis=0)
            
            if len(centered) < 2:
                return 1.0
            
            try:
                _, s, _ = np.linalg.svd(centered)
                # Good line: first singular value dominates
                if s[0] < 1e-10:
                    return 0.0
                linearity = 1.0 - (s[1] / s[0]) if len(s) > 1 else 1.0
                return float(linearity)
            except:
                return 0.0
        
        elif target_formation == 'plane':
            if len(positions) < 3:
                return 1.0
            
            centered = positions - np.mean(positions, axis=0)
            
            try:
                _, s, _ = np.linalg.svd(centered)
                # Good plane: third singular value is small
                if s[0] < 1e-10:
                    return 0.0
                planarity = 1.0 - (s[2] / s[0]) if len(s) > 2 else 1.0
                return float(planarity)
            except:
                return 0.0
        
        elif target_formation == 'tetrahedron':
            if len(positions) != 4:
                return 0.0
            
            # Check regularity of tetrahedron
            distances = []
            for i in range(4):
                for j in range(i+1, 4):
                    distances.append(np.linalg.norm(positions[i] - positions[j]))
            
            mean_dist = np.mean(distances)
            if mean_dist < 1e-10:
                return 0.0
            
            std_dist = np.std(distances)
            regularity = 1.0 - (std_dist / mean_dist)
            return max(0.0, float(regularity))
        
        else:
            return 0.0
    
    def get_communication_efficiency(self) -> float:
        """
        Evaluate communication efficiency of the constellation.
        
        Based on:
        - All groups being able to communicate
        - Total data throughput
        """
        if not self.constellation.is_constellation_connected():
            return 0.0
        
        links = self.constellation.get_communication_links()
        
        if not links:
            return 1.0  # Single group
        
        # Calculate average data rate across all links
        total_rate = sum(info['data_rate'] for info in links.values())
        num_links = len(links)
        
        avg_rate = total_rate / num_links
        
        # Normalize by reference rate
        efficiency = min(1.0, avg_rate / self.constellation.comm_reqs.base_data_rate)
        
        return efficiency
    
    def get_propulsion_efficiency(self) -> float:
        """
        Get fraction of propulsion remaining.
        """
        total_remaining = self.constellation.get_total_delta_v_remaining()
        total_capacity = sum(
            p.max_delta_v for p in self.constellation._cube_propulsion.values()
        )
        
        if total_capacity <= 0:
            return 0.0
        
        return total_remaining / total_capacity

    # -------------------------------------------------------------------------
    # Physical metrics (new)
    # -------------------------------------------------------------------------

    # Physical constants
    SOLAR_CONSTANT_W_PER_M2 = 1361.0   # W/m² at 1 AU
    CUBE_FACE_AREA_M2 = 0.01           # 10 cm × 10 cm per face
    CUBE_EMISSIVITY = 0.85             # typical spacecraft MLI exterior
    STEFAN_BOLTZMANN = 5.67e-8         # W/(m²·K⁴)
    SPACE_TEMP_K = 4.0                 # cosmic background
    OPERATING_TEMP_K = 293.0           # ~20 °C nominal

    def get_solar_power_watts(
        self,
        sun_direction: Tuple[float, float, float],
        sun_distance_au: float = 10.0,
    ) -> float:
        """
        Total solar power collected by the whole constellation (Watts).

        Aggregates solar_array_efficiency across all group swarms and scales
        by the inverse-square law.  Physical range: 0 W … n_cubes × P_face.
        """
        from rewards.metrics import SwarmFaceAnalyzer

        solar_flux = self.SOLAR_CONSTANT_W_PER_M2 / (sun_distance_au ** 2)
        panel_watts_per_cube = solar_flux * self.CUBE_FACE_AREA_M2  # one face area

        # total_watts = 0.0
        # for group in self.constellation._groups.values():
        #     analyzer = SwarmFaceAnalyzer(group.swarm)
        #     efficiency = analyzer.compute_solar_array_efficiency(sun_direction)
        #     n = len(group.cube_ids)
        #     total_watts += efficiency * n * panel_watts_per_cube
        analyzer = SwarmFaceAnalyzer(self.constellation.swarm)
        efficiency = analyzer.compute_solar_array_efficiency(sun_direction)
        n = self.constellation.swarm.num_cubes
        total_watts = efficiency * n * panel_watts_per_cube

        return total_watts

    def get_heat_loss_watts(self) -> float:
        """
        Estimated radiative heat loss from the constellation (Watts).

        Uses Stefan-Boltzmann radiation from all exposed faces across all groups.
        Lower = better for cruise/power-conservation mode.
        """
        # total_exposed_faces = 0
        # for group in self.constellation._groups.values():
        #     total_exposed_faces += group.swarm.get_surface_area()
        total_exposed_faces = self.constellation.swarm.get_surface_area()

        exposed_area_m2 = total_exposed_faces * self.CUBE_FACE_AREA_M2
        # Net radiation (outward) per unit area
        q_rad = (
            self.CUBE_EMISSIVITY
            * self.STEFAN_BOLTZMANN
            * (self.OPERATING_TEMP_K ** 4 - self.SPACE_TEMP_K ** 4)
        )
        return exposed_area_m2 * q_rad

    def get_earth_array_gain(
        self,
        earth_direction: Tuple[float, float, float],
    ) -> float:
        """
        Effective coherent aperture area of the Earth-pointing antenna array (m²).

        For a flat phased array, effective area = physical area × cos(θ) where θ
        is the angle between the array normal and the Earth vector.  We aggregate
        across all exposed high-gain antenna faces in all groups.

        Returns area in m² — the physical proxy for downlink data-rate capacity.
        """
        from rewards.metrics import SwarmFaceAnalyzer
        from core.cube_faces import FaceFunction

        earth = np.array(earth_direction, dtype=float)
        earth /= np.linalg.norm(earth) + 1e-12

        # total_effective_area = 0.0
        # for group in self.constellation._groups.values():
        #     analyzer = SwarmFaceAnalyzer(group.swarm)
        #     result = analyzer.compute_antenna_effectiveness(earth_direction, FaceFunction.ANTENNA_HIGH_GAIN)
        #     # effective_aperture is already alignment-weighted count
        #     total_effective_area += result.get('effective_aperture', 0.0) * self.CUBE_FACE_AREA_M2
        analyzer = SwarmFaceAnalyzer(self.constellation.swarm)
        result = analyzer.compute_antenna_effectiveness(earth_direction, FaceFunction.ANTENNA_HIGH_GAIN)
        total_effective_area = result.get('effective_aperture', 0.0) * self.CUBE_FACE_AREA_M2

        return total_effective_area

    def get_angular_resolution_urad(
        self,
        baseline_m: Optional[float] = None,
        wavelength_m: float = 1e-6,  # visible light default
    ) -> float:
        """
        Diffraction-limited angular resolution in micro-radians (λ/B).

        Lower is better (finer resolution).  Uses max_baseline if not provided.
        Physical units allow direct comparison to science requirements.
        """
        if baseline_m is None:
            baseline_m = self.constellation.get_max_baseline()
        if baseline_m < 1e-3:
            return float('inf')
        return (wavelength_m / baseline_m) * 1e6  # convert to µrad

    def get_occultation_baseline_m(
        self,
        shadow_direction: Tuple[float, float, float],
    ) -> float:
        """
        Length of the constellation's baseline perpendicular to the shadow motion (m).

        For a stellar occultation, the science value is entirely in the component
        of separation that is *across* the shadow track.  Longer ⟹ finer spatial
        sampling of the atmosphere/ring profile.
        """
        shadow = np.array(shadow_direction, dtype=float)
        shadow /= np.linalg.norm(shadow) + 1e-12

        positions = [g.position for g in self.constellation._groups.values()]
        if len(positions) < 2:
            return 0.0

        # Project each group position onto the plane perpendicular to shadow
        def perp(pos):
            p = np.array(pos, dtype=float)
            return p - np.dot(p, shadow) * shadow

        perp_positions = [perp(p) for p in positions]

        max_perp_dist = 0.0
        for i, p1 in enumerate(perp_positions):
            for p2 in perp_positions[i + 1:]:
                d = np.linalg.norm(p1 - p2)
                if d > max_perp_dist:
                    max_perp_dist = d

        return max_perp_dist

    def get_critical_unit_shielding(
        self,
        threat_direction: Tuple[float, float, float],
        critical_cube_ids: Optional[Set[int]] = None,
    ) -> Dict[str, float]:
        """
        Quantify how well non-critical units shield critical ones from a threat.

        Args:
            threat_direction: Unit vector pointing FROM the threat TOWARD the swarm.
            critical_cube_ids: Set of cube IDs considered high-value. If None,
                               defaults to cubes with science/camera faces.

        Returns:
            {
              'shielded_fraction': fraction of critical cubes with ≥1 cube between them and threat,
              'mean_shield_depth': average number of shielding layers,
              'threat_flux_reduction': estimated flux reduction factor (0–1),
            }
        """
        from core.cube_faces import FaceFunction, FUNCTION_TO_FACE

        threat = np.array(threat_direction, dtype=float)
        threat /= np.linalg.norm(threat) + 1e-12

        # Collect all cubes and identify critical ones
        # all_cubes = {c.cube_id: c for group in self.constellation._groups.values()
        #              for c in group.swarm.get_all_cubes()}
        all_cubes = self.constellation.swarm.get_all_cubes()

        if critical_cube_ids is None:
            critical_cube_ids = set()
            for cube in all_cubes:
                if cube.has_camera:
                    critical_cube_ids.add(cube.cube_id)

        if not critical_cube_ids:
            return {'shielded_fraction': 1.0, 'mean_shield_depth': 0.0, 'threat_flux_reduction': 1.0}

        occupied_positions = {c.position for c in all_cubes}
        shielded = 0
        shield_depths = []

        for cid in critical_cube_ids:
            if cid not in all_cubes:
                continue
            cube = all_cubes[cid]
            pos = np.array(cube.position, dtype=float)

            # Ray-march from cube position toward threat, count intercepting cubes
            depth = 0
            step = pos.copy()
            for _ in range(50):  # max 50 cube-lengths away
                step = step + threat  # step one cube in threat direction
                rounded = tuple(int(round(v)) for v in step)
                if rounded in occupied_positions and rounded != cube.position:
                    depth += 1
                    if depth >= 3:  # enough shielding
                        break

            shield_depths.append(depth)
            if depth > 0:
                shielded += 1

        shielded_fraction = shielded / len(critical_cube_ids)
        mean_depth = float(np.mean(shield_depths)) if shield_depths else 0.0
        # Each layer attenuates ~30% of particle radiation (rough model)
        flux_reduction = max(0.0, 1.0 - 0.3 * mean_depth)

        return {
            'shielded_fraction': shielded_fraction,
            'mean_shield_depth': mean_depth,
            'threat_flux_reduction': flux_reduction,
        }

    def get_sampling_uniformity(self) -> float:
        """
        How uniformly the groups are distributed in 3D space (0–1, higher = more uniform).

        Uses the ratio of minimum to maximum pairwise inter-group distance.
        A regular tetrahedron or cube lattice scores near 1; a clump near 0.
        """
        positions = [g.position for g in self.constellation._groups.values()]
        if len(positions) < 2:
            return 1.0

        dists = []
        for i, p1 in enumerate(positions):
            for p2 in positions[i + 1:]:
                dists.append(np.linalg.norm(np.array(p1) - np.array(p2)))

        if not dists or max(dists) < 1e-6:
            return 0.0

        return float(min(dists) / max(dists))

    def get_all_metrics(self) -> Dict[str, float]:
        """Get all constellation metrics (extended with physical metrics)."""
        return {
            'num_groups': float(self.constellation.get_num_groups()),
            'effective_aperture': self.get_effective_aperture(),
            'coverage_volume': self.get_coverage_volume(),
            'communication_efficiency': self.get_communication_efficiency(),
            'propulsion_remaining': self.get_propulsion_efficiency(),
            'is_connected': float(self.constellation.is_constellation_connected()),
            'max_baseline': self.constellation.get_max_baseline(),
            'total_delta_v': self.constellation.get_total_delta_v_remaining(),
            'heat_loss_watts': self.get_heat_loss_watts(),
            'sampling_uniformity': self.get_sampling_uniformity(),
        }
    
    def get_all_metrics(self) -> Dict[str, float]:
        """Get all constellation metrics."""
        return {
            'num_groups': float(self.constellation.get_num_groups()),
            'effective_aperture': self.get_effective_aperture(),
            'coverage_volume': self.get_coverage_volume(),
            'communication_efficiency': self.get_communication_efficiency(),
            'propulsion_remaining': self.get_propulsion_efficiency(),
            'is_connected': float(self.constellation.is_constellation_connected()),
            'max_baseline': self.constellation.get_max_baseline(),
            'total_delta_v': self.constellation.get_total_delta_v_remaining(),
        }