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