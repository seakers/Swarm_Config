import numpy as np
from typing import Tuple, List, Set, Optional


def manhattan_distance(pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> int:
    """Calculate Manhattan distance between two positions."""
    return sum(abs(a - b) for a, b in zip(pos1, pos2))


def euclidean_distance(pos1: Tuple[int, int, int], pos2: Tuple[int, int, int]) -> float:
    """Calculate Euclidean distance between two positions."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))


def normalize_vector(vec: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalize a 3D vector to unit length."""
    arr = np.array(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        return (0.0, 0.0, 0.0)
    normalized = arr / norm
    return tuple(normalized.tolist())


def positions_to_relative(positions: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
    """
    Convert absolute positions to relative positions (centered at origin).
    
    Useful for comparing configurations regardless of translation.
    """
    if not positions:
        return []
    
    # Find minimum coordinates
    min_x = min(p[0] for p in positions)
    min_y = min(p[1] for p in positions)
    min_z = min(p[2] for p in positions)
    
    # Shift all positions
    return [(p[0] - min_x, p[1] - min_y, p[2] - min_z) for p in positions]


def configuration_signature(positions: List[Tuple[int, int, int]]) -> Tuple:
    """
    Create a canonical signature for a configuration.
    
    This is invariant to translation and can be used to compare
    configurations or detect duplicates.
    """
    relative = positions_to_relative(positions)
    return tuple(sorted(relative))


def random_connected_positions(num_cubes: int, seed: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """
    Generate random positions that form a connected structure.
    
    Uses a growth algorithm starting from origin.
    """
    if seed is not None:
        np.random.seed(seed)
    
    positions = [(0, 0, 0)]
    occupied = {(0, 0, 0)}
    
    directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    
    while len(positions) < num_cubes:
        # Pick a random existing position
        base = positions[np.random.randint(len(positions))]
        
        # Try random directions
        np.random.shuffle(directions)
        
        for d in directions:
            new_pos = (base[0] + d[0], base[1] + d[1], base[2] + d[2])
            if new_pos not in occupied:
                positions.append(new_pos)
                occupied.add(new_pos)
                break
    
    return positions