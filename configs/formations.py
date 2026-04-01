from typing import Tuple

from core.swarm import Swarm


def create_cube_formation(swarm: Swarm, size: int = 4, 
                          origin: Tuple[int, int, int] = (0, 0, 0)) -> bool:
    """
    Arrange cubes into a cubic formation (size x size x size).
    
    Args:
        swarm: The swarm to arrange
        size: Side length of the cube
        origin: Corner position of the cube
        
    Returns:
        True if successful
    """
    needed = size ** 3
    available = swarm.num_cubes
    # if swarm.num_cubes < needed:
    #     raise ValueError(f"Need {needed} cubes for {size}x{size}x{size} cube, "
    #                     f"but swarm only has {swarm.num_cubes}")
    
    cube_id = 0
    for x in range(size):
        for y in range(size):
            for z in range(size):
                if cube_id >= available:
                    swarm.auto_connect_all()
                    return True  # Placed all available cubes, done
                pos = (origin[0] + x, origin[1] + y, origin[2] + z)
                if not swarm.place_cube(cube_id, pos):
                    return False
                cube_id += 1
    
    swarm.auto_connect_all()
    return True


def create_plane_formation(swarm: Swarm, width: int = 8, height: int = 8,
                           origin: Tuple[int, int, int] = (0, 0, 0),
                           normal_axis: int = 2) -> bool:
    """
    Arrange cubes into a flat plane formation.
    
    Args:
        swarm: The swarm to arrange
        width: Width of the plane
        height: Height of the plane
        origin: Corner position
        normal_axis: Which axis is normal to the plane (0=X, 1=Y, 2=Z)
        
    Returns:
        True if successful
    """
    needed = width * height
    if swarm.num_cubes < needed:
        raise ValueError(f"Need {needed} cubes for {width}x{height} plane, "
                        f"but swarm only has {swarm.num_cubes}")
    
    # Determine which axes to spread across
    axes = [0, 1, 2]
    axes.remove(normal_axis)
    axis1, axis2 = axes
    
    cube_id = 0
    for i in range(width):
        for j in range(height):
            pos = list(origin)
            pos[axis1] += i
            pos[axis2] += j
            pos = tuple(pos)
            
            if not swarm.place_cube(cube_id, pos):
                return False
            cube_id += 1
    
    swarm.auto_connect_all()
    return True


def create_line_formation(swarm: Swarm, length: int = 64,
                          origin: Tuple[int, int, int] = (0, 0, 0),
                          axis: int = 0) -> bool:
    """
    Arrange cubes into a single line.
    
    Args:
        swarm: The swarm to arrange
        length: Number of cubes in the line
        origin: Starting position
        axis: Which axis to extend along (0=X, 1=Y, 2=Z)
        
    Returns:
        True if successful
    """
    if swarm.num_cubes < length:
        raise ValueError(f"Need {length} cubes for line, "
                        f"but swarm only has {swarm.num_cubes}")
    
    for i in range(length):
        pos = list(origin)
        pos[axis] += i
        pos = tuple(pos)
        
        if not swarm.place_cube(i, pos):
            return False
    
    swarm.auto_connect_all()
    return True


def create_sparse_formation(swarm: Swarm, spacing: int = 2,
                            grid_size: int = 4,
                            origin: Tuple[int, int, int] = (0, 0, 0)) -> bool:
    """
    Arrange cubes in a sparse 3D grid with gaps between them.
    
    Note: Cubes won't be connected in this formation!
    
    Args:
        swarm: The swarm to arrange
        spacing: Distance between adjacent cubes
        grid_size: Number of cubes along each dimension
        origin: Corner position
        
    Returns:
        True if successful
    """
    needed = grid_size ** 3
    if swarm.num_cubes < needed:
        raise ValueError(f"Need {needed} cubes, but swarm only has {swarm.num_cubes}")
    
    cube_id = 0
    for x in range(grid_size):
        for y in range(grid_size):
            for z in range(grid_size):
                pos = (
                    origin[0] + x * spacing,
                    origin[1] + y * spacing,
                    origin[2] + z * spacing
                )
                if not swarm.place_cube(cube_id, pos):
                    return False
                cube_id += 1
    
    # Don't auto-connect since cubes are spaced apart
    return True


def create_custom_formation(swarm: Swarm, 
                            positions: list[Tuple[int, int, int]],
                            auto_connect: bool = True) -> bool:
    """
    Place cubes at arbitrary specified positions.
    
    Args:
        swarm: The swarm to arrange
        positions: List of (x, y, z) positions for each cube
        auto_connect: Whether to automatically connect adjacent cubes
        
    Returns:
        True if successful
    """
    if len(positions) > swarm.num_cubes:
        raise ValueError(f"Got {len(positions)} positions but only {swarm.num_cubes} cubes")
    
    for cube_id, pos in enumerate(positions):
        if not swarm.place_cube(cube_id, pos):
            return False
    
    if auto_connect:
        swarm.auto_connect_all()
    
    return True