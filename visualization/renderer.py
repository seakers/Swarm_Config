import numpy as np
from typing import Tuple, Set, Optional

from core.swarm import Swarm
from mechanics.moves import HingeMove, MovementSystem


class SwarmVisualizer:
    """
    Visualizes the swarm configuration in 3D.
    
    Uses matplotlib for basic visualization. Can be extended to use
    other libraries (pyvista, plotly) for better interactivity.
    """
    
    def __init__(self, swarm: Swarm):
        self.swarm = swarm
        self._fig = None
        self._ax = None
    
    def render(self, 
               show_connections: bool = True,
               show_ids: bool = False,
               highlight_cubes: Optional[Set[int]] = None,
               title: str = "",
               elev: float = 20,
               azim: float = 45) -> None:
        """
        Render the current swarm configuration.
        
        Args:
            show_connections: Draw lines between connected cubes
            show_ids: Label each cube with its ID
            highlight_cubes: Set of cube IDs to highlight in a different color
            title: Title for the plot
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        if self._fig is None:
            self._fig = plt.figure(figsize=(12, 10))
            self._ax = self._fig.add_subplot(111, projection='3d')
        else:
            self._ax.clear()
        
        ax = self._ax
        highlight_cubes = highlight_cubes or set()
        
        # Draw each cube
        for cube in self.swarm.get_all_cubes():
            color = 'orange' if cube.cube_id in highlight_cubes else 'steelblue'
            alpha = 0.9 if cube.cube_id in highlight_cubes else 0.7
            self._draw_cube(ax, cube.position, color, alpha)
            
            if show_ids:
                ax.text(cube.position[0], cube.position[1], cube.position[2],
                       str(cube.cube_id), fontsize=8, ha='center', va='center')
        
        # Draw connections
        if show_connections:
            for conn in self.swarm._connections.get_all_connections():
                cube1 = self.swarm.get_cube(conn.cube_id_1)
                cube2 = self.swarm.get_cube(conn.cube_id_2)
                if cube1 and cube2:
                    xs = [cube1.position[0], cube2.position[0]]
                    ys = [cube1.position[1], cube2.position[1]]
                    zs = [cube1.position[2], cube2.position[2]]
                    ax.plot(xs, ys, zs, 'k-', linewidth=0.5, alpha=0.3)
        
        # Set axis properties
        bounds = self.swarm.get_bounds()
        margin = 1
        ax.set_xlim(bounds[0][0] - margin, bounds[1][0] + margin)
        ax.set_ylim(bounds[0][1] - margin, bounds[1][1] + margin)
        ax.set_zlim(bounds[0][2] - margin, bounds[1][2] + margin)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title or f"Swarm: {self.swarm}")
        
        # Equal aspect ratio
        max_range = max(
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2]
        ) / 2.0 + margin
        
        mid_x = (bounds[1][0] + bounds[0][0]) / 2
        mid_y = (bounds[1][1] + bounds[0][1]) / 2
        mid_z = (bounds[1][2] + bounds[0][2]) / 2
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def _draw_cube(self, ax, position: Tuple[int, int, int], 
                   color: str, alpha: float) -> None:
        """Draw a single cube at the given position."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        x, y, z = position
        
        # Define the 8 vertices of a unit cube centered at position
        r = 0.45  # Slightly smaller than 0.5 to show gaps between cubes
        vertices = [
            [x-r, y-r, z-r],
            [x+r, y-r, z-r],
            [x+r, y+r, z-r],
            [x-r, y+r, z-r],
            [x-r, y-r, z+r],
            [x+r, y-r, z+r],
            [x+r, y+r, z+r],
            [x-r, y+r, z+r]
        ]
        
        # Define the 6 faces using vertex indices
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
        ]
        
        # Create and add the cube
        cube = Poly3DCollection(faces, alpha=alpha, 
                                facecolor=color, 
                                edgecolor='darkblue',
                                linewidth=0.5)
        ax.add_collection3d(cube)
    
    def show(self) -> None:
        """Display the figure and block until closed."""
        import matplotlib.pyplot as plt
        plt.show()
    
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save the current figure to a file."""
        if self._fig is not None:
            self._fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close(self) -> None:
        """Close the figure."""
        import matplotlib.pyplot as plt
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None


def animate_move_sequence(swarm: Swarm, moves: list[HingeMove], 
                          interval: float = 3) -> None:
    """
    Animate a sequence of moves.
    
    Args:
        swarm: The swarm (will be modified)
        moves: List of moves to execute in order
        interval: Time between frames in seconds
    """
    import time
    
    movement = MovementSystem(swarm)
    viz = SwarmVisualizer(swarm)
    
    viz.render(title="Initial configuration")
    time.sleep(interval)
    
    for i, move in enumerate(moves):
        result = movement.execute_move(move)
        
        if result.success:
            viz.render(
                title=f"Move {i+1}/{len(moves)}: Cube {move.cube_id}",
                highlight_cubes={move.cube_id}
            )
        else:
            viz.render(
                title=f"Move {i+1} FAILED: {result.reason}",
                highlight_cubes={move.cube_id}
            )
        
        time.sleep(interval)
    
    viz.render(title="Final configuration")
    viz.show()