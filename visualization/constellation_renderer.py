import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from typing import Optional, Dict, List, Tuple, Set

from core.swarm import Swarm
from core.constellation import Constellation, GroupState


class ConstellationVisualizer:
    """
    Visualizer for multi-group constellations.
    
    Provides two view modes:
    1. Local view: Shows cube positions within a group (grid scale)
    2. Global view: Shows group positions in space (meter/km scale)
    """
    
    def __init__(self, constellation: Constellation):
        self.constellation = constellation
        self._fig = None
        self._axes = None
        
        # Color scheme for different groups
        self._group_colors = [
            'steelblue', 'coral', 'seagreen', 'orchid',
            'goldenrod', 'turquoise', 'salmon', 'mediumpurple'
        ]
    
    def render_local(self, group_id: Optional[int] = None,
                     title: str = "",
                     show_connections: bool = True,
                     elev: float = 20,
                     azim: float = 45) -> None:
        """
        Render local view of cube positions within group(s).
        
        Args:
            group_id: Specific group to show (None = show all)
            title: Plot title
            show_connections: Draw connection lines
            elev: Elevation angle
            azim: Azimuth angle
        """
        if self._fig is None:
            self._fig = plt.figure(figsize=(12, 10))
            self._axes = self._fig.add_subplot(111, projection='3d')
        else:
            self._axes.clear()
        
        ax = self._axes
        
        groups = self.constellation.get_all_groups()
        
        if group_id is not None:
            groups = [g for g in groups if g.group_id == group_id]
        
        # Draw cubes for each group
        for g_idx, group in enumerate(groups):
            color = self._group_colors[g_idx % len(self._group_colors)]
            
            for cube_id in group.cube_ids:
                cube = self.constellation.swarm.get_cube(cube_id)
                if cube:
                    self._draw_cube(ax, cube.position, color, alpha=0.7)
        
        # Draw connections if requested
        if show_connections:
            for conn in self.constellation.swarm._connections.get_all_connections():
                cube1 = self.constellation.swarm.get_cube(conn.cube_id_1)
                cube2 = self.constellation.swarm.get_cube(conn.cube_id_2)
                if cube1 and cube2:
                    xs = [cube1.position[0], cube2.position[0]]
                    ys = [cube1.position[1], cube2.position[1]]
                    zs = [cube1.position[2], cube2.position[2]]
                    ax.plot(xs, ys, zs, 'k-', linewidth=0.5, alpha=0.3)
        
        # Set axis properties
        bounds = self.constellation.swarm.get_bounds()
        margin = 1
        
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
        
        ax.set_xlabel('X (grid)')
        ax.set_ylabel('Y (grid)')
        ax.set_zlabel('Z (grid)')
        
        # Add legend for groups
        legend_elements = []
        for g_idx, group in enumerate(groups):
            color = self._group_colors[g_idx % len(self._group_colors)]
            from matplotlib.patches import Patch
            legend_elements.append(
                Patch(facecolor=color, label=f'Group {group.group_id} ({len(group.cube_ids)} cubes)')
            )
        ax.legend(handles=legend_elements, loc='upper left')
        
        ax.set_title(title or f"Constellation Local View ({len(groups)} groups)")
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def render_global(self, title: str = "",
                      show_comm_links: bool = True,
                      elev: float = 20,
                      azim: float = 45) -> None:
        """
        Render global view of group positions in space.
        
        Args:
            title: Plot title
            show_comm_links: Draw communication links between groups
            elev: Elevation angle
            azim: Azimuth angle
        """
        if self._fig is None:
            self._fig = plt.figure(figsize=(12, 10))
            self._axes = self._fig.add_subplot(111, projection='3d')
        else:
            self._axes.clear()
        
        ax = self._axes
        
        groups = self.constellation.get_all_groups()
        
        # Collect positions for bounds
        positions = []
        
        # Draw each group as a sphere
        for g_idx, group in enumerate(groups):
            color = self._group_colors[g_idx % len(self._group_colors)]
            pos = group.position
            positions.append(pos)
            
            # Size based on number of cubes
            size = 100 + len(group.cube_ids) * 10
            
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=color, s=size, marker='o', alpha=0.8,
                      label=f'Group {group.group_id} ({len(group.cube_ids)} cubes)')
            
            # Draw velocity vector
            vel = group.velocity
            vel_scale = 100  # Scale for visibility
            if np.linalg.norm(vel) > 0.01:
                ax.quiver(pos[0], pos[1], pos[2],
                         vel[0] * vel_scale, vel[1] * vel_scale, vel[2] * vel_scale,
                         color=color, alpha=0.5, arrow_length_ratio=0.1)
        
        # Draw communication links
        if show_comm_links and len(groups) > 1:
            links = self.constellation.get_communication_links()
            
            for (ga_id, gb_id), info in links.items():
                ga = self.constellation.get_group(ga_id)
                gb = self.constellation.get_group(gb_id)
                
                if ga and gb:
                    if info['can_communicate']:
                        linestyle = '-'
                        alpha = 0.5
                    else:
                        linestyle = ':'
                        alpha = 0.2
                    
                    ax.plot([ga.position[0], gb.position[0]],
                           [ga.position[1], gb.position[1]],
                           [ga.position[2], gb.position[2]],
                           linestyle=linestyle, color='gray', alpha=alpha)
        
        # Set axis properties
        if positions:
            positions = np.array(positions)
            
            # Add margin
            margin = max(1000, np.max(np.abs(positions)) * 0.2)
            
            center = np.mean(positions, axis=0)
            max_range = np.max(np.abs(positions - center)) + margin
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # Add info text
        info_text = (f"Time: {self.constellation.get_time():.0f}s\n"
                    f"Max baseline: {self.constellation.get_max_baseline():.0f}m\n"
                    f"Connected: {self.constellation.is_constellation_connected()}")
        ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
                 verticalalignment='top', fontsize=10, family='monospace')
        
        ax.legend(loc='upper right')
        ax.set_title(title or f"Constellation Global View ({len(groups)} groups)")
        ax.view_init(elev=elev, azim=azim)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def render_dual(self, title: str = "") -> None:
        """
        Render both local and global views side by side.
        """
        if self._fig is None or len(self._fig.axes) != 2:
            self._fig = plt.figure(figsize=(16, 7))
            self._axes = [
                self._fig.add_subplot(121, projection='3d'),
                self._fig.add_subplot(122, projection='3d')
            ]
        else:
            for ax in self._axes:
                ax.clear()
        
        # Local view (left)
        ax_local = self._axes[0]
        self._render_local_to_axis(ax_local)
        ax_local.set_title("Local View (Cube Positions)")
        
        # Global view (right)
        ax_global = self._axes[1]
        self._render_global_to_axis(ax_global)
        ax_global.set_title("Global View (Group Positions)")
        
        self._fig.suptitle(title or f"Constellation ({self.constellation.get_num_groups()} groups)", 
                          fontsize=14)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)
    
    def _render_local_to_axis(self, ax) -> None:
        """Render local view to specific axis."""
        groups = self.constellation.get_all_groups()
        
        for g_idx, group in enumerate(groups):
            color = self._group_colors[g_idx % len(self._group_colors)]
            
            for cube_id in group.cube_ids:
                cube = self.constellation.swarm.get_cube(cube_id)
                if cube:
                    self._draw_cube(ax, cube.position, color, alpha=0.7)
        
        # Set bounds
        bounds = self.constellation.swarm.get_bounds()
        margin = 1
        max_range = max(
            bounds[1][0] - bounds[0][0],
            bounds[1][1] - bounds[0][1],
            bounds[1][2] - bounds[0][2]
        ) / 2.0 + margin
        
        mid = [(bounds[1][i] + bounds[0][i]) / 2 for i in range(3)]
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    def _render_global_to_axis(self, ax) -> None:
        """Render global view to specific axis."""
        groups = self.constellation.get_all_groups()
        positions = []
        
        for g_idx, group in enumerate(groups):
            color = self._group_colors[g_idx % len(self._group_colors)]
            pos = group.position
            positions.append(pos)
            
            size = 100 + len(group.cube_ids) * 10
            ax.scatter(pos[0], pos[1], pos[2], 
                      c=color, s=size, marker='o', alpha=0.8)
        
        # Communication links
        links = self.constellation.get_communication_links()
        for (ga_id, gb_id), info in links.items():
            ga = self.constellation.get_group(ga_id)
            gb = self.constellation.get_group(gb_id)
            
            if ga and gb:
                style = '-' if info['can_communicate'] else ':'
                ax.plot([ga.position[0], gb.position[0]],
                       [ga.position[1], gb.position[1]],
                       [ga.position[2], gb.position[2]],
                       style, color='gray', alpha=0.4)
        
        if positions:
            positions = np.array(positions)
            margin = max(1000, np.max(np.abs(positions)) * 0.2)
            center = np.mean(positions, axis=0)
            max_range = np.max(np.abs(positions - center)) + margin
            
            ax.set_xlim(center[0] - max_range, center[0] + max_range)
            ax.set_ylim(center[1] - max_range, center[1] + max_range)
            ax.set_zlim(center[2] - max_range, center[2] + max_range)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
    
    def _draw_cube(self, ax, position: Tuple[int, int, int],
                   color: str, alpha: float) -> None:
        """Draw a single cube."""
        x, y, z = position
        r = 0.45
        
        vertices = [
            [x-r, y-r, z-r], [x+r, y-r, z-r],
            [x+r, y+r, z-r], [x-r, y+r, z-r],
            [x-r, y-r, z+r], [x+r, y-r, z+r],
            [x+r, y+r, z+r], [x-r, y+r, z+r]
        ]
        
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]
        
        cube = Poly3DCollection(faces, alpha=alpha,
                                facecolor=color,
                                edgecolor='darkblue',
                                linewidth=0.5)
        ax.add_collection3d(cube)
    
    def show(self) -> None:
        """Display figure."""
        plt.show()
    
    def save(self, filename: str, dpi: int = 150) -> None:
        """Save figure to file."""
        if self._fig is not None:
            self._fig.savefig(filename, dpi=dpi, bbox_inches='tight')
    
    def close(self) -> None:
        """Close figure."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axes = None