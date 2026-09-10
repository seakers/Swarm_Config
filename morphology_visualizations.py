"""
Generates a figure for the ROSE extended abstract:
(left)  a single cubesat unit with per-face functions labeled
(right) three target morphologies: compact packing, dense planar array,
        and sparse separated constellation.

All cubes are colored consistently by face function across every panel.

Requires: numpy, matplotlib
Run:      python make_morphology_figure.py
Output:   morphology_figure.pdf (and .png)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# -----------------------------------------------------------------------------
# Face function assignment, matching STANDARD_FACE_ASSIGNMENT in cube_faces.py:
#   POS_Z (top)    -> high-gain antenna (Earth comms)
#   NEG_Z (bottom) -> solar array
#   POS_X (front)  -> camera
#   NEG_X (back)   -> radiator
#   POS_Y (left)   -> inter-satellite antenna
#   NEG_Y (right)  -> science instruments
# -----------------------------------------------------------------------------
FACE_INFO = {
    "top":    ("High-gain antenna", "#d98c5f"),  # +Z
    "bottom": ("Solar array",       "#3a6ea5"),  # -Z
    "front":  ("Camera",            "#6aa84f"),  # +X
    "back":   ("Radiator",          "#b0b0b0"),  # -X
    "left":   ("Inter-sat antenna", "#e0c341"),  # +Y
    "right":  ("Science instr.",    "#8e6fb0"),  # -Y
}

# Unit cube corner vertices (0/1 in each axis)
_CORNERS = np.array([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
])

# Which corners make up each face, keyed by direction label
_FACE_CORNERS = {
    "bottom": [0, 1, 2, 3],  # -Z
    "top":    [4, 5, 6, 7],  # +Z
    "front":  [1, 2, 6, 5],  # +X
    "back":   [0, 3, 7, 4],  # -X
    "left":   [3, 2, 6, 7],  # +Y
    "right":  [0, 1, 5, 4],  # -Y
}

# Outward normal direction of each face (for placing labels)
_FACE_NORMAL = {
    "top":    (0, 0, 1),
    "bottom": (0, 0, -1),
    "front":  (1, 0, 0),
    "back":   (-1, 0, 0),
    "left":   (0, 1, 0),
    "right":  (0, -1, 0),
}


def _draw_cube(ax, origin=(0, 0, 0), alpha=0.9, edgecolor="k", lw=0.6):
    """Draw a unit cube at `origin`, coloring each face by its function."""
    ox, oy, oz = origin
    verts = _CORNERS + np.array([ox, oy, oz])

    for label, idx in _FACE_CORNERS.items():
        face = [verts[i] for i in idx]
        fc = FACE_INFO[label][1]  # always color by face function
        poly = Poly3DCollection([face], alpha=alpha)
        poly.set_facecolor(fc)
        poly.set_edgecolor(edgecolor)
        poly.set_linewidth(lw)
        ax.add_collection3d(poly)


def _set_equal_3d(ax, pts):
    """Give the 3D axes an equal aspect ratio around the given points."""
    pts = np.asarray(pts, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    centers = (mins + maxs) / 2.0
    span = (maxs - mins).max()
    span = max(span, 1.0)
    r = span / 2.0 * 1.15
    ax.set_xlim(centers[0] - r, centers[0] + r)
    ax.set_ylim(centers[1] - r, centers[1] + r)
    ax.set_zlim(centers[2] - r, centers[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def _draw_config(ax, origins, title):
    """Draw a morphology given a list of integer cube origins.

    Cubes are drawn back-to-front relative to the view so that face
    coloring reads correctly despite matplotlib's imperfect 3D depth sort.
    """
    # Sort cubes back-to-front for the chosen view direction (elev=22, azim=-58)
    ordered = sorted(origins, key=lambda o: (o[0] + o[1] - o[2]))
    for o in ordered:
        _draw_cube(ax, origin=o)
    all_pts = _CORNERS[None, :, :] + np.array(origins)[:, None, :]
    _set_equal_3d(ax, all_pts.reshape(-1, 3))
    ax.set_title(title, fontsize='x-large', pad=-2)
    ax.view_init(elev=22, azim=-58)


# -----------------------------------------------------------------------------
# Morphology definitions (integer grid positions, matching the discrete grid
# used in grid.py / cube.py)
# -----------------------------------------------------------------------------
def compact_packing():
    """A dense 2x2x2 block: minimal exposed surface area for cruise."""
    return [(x, y, z) for x in range(2) for y in range(2) for z in range(2)]


def planar_array():
    """A flat 4x4 sheet: many co-aligned faces for a communication array."""
    return [(x, y, 0) for x in range(4) for y in range(4)]


def sparse_constellation():
    """Several small separated groups spread along a wide baseline."""
    groups = []
    for gx in (0, 6, 13):
        groups.append((0, gx, 0))
        groups.append((1, gx, 0))
    return groups


# -----------------------------------------------------------------------------
# Build the figure
# -----------------------------------------------------------------------------
def main():
    fig = plt.figure()

    # Panel 1: single labeled unit
    ax0 = fig.add_subplot(2, 2, 1, projection="3d")
    _draw_cube(ax0, origin=(0, 0, 0), alpha=0.95)
    _set_equal_3d(ax0, _CORNERS.astype(float))
    ax0.set_title("Single unit", fontsize='x-large', pad=-2)
    ax0.view_init(elev=22, azim=-58)

    # Text labels on the visible faces
    # for label in ("top", "front", "left"):
    #     nx, ny, nz = _FACE_NORMAL[label]
    #     center = np.array([0.5, 0.5, 0.5]) + 0.75 * np.array([nx, ny, nz])
    #     ax0.text(center[0], center[1], center[2],
    #              FACE_INFO[label][0], fontsize=6.5, ha="center", va="center")

    # Panels 2-4: morphologies, all colored consistently by face function
    ax1 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_config(ax1, compact_packing(), "Compact (cruise)")

    ax2 = fig.add_subplot(2, 2, 3, projection="3d")
    _draw_config(ax2, planar_array(), "Dense array (downlink)")

    ax3 = fig.add_subplot(2, 2, 4, projection="3d")
    _draw_config(ax3, sparse_constellation(), "Sparse (observation)")

    # A shared legend for the face functions
    legend_handles = [Patch(facecolor=c, edgecolor="k", label=name)
                      for name, c in FACE_INFO.values()]
    fig.legend(handles=legend_handles, loc="center left",
               fontsize='large',
               bbox_to_anchor=(0.9, 0.5))

    fig.subplots_adjust(left=0.02, right=0.86, top=0.97, bottom=0.03,
                        wspace=0.01, hspace=0.01)
    fig.savefig("morphology_figure.pdf", bbox_inches="tight", dpi=300)
    fig.savefig("morphology_figure.png", bbox_inches="tight", dpi=300)
    print("Wrote morphology_figure.pdf and morphology_figure.png")


if __name__ == "__main__":
    main()