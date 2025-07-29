import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import math
from typing import Tuple

def load_xyzr(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load an .xyzr file.

    Args:
        path: filesystem path to the .xyzr text file

    Returns:
        group_ids: (N,)   array of 1-indexed group IDs (unitless)
        coords:    (N, 3) array of [x, y, z] positions (px)
        radii:     (N,)   array of sphere radii         (px)
    """
    data = np.loadtxt(path)

    group_ids = data[:, 0].astype(int)   # first column → group ID
    coords    = data[:, 1:4]             # next three columns → X, Y, Z (px)
    radii     = data[:, 4]               # last column → radius (px)

    return group_ids, coords, radii

def analyze_radii(
    radii: np.ndarray,
    n_modes: int = 3,
    domain_volume: float = 1000**3
) -> None:
    """
    Build histogram C(ρ), identify the top `n_modes` peaks by count among local maxima,
    and print their radii + volume fractions.

    radii:           (N,) array of radii (px)
    n_modes:         number of modes to identify
    domain_volume:   total domain volume (px³)
    """
    # histogram counts C_i and bin centers (px)
    counts, edges = np.histogram(radii, bins='auto')
    centers = (edges[:-1] + edges[1:]) / 2

    # find all local maxima indices
    local_max_idxs = [
        i for i in range(1, len(counts)-1)
        if counts[i] > counts[i-1] and counts[i] > counts[i+1]
    ]

    # if fewer peaks than requested, pad with highest-count bins
    if len(local_max_idxs) < n_modes:
        # global top bins
        sorted_idxs = list(np.argsort(counts)[::-1])
        for idx in sorted_idxs:
            if idx not in local_max_idxs:
                local_max_idxs.append(idx)
            if len(local_max_idxs) == n_modes:
                break

    # select top n_modes from local maxima by count
    local_max_idxs.sort(key=lambda i: counts[i], reverse=True)
    selected_idxs = local_max_idxs[:n_modes]

    # sort selected modes by radius value
    selected_idxs.sort(key=lambda i: centers[i])
    mode_radii = [float(centers[i]) for i in selected_idxs]

    # print detected modes
    print(f"Detected {len(mode_radii)} mode(s):")
    for j, r in enumerate(mode_radii, start=1):
        print(f"  Mode {j}: radius ≈ {r:.2f} px")

    # compute volume per sphere (px³) and assign to nearest mode
    volumes = (4.0/3.0) * math.pi * radii**3
    mode_vols = [0.0]*len(mode_radii)
    for r, v in zip(radii, volumes):
        nearest = min(range(len(mode_radii)), key=lambda j: abs(r - mode_radii[j]))
        mode_vols[nearest] += v

    # print volume fractions
    print(f"\nDomain volume: {domain_volume:.0f} px³")
    print("Volume fraction per mode:")
    for j, mv in enumerate(mode_vols, start=1):
        frac = mv / domain_volume
        print(f"  Mode {j}: {frac:.6f}")

    # plot histogram and mark modes
    plt.figure()
    plt.hist(radii, bins='auto', edgecolor='black')
    for r in mode_radii:
        plt.axvline(r, linestyle='--', label=f'{r:.2f}px')
    plt.title("Sphere Radii Distribution")
    plt.xlabel("Radius (px)")
    plt.ylabel("Count")
    plt.legend(title="Modes")
    plt.show()

def visualize_xyzr(path: str) -> None:
    """
    Render the spheres in a 3D interactive window.
    """
    group_ids, coords, radii = load_xyzr(path)
    plotter = pv.Plotter()
    for (x, y, z), r in zip(coords, radii):
        sphere = pv.Sphere(radius=r, center=(x, y, z))
        plotter.add_mesh(sphere, opacity=0.6)
    plotter.show()


def visualize_xyzr_2d(path: str, plane: str, coord: float) -> None:
    """
    Render a 2D slice of the spheres in an .xyzr file.

    Args:
        path:  filesystem path to the .xyzr text file
        plane: one of 'xy', 'xz', 'yz' indicating the slicing plane
        coord: coordinate along the orthogonal axis (px) where the slice is taken
    """
    group_ids, coords, radii = load_xyzr(path)

    # pick which axes are x,y vs. orthogonal
    axes_map = {'xy': (0, 1, 2), 'xz': (0, 2, 1), 'yz': (1, 2, 0)}
    if plane not in axes_map:
        raise ValueError("plane must be 'xy', 'xz' or 'yz'")
    i_axis, j_axis, orth_axis = axes_map[plane]

    # distance from each sphere center to the slice plane
    d_orth = np.abs(coords[:, orth_axis] - coord)  # (N,)

    # mask only those that actually intersect the plane
    mask = d_orth <= radii
    pts_2d     = coords[mask][:, [i_axis, j_axis]]                 # (M,2) [px]
    slice_rad  = np.sqrt(radii[mask]**2 - d_orth[mask]**2)         # (M,) [px]
    groups_2d  = group_ids[mask]                                   # (M,)

    # choose distinct colors per group
    unique_groups = np.unique(groups_2d)
    cmap = plt.get_cmap('tab10')
    color_map = {g: cmap((g-1) % cmap.N) for g in unique_groups}

    fig, ax = plt.subplots()
    # draw each cross‑section circle
    for (x, y), r, g in zip(pts_2d, slice_rad, groups_2d):
        circ = plt.Circle((x, y), r,
                          facecolor=color_map[g],
                          edgecolor='black',
                          alpha=0.5)
        ax.add_patch(circ)

    # now *expand* the view to include all circles
    if pts_2d.shape[0] > 0:
        x_min, x_max = pts_2d[:, 0].min(), pts_2d[:, 0].max()
        y_min, y_max = pts_2d[:, 1].min(), pts_2d[:, 1].max()
        r_max = slice_rad.max()
        ax.set_xlim(x_min - r_max, x_max + r_max)
        ax.set_ylim(y_min - r_max, y_max + r_max)

    # labels, aspect, title
    axis_names = ['X','Y','Z']
    ax.set_xlabel(f"{axis_names[i_axis]} (px)")
    ax.set_ylabel(f"{axis_names[j_axis]} (px)")
    ax.set_aspect('equal', 'box')
    ax.set_title(f"Slice {plane.upper()} at {coord:.1f}px")

    # legend
    handles = [
        Patch(facecolor=color_map[g], edgecolor='black',
              label=f"Group {g}", alpha=0.5)
        for g in unique_groups
    ]
    ax.legend(handles=handles, title="Group")

    plt.tight_layout()
    plt.show()


def main():
    filepath = "output/spheres_0.xyzr"
    group_ids, coords, radii = load_xyzr(filepath)
    analyze_radii(radii, n_modes=2)   # change n_modes as needed
    #visualize_xyzr(filepath)           # interactive 3D view
    visualize_xyzr_2d("output/spheres_0.xyzr", plane="xy", coord=150.0)


if __name__ == "__main__":
    main()
