import numpy as np
import pyvista as pv
from typing import Tuple

def load_xyzr(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an .xyzr file.
    Returns:
        coords: (N×3) array of [x, y, z] positions (unit: same as file)
        radii:  (N,)   array of sphere radii         (unit: same as file)
    """
    data = np.loadtxt(path)      # vectorized load of floats
    return data[:, :3], data[:, 3]

def visualize_xyzr(path: str) -> None:
    """
    Render the spheres in a 3D interactive window.
    """
    coords, radii = load_xyzr(path)
    plotter = pv.Plotter()
    # add each sphere; opacity can be adjusted as needed
    for (x, y, z), r in zip(coords, radii):
        sphere = pv.Sphere(radius=r, center=(x, y, z))
        plotter.add_mesh(sphere, opacity=0.6)
    plotter.show()

if __name__ == "__main__":
    visualize_xyzr("output/spheres_0.xyzr")