"""genpolar module - Polar coordinate generation for ray bundles."""

import numpy as np

def rtrings(rmax, nrings, multi):
    """
    Generates radius and theta coordinates for concentric rings of points.

    This generator yields points arranged in concentric rings from the center
    outward. The center point (0, 0) is yielded first. The number of points
    per ring increases linearly with the ring index.

    Args:
        rmax (float): Maximum radius to extend the bundle to in millimeters.
            The outermost ring will be at approximately this radius.
        nrings (int): Number of concentric rings to produce (excluding the
            center point). Must be a positive integer.
        multi (int): Multiplier that determines how many points are placed on
            each ring. The number of points on ring i is i * multi, where i
            starts from 1 for the first ring.

    Yields:
        tuple[float, float]: A tuple (r, theta) where:
            r (float): Radial distance from origin in millimeters.
            theta (float): Angular coordinate in radians, ranging from 0 to 2π.
    """
    yield 0.0, 0.0
    radius = np.linspace(0, rmax, nrings+1)
    for i in range(1, len(radius)):
        npoints = i * multi
        for j in range(npoints):
            theta = (2*np.pi / npoints)*j
            yield radius[i], theta
