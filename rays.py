"""Rays module for optical raytracing"""

import numpy as np
import matplotlib.pyplot as plt
from raytracer.genpolar import rtrings


class Ray:
    """
    A class representing an optical ray in 3D space.

    The ray is defined by its current position and direction in Cartesian
    coordinates. The z-axis is taken as the optical axis. The ray stores its
    entire propagation history as a sequence of points.

    Attributes:
        __pos (list): Private list of numpy arrays containing all points along
            the ray path. The last element is the current position.
        __direc (np.ndarray): Private numpy array representing the current
            normalised direction vector of the ray.
        __lambda (float): Private wavelength value stored in mm.
    """

    def __init__(self, pos=[0, 0, 0], direc=[0, 0, 1], wavelength=588e-9):
        """
        Initialises a Ray object.

        Args:
            pos (list or np.ndarray, optional): 3D position vector [x, y, z].
                Defaults to [0, 0, 0].
            direc (list or np.ndarray, optional): 3D direction vector [dx, dy, dz].
                Defaults to [0, 0, 1].
            wavelength (float, optional): Wavelength of the ray in meters.
                Defaults to 588e-9.

        Raises:
            TypeError: If pos or direc is not a list or numpy array.
            ValueError: If pos or direc does not have exactly 3 elements.
            ValueError: If direction vector is zero.
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise TypeError("Position must be a list or numpy array")

        position = np.array(pos, dtype=float)
        if len(position) != 3:
            raise ValueError("Position must have exactly 3 elements")

        if not isinstance(direc, (list, np.ndarray)):
            raise TypeError("Direction must be a list or numpy array")

        direction = np.array(direc, dtype=float)
        if len(direction) != 3:
            raise ValueError("Direction must have exactly 3 elements")

        if np.isclose(np.sqrt(np.dot(direction, direction)), 0):
            raise ValueError("Direction vector cannot be zero")

        self.__pos = [position]
        self.__direc = direction / np.sqrt(np.dot(direction, direction))  # normalisation of direction vector
        self.__lambda = wavelength * 1000    # convert wavelength from m to mm for consistency

    def pos(self):
        """
        Returns the current position of the ray.

        Returns:
            np.ndarray: Current 3D position vector [x, y, z] of the ray.
        """
        return self.__pos[-1]

    def direc(self):
        """
        Returns the current direction vector of the ray.

        Returns:
            np.ndarray: Current normalised 3D direction vector [dx, dy, dz] of the ray.
        """
        return self.__direc

    def wavelength(self):
        """
        Returns the wavelength of the ray.

        Returns:
            float: Wavelength of the ray in mm.
        """
        return self.__lambda

    def append(self, pos, direc):
        """
        Appends a new point and direction to the ray's propagation history.

        This method is called when the ray interacts with an optical element
        to record the new position and direction.

        Args:
            pos (list or np.ndarray): New 3D position vector [x, y, z] after
                interaction with an optical element.
            direc (list or np.ndarray): New 3D direction vector [dx, dy, dz]
                after interaction.

        Raises:
            TypeError: If pos or direc is not a list or numpy array.
            ValueError: If pos or direc does not have exactly 3 elements.
            ValueError: If direction vector is zero.
        """
        if not isinstance(pos, (list, np.ndarray)):
            raise TypeError("Position must be a list or numpy array")

        position = np.array(pos, dtype=float)
        if len(position) != 3:
            raise ValueError("Position must have exactly 3 elements")

        if not isinstance(direc, (list, np.ndarray)):
            raise TypeError("Direction must be a list or numpy array")

        direction = np.array(direc, dtype=float)
        if len(direction) != 3:
            raise ValueError("Direction must have exactly 3 elements")

        if np.isclose(np.sqrt(np.dot(direction, direction)), 0):
            raise ValueError("Direction vector cannot be zero")

        new_pos = position
        new_direc = direction / np.sqrt(np.dot(direction, direction))
        self.__pos.append(new_pos)
        self.__direc = new_direc

    def vertices(self):
        """
        Returns all points along the ray's propagation path.

        Returns:
            list: A list containing all position vectors (as numpy arrays) along
            the ray path, in chronological order.
        """
        return self.__pos

    def z_int(self):
        """
        Calculates the z-coordinate where the ray intercepts the optical axis.

        The optical axis is defined as the line where x = 0 and y = 0. This
        method finds the z-coordinate at which the ray crosses this axis.

        Returns:
            float: The z-coordinate of the intercept point.

        Raises:
            ValueError: If the ray does not intercept the optical axis
                (i.e., when the ray is parallel to the z-axis).
        """
        p = self.pos()
        k = self.direc()

        if np.isclose(k[1], 0.) and np.isclose(k[0], 0.):
            raise ValueError("Ray does not intercept optical axis")
        l = -p[1] / k[1]

        return p[2] + l * k[2]


class RayBundle:
    """
    Represents a circular bundle of rays transverse to the optical axis.

    This class generates a set of rays arranged in concentric rings around the
    optical axis (z-axis). The rays are initially directions along the +z axis.

    Attributes:
        __rays (list): Private list containing all Ray objects in the bundle.
    """
    def __init__(self, rmax=5., nrings=5., multi=6):
        """
        Initialises a RayBundle object.

        The rays are generated using the function rtrings(), which yields the
        radial and angular coordinates, with points distributed evenly around
        each ring. The innermost ring includes the central ray at r=0.

        Args:
            rmax (float, optional): Maximum radius of the bundle in mm.
                Defaults to 5.0.
            nrings (int or float, optional): Number of concentric rings.
                Defaults to 5.
            multi (int or float, optional): Multiplier determining the number
                of points per ring. Total rays = nrings * multi.
                Defaults to 6.
        """
        self.__rays = []
        for r, t in rtrings(rmax, int(nrings), int(multi)):
            x = r * np.cos(t)
            y = r * np.sin(t)
            ray = Ray(pos=[x, y, 0])
            self.__rays.append(ray)

    def rays(self):
        """
        Returns the list of ray objects in the bundle.

        Returns:
            list: A list containing all Ray objects in the bundle.
        """
        return self.__rays

    def propagate_bundle(self, elements):
        """
        Propagates every ray in the bundle through a list of optical elements.

        Args:
            elements (list): A list of OpticalElement objects to propagate the
            rays through.
        """
        for i in self.rays():
            for j in elements:
                j.propagate_ray(i)

    def track_plot(self, title=None):
        """
        Creates a 2D plot showing the path of each ray in the bundle.

        The plot shows the ray trajectories in the y-z plane (since the system
        is rotationally symmetric around the z-axis).

        Returns:
            matplotlib.figure.Figure: A figure object containing the ray
            trajectory plot.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("z-direction (mm)")
        ax.set_ylabel("y-direction (mm)")

        if title is not None:
            ax.set_title(title)

        for i in self.rays():
            point = np.array(i.vertices())
            z = point[:, 2]
            y = point[:, 1]
            ax.plot(z, y)

        return fig

    def rms(self):
        """
        Calculates the root-mean-square (RMS) spot size of the ray bundle.

        The RMS value is computed as the square root of the mean squared radial
        distance of all rays from the optical axis at their current positions.

        Returns:
            float: The RMS spot size in millimeters.
        """
        r_squared = []
        for i in self.rays():
            point = i.pos()
            x = point[0]
            y = point[1]
            r_squared.append(x*x + y*y)

        return np.sqrt(np.mean(r_squared))

    def spot_plot(self, title=None):
        """
        Creates a scatter plot of the ray positions in the x-y plane.

        This plot shows the current cross-sectional distribution of rays
        perpendicular to the optical axis (z-axis). Each ray is represented as
        a dot at its (x, y) coordinate.

        Returns:
            matplotlib.figure.Figure: A figure object containing the spot
            diagram.
        """
        fig, ax = plt.subplots()
        ax.set_xlabel("x-direction (mm)")
        ax.set_ylabel("y-direction (mm)")

        if title is not None:
            ax.set_title(title)

        x = []
        y = []
        for i in self.rays():
            point = i.pos()
            x.append(point[0])
            y.append(point[1])
        ax.plot(x, y, '.')

        return fig
