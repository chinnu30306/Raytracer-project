"""Elements module for optical raytracing"""

import numpy as np
from raytracer.physics import refract, reflect


class OpticalElement:
    """
    Base class for all optical elements.
    
    This abstract class defines the interface that all optical elements must
    implement. Subclasses must provide their own implementations of intercept()
    and propagate_ray().
    """
    def intercept(self, ray):
        """
        Calculates the intersection point of a ray with the optical element.

        Args:
            ray (Ray): The ray object to intersect with the element.

        Returns:
            np.ndarray or None: The (x, y, z) position vector of the intersection
            point if found, otherwise None.
        """
        raise NotImplementedError('intercept() needs to be implemented in derived classes')
    def propagate_ray(self, ray):
        """
        Propagates a ray through the optical element.

        This method performs three steps:
            1. Determines if and where the ray intercepts the element.
            2. Calculates any change to the ray's direction vector.
            3. Updates the ray's internal state with the new position and direction.

        Args:
            ray (Ray): The ray object to propagate through the element.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError('propagate_ray() needs to be implemented in derived classes')


class SphericalSurface(OpticalElement):
    """
    Represents a spherical optical surface.

    This class provides the basic geometry for a spherical surface defined by
    its intersection with the z-axis, its aperture, and its curvature. It
    implements the intercept calculation but leaves propagate_ray() to
    subclasses (SphericalRefraction and SphericalReflection).

    Attributes:
        __z0 (float): The z-coordinate where the surface crosses the optical axis.
        __aper (float): The maximum extent of the surface from the optical axis.
        __curv (float): The curvature of the surface (1 / radius of curvature).
    """
    def __init__(self, z_0, aperture, curvature):
        """
        Initialises a SphericalSurface object.

        Args:
            z_0 (float): The z-coordinate where the surface intercepts the optical axis.
            aperture (float): The maximum radial extent of the surface from the
                optical axis. Rays outside this aperture are not transmitted.
            curvature (float): The curvature of the surface, given as a signed
                quantity with magnitude 1/(radius of curvature). Positive curvature
                means the center of curvature is at z > z_0 (convex to incident light).
        """
        self.__z0 = z_0
        self.__aper = aperture
        self.__curv = curvature

    def z_0(self):
        """
        Returns the z-coordinate where the surface intercepts the optical axis.

        Returns:
            float: The z-intercept of the surface.
        """
        return self.__z0

    def aperture(self):
        """
        Returns the aperture (maximum extent) of the surface from the optical axis.

        Returns:
            float: The radial aperture in mm.
        """
        return self.__aper

    def curvature(self):
        """
        Returns the curvature of the surface.

        Returns:
            float: The signed curvature (1 / radius of curvature).
        """
        return self.__curv

    def intercept(self, ray):
        """
        Calculates the first valid intersection point of a ray with the spherical surface.

        The method finds the parameter l along the ray where it intersects the
        spherical surface. The intersection is only valid if it falls within the aperture.

        Args:
            ray (Ray): The ray object to intersect with the surface.

        Returns:
            np.ndarray or None: The (x, y, z) position vector of the first valid
            intersection point, or None if no intersection exists or the
            intersection lies outside the aperture.
        """
        p = ray.pos()
        k = ray.direc()

        if np.isclose(self.curvature(), 0.):
            l = (self.z_0() - p[2]) / k[2]
            q = p + l * k
            return q

        rad = 1.0 / self.curvature()
        orig = np.array([0, 0, self.z_0() + rad])
        r = p - orig
        l = - np.dot(r, k) - np.sign(rad) * np.sqrt(np.dot(r, k)*np.dot(r, k) - (np.dot(r, r) - rad*rad))
        q = p + l * k

        return q if np.sqrt(q[0]*q[0] + q[1]*q[1]) < self.aperture() else None

    def propagate_ray(self, ray):
        """
        Propagates a ray through the spherical surface.

        Args:
            ray (Ray): The ray object to propagate.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("propagate_ray() needs to be implemented in derived classes")


class SphericalRefraction(SphericalSurface):
    """
    Represents a spherical refracting surface (Snell's law).

    This class models the refraction of a ray at a spherical interface between
    two media with potentially different refractive indices. The refractive
    indices can be constants or wavelength-dependent (dispersive) materials.

    Attributes:
        __n1 (float or DispersiveMaterial): Refractive index of the incident medium.
        __n2 (float or DispersiveMaterial): Refractive index of the transmitting medium.
    """
    def __init__(self, z_0, aperture, curvature, n_1, n_2):
        """
        Initialises a SphericalRefraction object.

        Args:
            z_0 (float): The z-coordinate where the surface intercepts the optical axis.
            aperture (float): The maximum radial extent of the surface.
            curvature (float): The signed curvature of the surface.
            n_1 (float or DispersiveMaterial): Refractive index of the medium
                on the incident side (left).
            n_2 (float or DispersiveMaterial): Refractive index of the medium
                on the transmitting side (right).
        """
        SphericalSurface.__init__(self, z_0, aperture, curvature)
        self.__n1 = n_1
        self.__n2 = n_2

    def n_1(self, wavelength=None):
        """
        Returns the refractive index on the left (incident) side of the surface.

        Args:
            wavelength (float, optional): Wavelength in mm for dispersive materials.
                Required if the refractive index is a DispersiveMaterial.
                Defaults to None.

        Returns:
            float: The refractive index at the specified wavelength.
        """
        if isinstance(self.__n1, (float, int)):
            return self.__n1
        return self.__n1.ref_index(wavelength)

    def n_2(self, wavelength=None):
        """
        Returns the refractive index on the right (transmitting) side of the surface.

        Args:
            wavelength (float, optional): Wavelength in mm for dispersive materials.
                Required if the refractive index is a DispersiveMaterial.
                Defaults to None.

        Returns:
            float: The refractive index at the specified wavelength.
        """
        if isinstance(self.__n2, (int, float)):
            return self.__n2
        return self.__n2.ref_index(wavelength)

    def propagate_ray(self, ray):
        """
        Propagates a ray through the refracting surface using Snell's law.

        The method finds the intersection point, computes the surface normal,
        calculates the refracted direction using Snell's law, and appends the
        new state to the ray's history.

        Args:
            ray (Ray): The ray object to propagate through the surface.
        """
        new_pos = self.intercept(ray)
        if new_pos is not None:
            if np.isclose(self.curvature(), 0.):
                normal = np.array([0, 0, 1])
            else:
                rad = 1.0 / self.curvature()
                orig = np.array([0, 0, self.z_0() + rad])
                normal = new_pos - orig

            n1 = self.n_1(ray.wavelength())
            n2 = self.n_2(ray.wavelength())
            new_direc = refract(ray.direc(), normal, n1, n2)

            if new_direc is not None:
                ray.append(new_pos, new_direc)

    def focal_point(self, wavelength=None):
        """
        Calculates the focal point of the spherical refracting surface.

        Args:
            wavelength (float, optional): Wavelength in mm for dispersive materials.
                Required if the refractive indices are DispersiveMaterial objects.
                Defaults to None.

        Returns:
            float or None: The z-coordinate of the focal point, or None if the
            surface is planar (curvature = 0).
        """
        if np.isclose(self.curvature(), 0.):
            return None

        n1 = self.n_1(wavelength)
        n2 = self.n_2(wavelength)
        f = n2 / ((n2 - n1) * self.curvature())
        return f + self.z_0()


class OutputPlane(SphericalSurface):
    """
    Represents an output plane where rays terminate.

    This class is used to capture rays at a specific z-plane. It propagates rays
    to the plane without changing their direction.

    Attributes:
        __z0 (float): The z-coordinate of the output plane.
    """
    def __init__(self, z_0):
        """
        Initialises a OutputPlane object.

        Args:
            z_0 (float): The z-coordinate of the output plane.
        """
        SphericalSurface.__init__(self, z_0, aperture=None, curvature=0)

    def propagate_ray(self, ray):
        """
        Propagates a ray to the output plane without changing its direction.

        The method finds the intersection point of the ray with the output plane
        and appends this position to the ray's history. The ray's direction
        remains unchanged.

        Args:
            ray (Ray): The ray object to propagate to the output plane.
        """
        new_pos = self.intercept(ray)
        ray.append(new_pos, ray.direc())


class SphericalReflection(SphericalSurface):
    """
    Represents a spherical reflecting surface (mirror).

    This class models the reflection of a ray at a spherical mirror surface.
    The angle of incidence equals the angle of reflection.

    Attributes:
        __z0 (float): The z-coordinate where the surface intercepts the optical axis.
        __aper (float): The maximum radial extent of the surface.
        __curv (float): The signed curvature of the surface.
    """
    def __init__(self, z_0, aperture, curvature):
        """
        Initialises a SphericalRefraction object.

        Args:
            z_0 (float): The z-coordinate where the surface intercepts the optical axis.
            aperture (float): The maximum radial extent of the surface.
            curvature (float): The signed curvature of the surface (1 / radius of curvature).
        """
        SphericalSurface.__init__(self, z_0, aperture, curvature)

    def propagate_ray(self, ray):
        """
        Propagates a ray through the reflecting surface using the law of reflection.

        Args:
            ray (Ray): The ray object to reflect off the surface.
        """
        new_pos = self.intercept(ray)
        if new_pos is not None:
            if np.isclose(self.curvature(), 0.):
                normal = np.array([0, 0, -1])
            else:
                rad = 1.0 / self.curvature()
                orig = np.array([0, 0, self.z_0() + rad])
                normal = new_pos - orig

            new_direc = reflect(ray.direc(), normal)

            if new_direc is not None:
                ray.append(new_pos, new_direc)

    def focal_point(self):
        """
        Calculates the focal point of the spherical reflecting surface.

        Returns:
            float or None: The z-coordinate of the focal point, or None if the
            surface is planar (curvature = 0).
        """
        if np.isclose(self.curvature(), 0.):
            return None

        f = 0.5 / self.curvature()
        return f + self.z_0()
