"""Lenses module for composite optical elements"""

from raytracer.elements import OpticalElement, SphericalRefraction

class Lens(OpticalElement):
    """
    Base class representing a lens made of two spherical refracting surfaces.

    This class is not intended to be instantiated directly. Instead, use
    specific lens types such as PlanoConvex, BiConvex, or ConvexPlano.

    Attributes:
        __z0 (float): Z-coordinate of the leftmost surface.
        __curv1 (float): Curvature of the first surface.
        __curv2 (float): Curvature of the second surface.
        __n_in (float or DispersiveMaterial): Refractive index inside the lens.
        __n_out (float or DispersiveMaterial): Refractive index outside the lens.
        __thick (float): Lens thickness along the optical axis.
        __aper (float): Lens aperture (maximum radial extent).
        __element1 (SphericalRefraction): First refracting surface.
        __element2 (SphericalRefraction): Second refracting surface.
    """
    def __init__(self, z_0, curvature1, curvature2, n_inside, n_outside, thickness, aperture):
        """
        Initialises a Lens object.

        Args:
            z_0 (float): Z-coordinate of the leftmost surface where it
                intercepts the optical axis.
            curvature1 (float): Curvature of the first surface (signed quantity,
                positive = convex toward incident light).
            curvature2 (float): Curvature of the second surface (signed quantity,
                positive = convex toward incident light).
            n_inside (float or DispersiveMaterial): Refractive index of the lens
                material (may be constant or wavelength-dependent).
            n_outside (float or DispersiveMaterial): Refractive index of the
                surrounding medium (usually 1.0 for air).
            thickness (float): Lens thickness along the optical axis (z-direction)
                in millimeters.
            aperture (float): Maximum radial extent of the lens surfaces from
                the optical axis in millimeters.
        """
        self.__z0 = z_0
        self.__curv1 = curvature1
        self.__curv2 = curvature2
        self.__n_in = n_inside
        self.__n_out = n_outside
        self.__thick = thickness
        self.__aper = aperture

        self.__element1 = SphericalRefraction(
            z_0=self.__z0, aperture=self.__aper,
            curvature=self.__curv1, n_1=self.__n_out, n_2=self.__n_in
        )
        self.__element2 = SphericalRefraction(
            z_0=self.__z0 + self.__thick, aperture=self.__aper,
            curvature=self.__curv2, n_1=self.__n_in, n_2=self.__n_out
        )

    def intercept(self, ray):
        """
        Calculates intercept points with both lens surfaces.

        Args:
            ray (Ray): The ray object to find intercepts for.
        """
        self.__element1.intercept(ray)
        self.__element2.intercept(ray)

    def propagate_ray(self, ray):
        """
        Propagates a ray through both lens surfaces.

        The ray is first refracted by the first surface, then by the second
        surface. The ray's internal state is updated with the new position and
        direction after each surface.

        Args:
            ray (Ray): The ray object to propagate through the lens.
        """
        self.__element1.propagate_ray(ray)
        self.__element2.propagate_ray(ray)

    def focal_point(self):
        """
        Calculates the focal point of the lens using the Lensmaker's equation
        and the back focal distance.

        Returns:
            float: z-coordinate of the focal point in mm.
        """
        n = self.__n_in / self.__n_out
        f = 1.0 / (
            (n - 1) * (
                self.__curv1 - self.__curv2 +
                (n - 1) * self.__thick * self.__curv1 * self.__curv2 / n
            )
        )
        bfd = f * (1 - (n - 1) * self.__thick * self.__curv1 / n)
        return bfd + self.__z0 + self.__thick


class PlanoConvex(Lens):
    """
    Plano-convex lens with one planar surface and one convex surface.

    The orientation depends on the sign of the curvature:
        If curvature > 0: convex surface faces incident light (left), planar on right
        If curvature < 0: planar surface faces incident light, convex on right
    """
    def __init__(self, z_0, curvature, n_inside, n_outside, thickness, aperture):
        """
        Initialises a PlanoConvex object.

        The lens is automatically oriented so that the convex surface is
        placed appropriately based on the sign of curvature.

        Args:
            z_0 (float): Z-coordinate of the leftmost surface.
            curvature (float): Curvature of the convex surface. The sign
                determines which side the convex surface is on.
            n_inside (float or DispersiveMaterial): Refractive index of the lens.
            n_outside (float or DispersiveMaterial): Refractive index outside.
            thickness (float): Lens thickness in mm.
            aperture (float): Lens aperture in mm.
        """
        if curvature > 0:
            curv1 = curvature
            curv2 = 0.
        else:
            curv1 = 0.
            curv2 = curvature

        super().__init__(z_0, curv1, curv2, n_inside, n_outside, thickness, aperture)


class BiConvex(Lens):
    """Bi-convex lens with both surfaces convex outward."""
    def __init__(self, z_0, curvature1, curvature2, n_inside, n_outside, thickness, aperture):
        """
        Initialises a BiConvex object.

        Args:
            z_0 (float): Z-coordinate of the leftmost surface.
            curvature (float): Curvature of the convex surface. The sign
                determines which side the convex surface is on.
            n_inside (float or DispersiveMaterial): Refractive index of the lens.
            n_outside (float or DispersiveMaterial): Refractive index outside.
            thickness (float): Lens thickness in mm.
            aperture (float): Lens aperture in mm.

        Raises:
            ValueError: If curvature1 is not positive or curvature2 is not negative.
        """
        if curvature1 <= 0:
            raise ValueError("Curvature1 must be positive")

        if curvature2 >= 0:
            raise ValueError("Curvature2 must be negative")

        super().__init__(z_0, curvature1, curvature2, n_inside, n_outside, thickness, aperture)


class ConvexPlano(Lens):
    """Convex-plano lens with convex surface first, then planar surface."""
    def __init__(self, z_0, curvature, n_inside, n_outside, thickness, aperture):
        """
        Initialises a ConvexPlano object.

        Args:
            z_0 (float): Z-coordinate of the leftmost surface.
            curvature (float): Curvature of the convex surface. The sign
                determines which side the convex surface is on.
            n_inside (float or DispersiveMaterial): Refractive index of the lens.
            n_outside (float or DispersiveMaterial): Refractive index outside.
            thickness (float): Lens thickness in mm.
            aperture (float): Lens aperture in mm.
        
        Raises:
            ValueError: If curvature is not positive.
        """
        if curvature <= 0:
            raise ValueError("Curvature must be positive")

        super().__init__(z_0, curvature, 0., n_inside, n_outside, thickness, aperture)
