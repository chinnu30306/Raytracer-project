"""Physics module for optical ray tracing"""

import numpy as np

def refract(direc, normal, n_1, n_2):
    """
    Calculates the refracted ray direction using Snell's law.

    This function computes the direction of a ray after it passes through an
    interface between two media with different refractive indices. If the
    incident angle exceeds the critical angle, total internal reflection (TIR)
    occurs and the function returns None.

    Args:
        direc (list or np.ndarray): Incident direction vector [dx, dy, dz].
        normal (list or np.ndarray): Surface normal vector [nx, ny, nz].
        n_1 (float): Refractive index of the incident (first) material.
        n_2 (float): Refractive index of the transmitting (second) material.

    Returns:
        np.ndarray or None: The normalized refracted direction vector if
        refraction occurs, or None if total internal reflection occurs.

    Raises:
        TypeError: If direc or normal is not a list or numpy array.
        ValueError: If direc or normal does not have exactly 3 elements.
        ValueError: If direc or normal is a zero vector.
    """
    if not isinstance(normal, (list, np.ndarray)):
        raise TypeError("Normal must be a list or numpy array")

    norm = np.array(normal, dtype=float)
    if len(norm) != 3:
        raise ValueError("Normal must have exactly 3 elements")

    if np.sqrt(np.dot(norm, norm)) == 0:
        raise ValueError("Normal vector cannot be zero")

    if not isinstance(direc, (list, np.ndarray)):
        raise TypeError("Direction must be a list or numpy array")

    direction = np.array(direc, dtype=float)
    if len(direction) != 3:
        raise ValueError("Direction must have exactly 3 elements")

    if np.sqrt(np.dot(direction, direction)) == 0:
        raise ValueError("Direction vector cannot be zero")

    k1_hat = direction / np.sqrt(np.dot(direction, direction))
    n_hat = norm / np.sqrt(np.dot(norm, norm))

    if np.dot(k1_hat, n_hat) > 0.:  # ensures that incident angle is less than 90˚
        n_hat = -n_hat

    theta_1 = np.arccos(-np.dot(k1_hat, n_hat))
    if np.sin(theta_1) == 0:
        return k1_hat
    if np.sin(theta_1) <= n_2/n_1:
        theta_2 = np.arcsin(n_1 * np.sin(theta_1) / n_2)
        tangential = (k1_hat + np.cos(theta_1) * n_hat) / np.sin(theta_1)
        k2 = np.sin(theta_2) * tangential - np.cos(theta_2) * n_hat
        return k2 / np.sqrt(np.dot(k2, k2))
    return None


def reflect(direc, normal):
    """
    Calculates the reflected ray direction using the law of reflection.

    This function computes the direction of a ray after it reflects off a
    surface. The angle of incidence equals the angle of reflection, and the
    reflected ray lies in the plane defined by the incident ray and the surface
    normal.

    Args:
        direc (list or np.ndarray): Incident direction vector [dx, dy, dz].
        normal (list or np.ndarray): Surface normal vector [nx, ny, nz].

    Returns:
        np.ndarray: The normalized reflected direction vector.

    Raises:
        TypeError: If direc or normal is not a list or numpy array.
        ValueError: If direc or normal does not have exactly 3 elements.
        ValueError: If direc or normal is a zero vector.
    """
    if not isinstance(normal, (list, np.ndarray)):
        raise TypeError("Normal must be a list or numpy array")

    norm = np.array(normal, dtype=float)
    if len(norm) != 3:
        raise ValueError("Normal must have exactly 3 elements")

    if np.isclose(np.sqrt(np.dot(norm, norm)), 0.):
        raise ValueError("Normal vector cannot be zero")

    if not isinstance(direc, (list, np.ndarray)):
        raise TypeError("Direction must be a list or numpy array")

    direction = np.array(direc, dtype=float)
    if len(direction) != 3:
        raise ValueError("Direction must have exactly 3 elements")

    if np.isclose(np.sqrt(np.dot(direction, direction)), 0.):
        raise ValueError("Direction vector cannot be zero")

    k1_hat = direction / np.sqrt(np.dot(direction, direction))
    n_hat = norm / np.sqrt(np.dot(norm, norm))

    if np.dot(k1_hat, n_hat) > 0.:
        n_hat = -n_hat
    theta = np.arccos(-np.dot(k1_hat, n_hat))

    if np.isclose(np.sin(theta), 0.):
        return -k1_hat

    tangential = (k1_hat + np.cos(theta) * n_hat) / np.sin(theta)
    k2 = np.sin(theta) * tangential + np.cos(theta) * n_hat
    return k2 / np.sqrt(np.dot(k2, k2))


class DispersiveMaterial:
    """
    Represents a dispersive material with wavelength-dependent refractive index.

    This class models optical dispersion using the Sellmeier equation, which
    describes how the refractive index varies with wavelength. The Sellmeier
    coefficients are material-specific constants.

    Attributes:
        __b (np.ndarray): Private array of Sellmeier B coefficients (3 values).
        __c (np.ndarray): Private array of Sellmeier C coefficients (3 values, in μm²).
    """
    def __init__(self, b_coeffs, c_coeffs):
        """
        Initialises a DispersiveMaterial object.

        Args:
            b_coeffs (list or np.ndarray): Sellmeier B coefficients (3 values).
            c_coeffs (list or np.ndarray): Sellmeier C coefficients (3 values, in μm²).

        Raises:
            TypeError: If b_coeffs or c_coeffs is not a list or numpy array.
            ValueError: If b_coeffs or c_coeffs does not have exactly 3 elements.
        """
        if not isinstance(b_coeffs, (list, np.ndarray)):
            raise TypeError("Sellmeier B coefficients must be a list or numpy array")

        b = np.array(b_coeffs, dtype=float)
        if len(b) != 3:
            raise ValueError("Sellmeier B coefficients must have exactly 3 elements")

        if not isinstance(c_coeffs, (list, np.ndarray)):
            raise TypeError("Sellmeier C coefficients must be a list or numpy array")

        c = np.array(c_coeffs, dtype=float)
        if len(c) != 3:
            raise ValueError("Sellmeier C coefficients must have exactly 3 elements")

        self.__b = b
        self.__c = c

    def ref_index(self, wavelength):
        """
        Calculates the refractive index for a given wavelength using the Sellmeier equation.

        Args:
            wavelength (float): Wavelength in millimeters (mm).

        Returns:
            float: The calculated refractive index at the specified wavelength.
        """
        wl = wavelength * 1e3
        n_squared = 1
        for i in range(3):
            n_squared += wl*wl * (self.__b[i] / (wl*wl - self.__c[i]))

        return np.sqrt(n_squared)
