"""Analysis module."""
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from raytracer._utils.decorators import SaveOutput
from raytracer.rays import Ray, RayBundle
from raytracer.elements import SphericalRefraction, SphericalReflection, OutputPlane
from raytracer.lenses import PlanoConvex, BiConvex, ConvexPlano
from raytracer.physics import DispersiveMaterial


def task8():
    """
    Task 8.

    In this function you should check your propagate_ray function properly
    finds the correct intercept and correctly refracts a ray. Don't forget
    to check that the correct values are appended to your Ray object.
    """
    sr = SphericalRefraction(z_0=10, aperture=5, curvature=0.2, n_1=1., n_2=1.5)
    r1 = Ray()
    r2 = Ray(pos=[1, 2, 3], direc=[4, 5, 6])

    print(sr.intercept(r1))
    sr.propagate_ray(r1)
    print(r1.vertices())
    print(r1.direc())

    print(sr.intercept(r2))
    sr.propagate_ray(r2)
    print(r2.vertices())
    print(r2.direc())


@SaveOutput("task10")
def task10():
    """
    Task 10.

    In this function you should create Ray objects with the given initial positions.
    These rays should be propagated through the surface, up to the output plane.
    You should then plot the tracks of these rays.
    This function should return the matplotlib figure of the ray paths.

    Returns:
        Figure: the ray path plot.
    """
    lens = SphericalRefraction(z_0=100, aperture=34, curvature=0.03, n_1=1., n_2=1.5)
    screen = OutputPlane(z_0=250)

    init_pos = [[0, 4, 0], [0, 1, 0], [0, 0.2, 0], [0, 0, 0], [0, -0.2, 0], [0, -1, 0], [0, -4, 0]]
    fig, ax = plt.subplots()

    for i in init_pos:
        r = Ray(pos=i)
        lens.propagate_ray(r)
        screen.propagate_ray(r)
        point = np.array(r.vertices())
        z = point[:, 2]
        y = point[:, 1]
        ax.plot(z, y)

    ax.set_xlabel("z-direction (mm)")
    ax.set_ylabel("y-direction (mm)")
    ax.set_title("Task 10")
    # fig.savefig("plots_for_marking/task10.png")

    return fig


@SaveOutput("task11", plot_output_indices=itemgetter(0))
def task11():
    """
    Task 11.

    In this function you should propagate the three given paraxial rays through the system
    to the output plane and the tracks of these rays should then be plotted.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for ray paths
    2. the calculated focal point.

    Returns:
        tuple[Figure, float]: the ray path plot and the focal point
    """
    lens = SphericalRefraction(z_0=100, aperture=34, curvature=0.03, n_1=1., n_2=1.5)
    f = lens.focal_point()
    screen = OutputPlane(z_0=f)

    init_pos = [[0.1, 0.1, 0], [0, 0, 0], [-0.1, -0.1, 0]]
    fig, ax = plt.subplots()

    for i in init_pos:
        r = Ray(pos=i)
        lens.propagate_ray(r)
        screen.propagate_ray(r)
        point = np.array(r.vertices())
        z = point[:, 2]
        y = point[:, 1]
        ax.plot(z, y)

    ax.set_xlabel("z-direction (mm)")
    ax.set_ylabel("y-direction (mm)")
    ax.set_title("Task 11")
    # fig.savefig("plots_for_marking/task11.png")

    return fig, f


@SaveOutput("task12")
def task12():
    """
    Task 12.

    In this function you should create a RayBunble and propagate it to the output plane
    before plotting the tracks of the rays.
    This function should return the matplotlib figure of the track plot.

    Returns:
        Figure: the track plot.
    """
    rb = RayBundle()
    lens = SphericalRefraction(z_0=100, aperture=34, curvature=0.03, n_1=1., n_2=1.5)
    screen = OutputPlane(z_0=lens.focal_point())

    rb.propagate_bundle([lens, screen])
    fig = rb.track_plot(title="Task 12")
    # fig.savefig("plots_for_marking/task12.png")

    return fig


@SaveOutput("task13", plot_output_indices=itemgetter(0))
def task13():
    """
    Task 13.

    In this function you should again create and propagate a RayBundle to the output plane
    before plotting the spot plot.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot
    2. the simulation RMS

    Returns:
        tuple[Figure, float]: the spot plot and rms
    """
    rb = RayBundle()
    lens = SphericalRefraction(z_0=100, aperture=34, curvature=0.03, n_1=1., n_2=1.5)
    screen = OutputPlane(z_0=lens.focal_point())

    rb.propagate_bundle([lens, screen])
    rms = rb.rms()
    fig = rb.spot_plot(title='Task 13')
    # fig.savefig("plots_for_marking/task13.png")

    return fig, rms


@SaveOutput("task14", plot_output_indices=itemgetter(0))
def task14():
    """
    Task 14.

    In this function you will trace a number of RayBundles through the optical system and
    plot the RMS and diffraction scale dependence on input beam radii.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the simulation RMS for input beam radius 2.5
    3. the diffraction scale for input beam radius 2.5

    Returns:
        tuple[Figure, float, float]: the plot, the simulation RMS value, the diffraction scale.
    """
    lens = SphericalRefraction(z_0=100, aperture=34, curvature=0.03, n_1=1., n_2=1.5)
    screen = OutputPlane(z_0=lens.focal_point())
    bundle_rad = np.linspace(0.1, 10)
    rms = []
    diff_scale = []

    fig, ax = plt.subplots()
    for i in bundle_rad:
        rb = RayBundle(rmax=i)
        rb.propagate_bundle([lens, screen])
        rms.append(rb.rms())
        diff_scale.append(588e-6 * (lens.focal_point() - 100) / (2 * i))

    ax.plot(bundle_rad, rms, label="RMS")
    ax.plot(bundle_rad, diff_scale, label="Diffraction scale")
    ax.set_xlabel("Bundle radius (mm)")
    ax.set_ylabel("(mm)")
    ax.set_title("Task 14")
    ax.legend()
    # fig.savefig("plots_for_marking/task14.png")

    rb_specific = RayBundle(rmax=2.5)
    rb_specific.propagate_bundle([lens, screen])
    rms_specific = rb_specific.rms()
    diff_scale_specific = 588e-6 * (lens.focal_point() - 100) / 5

    return fig, rms_specific, diff_scale_specific


@SaveOutput(["task15a", "task15b"], plot_output_indices=itemgetter(0, 2))
def task15():
    """
    Task 15.

    In this function you will create plano-convex lenses in each orientation and propagate a RayBundle
    through each to their respective focal point. You should then plot the spot plot for each orientation.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the spot plot for the plano-convex system
    2. the focal point for the plano-convex lens
    3. the matplotlib figure object for the spot plot for the convex-plano system
    4  the focal point for the convex-plano lens


    Returns:
        tuple[Figure, float, Figure, float]: the spot plots and rms for plano-convex and convex-plano.
    """
    rb1 = RayBundle()
    pc = PlanoConvex(z_0=100, curvature=-0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f1 = pc.focal_point()
    screen1 = OutputPlane(z_0=f1)
    rb1.propagate_bundle([pc, screen1])
    fig1 = rb1.spot_plot(title='Task 15a')
    # fig1.savefig("plots_for_marking/task15a.png")

    rb2 = RayBundle()
    cp = PlanoConvex(z_0=100, curvature=0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f2 = cp.focal_point()
    screen2 = OutputPlane(z_0=f2)
    rb2.propagate_bundle([cp, screen2])
    fig2 = rb2.spot_plot(title='Task 15b')
    # fig2.savefig("plots_for_marking/task15b.png")

    return fig1, f1, fig2, f2


@SaveOutput("task16", plot_output_indices=itemgetter(0))
def task16():
    """
    Task 16.

    In this function you will be again plotting the radial dependence of the RMS and diffraction values
    for each orientation of your lens.
    This function should return the following items as a tuple in the following order:
    1. the matplotlib figure object for the diffraction scale plot
    2. the RMS for input beam radius 3.5 for the plano-convex system
    3. the RMS for input beam radius 3.5 for the convex-plano system
    4  the diffraction scale for input beam radius 3.5

    Returns:
        tuple[Figure, float, float, float]: the plot, RMS for plano-convex, RMS for convex-plano, diffraction scale.
    """
    pc = PlanoConvex(z_0=100, curvature=-0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f1 = pc.focal_point()
    screen1 = OutputPlane(z_0=f1)

    cp = PlanoConvex(z_0=100, curvature=0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f2 = cp.focal_point()
    screen2 = OutputPlane(z_0=f2)

    bundle_rad = np.linspace(0.1, 10)
    rms1 = []
    rms2 = []
    diff_scale = []

    fig, ax = plt.subplots()
    for i in bundle_rad:
        rb1 = RayBundle(rmax=i)
        rb1.propagate_bundle([pc, screen1])
        rms1.append(rb1.rms())
        diff_scale.append(588e-6 * (pc.focal_point() - 100 - 5) / (2 * i))

        rb2 = RayBundle(rmax=i)
        rb2.propagate_bundle([cp, screen2])
        rms2.append(rb2.rms())

    ax.plot(bundle_rad, rms1, label="RMS pc")
    ax.plot(bundle_rad, rms2, label="RMS cp")
    ax.plot(bundle_rad, diff_scale, label="Diffraction scale")
    ax.set_xlabel("Bundle radius (mm)")
    ax.set_ylabel("(mm)")
    ax.set_title("Task 16")
    ax.legend()
    # fig.savefig("plots_for_marking/task16.png")

    rb_specific1 = RayBundle(rmax=3.5)
    rb_specific1.propagate_bundle([pc, screen1])
    rms_specific1 = rb_specific1.rms()
    rb_specific2 = RayBundle(rmax=3.5)
    rb_specific2.propagate_bundle([cp, screen2])
    rms_specific2 = rb_specific2.rms()
    diff_scale_specific = 588e-6 * (pc.focal_point() - 100 - 5) / 7

    return fig, rms_specific1, rms_specific2, diff_scale_specific


@SaveOutput("task17", plot_output_indices=itemgetter(0))
def task17():
    """
    Task 17.

    In this function you will be first plotting the spot plot for your PlanoConvex lens with the curved
    side first (at the focal point). You will then be optimising the curvatures of a BiConvex lens
    in order to minimise the RMS spot size at the same focal point. This function should return
    the following items as a tuple in the following order:
    1. The comparison spot plot for both PlanoConvex (curved side first) and BiConvex lenses at PlanoConvex focal point.
    2. The RMS spot size for the PlanoConvex lens at focal point
    3. the RMS spot size for the BiConvex lens at PlanoConvex focal point

    Returns:
        tuple[Figure, float, float]: The combined spot plot, RMS for the PC lens, RMS for the BiConvex lens
    """
    fig, ax = plt.subplots()
    ax.set_xlabel("x-direction (mm)")
    ax.set_ylabel("y-direction (mm)")
    ax.set_title("Task 17")

    rb2 = RayBundle()
    cp = PlanoConvex(z_0=100, curvature=0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f2 = cp.focal_point()
    screen2 = OutputPlane(z_0=f2)
    rb2.propagate_bundle([cp, screen2])
    rms_cp = rb2.rms()
    x2 = []
    y2 = []
    for i in rb2.rays():
        point = i.pos()
        x2.append(point[0])
        y2.append(point[1])
    ax.plot(x2, y2, '.', label="cp lens")

    def rms_func(curvatures):
        c1, c2 = curvatures
        rb = RayBundle()
        bc = BiConvex(z_0=100., curvature1=c1, curvature2=c2, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
        screen = OutputPlane(z_0=f2)
        rb.propagate_bundle([bc, screen])
        return rb.rms()
    min_rms = minimize(rms_func, [0.02, -0.02], bounds=[(0.001, 0.1), (-0.1, -0.001)])
    c1, c2 = min_rms.x

    rb3 = RayBundle()
    bc = BiConvex(z_0=100., curvature1=c1, curvature2=c2, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    # f3 = bc.focal_point()
    screen3 = OutputPlane(z_0=f2)
    rb3.propagate_bundle([bc, screen3])
    x3 = []
    y3 = []
    for i in rb3.rays():
        point = i.pos()
        x3.append(point[0])
        y3.append(point[1])
    ax.plot(x3, y3, '.', label="bc lens")
    ax.legend()

    # fig.savefig("plots_for_marking/task17.png")

    return fig, rms_cp, min_rms.fun


@SaveOutput("task18", plot_output_indices=itemgetter(0))
def task18():
    """
    Task 18.

    In this function you will be testing your reflection modelling. Create a new SphericalReflecting surface
    and trace a RayBundle through it to the OutputPlane.This function should return
    the following items as a tuple in the following order:
    1. The track plot showing reflecting ray bundle off SphericalReflection surface.
    2. The focal point of the SphericalReflection surface.

    Returns:
        tuple[Figure, float]: The track plot, the focal point
    """
    rb = RayBundle()
    sph_refl = SphericalReflection(z_0=100., aperture=6., curvature=-0.02)
    f = sph_refl.focal_point()
    screen = OutputPlane(z_0=50.)

    rb.propagate_bundle([sph_refl, screen])
    fig = rb.track_plot(title='Task 18')
    # fig.savefig("plots_for_marking/task18.png")

    return fig, f


@SaveOutput(["task19a", "task19b"])
def task19():
    """
    Task 19.

    In this function you will be quantifing the amount of spherical aberration.
    Create a new ConvexPlano lens and trace a RayBundle through it to the OutputPlane
    located at the focal point.
    This function should return two matplotlib figures as a tuple in the following order:
    1. The plot showing the transverse spherical aberration i.e. rms as a function of beam radius at focal point.
    2. The plot showing the longitudinal spherical aberration i.e. z intercept of a ray as a function
       of transverse distance from optical axis.

    Returns:
        tuple[Figure, Figure]: Transverse spherical aberration plot, longitudinal spherical aberration plot
    """
    cp = ConvexPlano(z_0=100, curvature=0.02, n_inside=1.5168, n_outside=1., thickness=5., aperture=50.)
    f = cp.focal_point()
    screen = OutputPlane(z_0=f)

    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    bundle_rad = np.linspace(0.1, 20)
    rms = []
    z_intercepts = []
    trans_dist = []

    for i in bundle_rad:
        rb = RayBundle(rmax=i)
        rb.propagate_bundle([cp, screen])
        rms.append(rb.rms())

        for j in rb.rays():
            if j.direc()[2] == 1:
                continue

            z_intercepts.append(j.z_int())
            point = j.vertices()[0]
            trans_dist.append(np.sqrt(point[0]*point[0] + point[1]*point[1]))

    ax1.plot(bundle_rad, rms)
    ax1.set_xlabel("Bundle radius (mm)")
    ax1.set_ylabel("RMS (mm)")
    ax1.set_title("Task 19a")
    # fig1.savefig("plots_for_marking/task19a.png")

    ax2.plot(trans_dist, z_intercepts, label='z-intercepts')
    ax2.axhline(y=f, label='focal point', color='red')
    ax2.set_xlabel("Transverse distance (mm)")
    ax2.set_ylabel("z-intercept (mm)")
    ax2.set_title("Task 19b")
    ax2.legend()
    # fig2.savefig("plots_for_marking/task19b.png")

    return fig1, fig2


@SaveOutput("task20")
def task20():
    """
    Task 20.

    In this function you will investigate dispersion. Create several Rays with different wavelengths.
    Plot the paths of these rays through a BiConvex lens made of BK7 glass to an OutputPlane at z=200.
    This function should return the track plot of the path of your Rays through the glass lens.

    Returns:
        Figure: The ray track plot.
    """
    material = DispersiveMaterial(
    b_coeffs=[1.03961212, 0.231792344, 1.01046945],
    c_coeffs=[0.00600069867, 0.0200179144, 103.560653]
)
    bc = BiConvex(
        z_0=100.,
        curvature1=0.02,
        curvature2=-0.02,
        n_inside=material,
        n_outside=1.,
        thickness=5.,
        aperture=50.
    )
    op = OutputPlane(z_0=200)
    wl = np.array([415, 437.5, 472.5, 532.5, 580, 610, 685]) * 1e-9
    colour = ['violet', 'indigo', 'blue', 'green', 'yellow', 'orange', 'red']

    fig, ax = plt.subplots()
    for i, lamb in enumerate(wl):
        ray = Ray(pos=[0, 1, 0], wavelength=lamb)
        bc.propagate_ray(ray)
        op.propagate_ray(ray)
        point = np.array(ray.vertices())
        z = point[:, 2]
        y = point[:, 1]
        ax.plot(z, y, color=colour[i])

    ax.set_xlabel("z-direction (mm)")
    ax.set_ylabel("y-direction (mm)")
    ax.set_title("Task 20")
    # fig.savefig("plots_for_marking/task20.png")

    return fig


if __name__ == "__main__":

    # Run task 8 function
    task8()

    # Run task 10 function
    FIG10 = task10()

    # Run task 11 function
    FIG11, FOCAL_POINT = task11()

    # Run task 12 function
    FIG12 = task12()

    # Run task 13 function
    FIG13, TASK13_RMS = task13()

    # Run task 14 function
    FIG14, TASK14_RMS, TASK14_DIFF_SCALE = task14()

    # Run task 15 function
    FIG15_PC, FOCAL_POINT_PC, FIG15_CP, FOCAL_POINT_CP = task15()

    # Run task 16 function
    FIG16, PC_RMS, CP_RMS, TASK16_DIFF_SCALE = task16()

    # Run task 17 function
    FIG17, CP_RMS, BICONVEX_RMS = task17()

    # Run task 18 function
    FIG18, FOCAL_POINT = task18()

    # Run task 19 function
    FIG19_TVSA, FIG19_LGSA = task19()

    # Run task 20 function
    FIG20 = task20()

    plt.show()
