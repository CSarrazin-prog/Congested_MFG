import numpy as np
from pysdot import PowerDiagram
from pysdot import OptimalTransport
from pysdot.domain_types import ConvexPolyhedraAssembly
from pysdot.radial_funcs import RadialFuncInBall

def initialize_weights(power_diagram, center, verbose=None):

    domain = power_diagram.get_domain()
    
    areas=power_diagram.integrals()

    phi = power_diagram.get_weights()


    while not np.any(areas):
        phi *= 2
        power_diagram.set_weights(phi)
        areas = power_diagram.integrals()
    zero_area=np.all(areas)
    if not zero_area:
        points=power_diagram.get_positions()
        in_dom = np.nonzero([domain.coeff_at(x) for x in points])[0]
        ratio=1
        while len(in_dom)!=len(points):
            ratio=ratio/2
            points[:]=ratio*points[:]+(1-ratio)*center
            in_dom = np.nonzero([domain.coeff_at(x) for x in points])[0]
        phi=(1-ratio)*np.sum((points-center)**2, axis=-1)
        power_diagram.set_weights(phi)


def proj_noncongested(points, domain, center=None, mass=None,
                      radial_func=RadialFuncInBall(), verbose=None):
    nb_points = len(points)
    assert(nb_points!=0)
    if mass is None:
        mass=np.ones(nb_points)/nb_points
    laguerre = OptimalTransport(positions=points, weights=None, masses=mass,
                                domain=domain, radial_func=radial_func, linear_solver="CuPyx")
    
    if not center is None:
        initialize_weights(power_diagram=laguerre.pd, center=center, verbose=verbose)

    laguerre.adjust_weights(relax=1)
    
    if np.linalg.norm(laguerre.pd.integrals() - mass) > 1e-5 :
        print("The Newton algorithm did not converge!")
        laguerre.display_vtk("debug_file/bad_Newton.vtk",points=True)
        laguerre.get_domain().display_boundaries_vtk("debug_file/bad_Newton_domain.vtk")
        np.save("debug_file/bad_positions", laguerre.get_positions())
        np.save("debug_file/integrals", laguerre.pd.integrals())
        np.save("debug_file/bad_weights", laguerre.get_weights())
        assert(False)
    return laguerre.pd

def dist_noncongested(points, domain, center=None, epsilon=1, mass=None,
                      radial_func=RadialFuncInBall(), verbose=None):
    nb_points = len(points)
    assert(nb_points!=0)
    if mass is None:
        mass=np.ones(nb_points)/nb_points
    laguerre=proj_noncongested(points=points,domain=domain,center=center,mass=mass, radial_func=radial_func,
                               verbose=verbose)
    transport_cost = 0.5/epsilon*np.sum(laguerre.second_order_moments())
    barycenters = laguerre.centroids()
    gradient = mass.reshape(nb_points,1)/epsilon*(points-barycenters)
    return transport_cost, gradient