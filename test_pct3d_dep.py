import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, micro

import pct
import pct3d
from pct3d import time_, pos_, vel_, omg_, quat_
from pct3d import force_drag3d, force_dep3d, force_grav3d
from pct3d import torque_drag3d, torque_dep3d


def electric_field(x, y, z):
    return np.r_[pct.func_Ex(x, y)[0], pct.func_Ey(x, y)[0], 0.]

def flow_velocity(x, y, z):
    return np.r_[0., 0., 0.]

params = pct.params
params.elecf = electric_field
params.vf = flow_velocity

params.eps_p = epsilon_0*9.5
params.eps_f = epsilon_0*8.3
params.sig_p = 600
params.sig_f = 2e-7
params.rho_p = 6150
params.rho_f = 967
params.mu_f = 1.1e-3
params.freq = 1e6
params.a = 0.5*micro/2.
params.b = 3.0*micro/2.
params.dt = 0.1e-6

pct.update(params, show=True)

if __name__ == "__main__":

    def check_time(u):
        return True if time_(u) <= 0.001 else False

    def check_height(u):
        return True if pos_(u)[1] > 0 else False
    
    pos0 = 2*micro, 4*micro, 0.
    quat0 = pct.euler2quat(np.deg2rad(120), 0, 0.)
    u0 = [0, pos0, (0, 0, 0), (0, 0, 0), quat0]
    
    # Trace ellipsoid particle
    sol1 = pct3d.trace3d(u0 = u0,
            forces  = [force_dep3d, force_drag3d, force_grav3d],
            torques = [torque_dep3d, torque_drag3d],
            params  = params,
            conditions = [check_time, check_height])
    
    # Trace sphere particle
    a, b = params.a, params.b
    params.a = params.b = (a**2*b)**(1/3)
    pct.update(params, show=False)
    
    sol2 = pct3d.trace3d(u0 = u0,
                forces  = [force_dep3d, force_drag3d, force_grav3d],
                torques = [torque_dep3d, torque_drag3d],
                params  = params,
                conditions = [check_time, check_height])
    
    params.a, params.b = a, b
    pct.update(params, show=False)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,6))
    ax1 = pct3d.plot_traj3d(ax1, sol1, length=1*micro, nskip=100)
    ax1.set_title("Ellipoid (3D)", fontsize=20)
    ax2 = pct3d.plot_traj3d(ax2, sol2, length=1*micro, nskip=100)
    ax2.set_title("Sphere (3D)", fontsize=20)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plt.show()