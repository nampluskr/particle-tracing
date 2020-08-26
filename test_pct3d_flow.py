import numpy as np
import matplotlib.pyplot as plt

import pct
import pct3d
from pct3d import time_, pos_, vel_, omg_, quat_
from pct3d import force_drag3d, force_grav3d, torque_drag3d


def flow_velocity(x, y, z, U0=0.6, R=2.1e-3):
    return np.r_[2*U0*(1 - (y**2/R)**2), 0, 0]

params = pct.params
params.vf = flow_velocity

params.rho_p = 2560
params.rho_f = 1.225
params.mu_f = 1.225*0.6*4.2e-3/169
params.a = 0.5e-6
params.b = params.a*5
params.dt = 10e-6

pct.update(params, show=True)

if __name__ == "__main__":
    
    def check_time(u):
        return True if time_(u) <= 0.5 else False

    def check_height(u):
        return True if pos_(u)[1] > -2.1e-3 else False
    
    pos0 = 0, -1.65e-3, 0
    quat0 = pct.euler2quat(np.pi/2., 0, 0.)
    u0 = [0, pos0, (0, 0, 0), (0, 0, 0), quat0]
    
    # Trace ellipsoid particle
    sol1 = pct3d.trace3d(u0 = u0,
                forces  = [force_drag3d, force_grav3d],
                torques = [torque_drag3d],
                params  = params,
                conditions = [check_time, check_height])

    # Trace sphere particle
    a, b = params.a, params.b
    params.a = params.b = (a**2*b)**(1/3)
    pct.update(params, show=True)

    sol2 = pct3d.trace3d(u0 = u0,
                forces  = [force_drag3d, force_grav3d],
                torques = [torque_drag3d],
                params  = params,
                conditions = [check_time, check_height])

    params.a, params.b = a, b
    pct.update(params, show=False)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,8), sharey=True)
    ax1 = pct3d.plot_traj3d(ax1, sol1, length=20e-6, nskip=1000)
    ax1.set_title("Ellipoid (3D)", fontsize=20)
    ax1.set_ylabel("y [m]", fontsize=15)
    ax2 = pct3d.plot_traj3d(ax2, sol2, length=20e-6, nskip=1000)
    ax2.set_title("Sphere (3D)", fontsize=20)

    for ax in (ax1, ax2):
        ax.set_xlabel("x [m]", fontsize=15)
        ax.grid()

    fig.tight_layout()
    plt.show()
