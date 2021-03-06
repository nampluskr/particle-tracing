import numpy as np
import matplotlib.pyplot as plt

import pct
import pct2d
from pct2d import time_, pos_, vel_, omg_, tht_
from pct2d import force_drag2d, force_grav2d, torque_drag2d


def flow_velocity(x, y, U0=0.6, R=2.1e-3):
    return np.r_[2*U0*(1 - (y/R)**2), 0]

params = pct.params
params.vf = flow_velocity

params.rho_p = 2560
params.rho_f = 1.225
params.mu_f = 1.225*0.6*4.2e-3/169
params.a = 0.5e-6
params.b = params.a*14
params.dt = 10e-6

pct.update(params, show=True)

if __name__ == "__main__":
    
    def check_time(u):
        return True if time_(u) <= 0.2 else False

    def check_height(u):
        return True if pos_(u)[1] > -2.1e-3 else False
    
    u0 = [0, (0, -1.65e-3), (0, 0), 0, np.deg2rad(90)]
    
    # Trace ellipsoid particle
    sol1 = pct2d.trace2d(u0 = u0,
                forces  = [force_drag2d, force_grav2d],
                torques = [torque_drag2d],
                params  = params,
                conditions = [check_time, check_height])

    # Trace sphere particle
    a, b = params.a, params.b
    params.a = params.b = (a**2*b)**(1/3)
    pct.update(params, show=True)

    sol2 = pct2d.trace2d(u0 = u0,
                forces  = [force_drag2d, force_grav2d],
                torques = [torque_drag2d],
                params  = params,
                conditions = [check_time, check_height])

    params.a, params.b = a, b
    pct.update(params, show=False)
    
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ax1 = pct2d.plot_traj2d(ax1, sol1, length=20e-6, nskip=100)
    ax1.set_title("Ellipoid (2D)", fontsize=20)
    ax1.set_ylabel("y [m]", fontsize=15)
    ax2 = pct2d.plot_traj2d(ax2, sol2, length=20e-6, nskip=100)
    ax2.set_title("Sphere (2D)", fontsize=20)

    for ax in (ax1, ax2):
        ax.set_xlabel("x [m]", fontsize=15)
        ax.grid()

    fig.tight_layout()
    plt.show()