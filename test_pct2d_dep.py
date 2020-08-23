import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import epsilon_0, micro

import pct
import pct2d
from pct2d import time_, pos_, vel_, omg_, tht_
from pct2d import force_drag2d, force_dep2d, force_grav2d
from pct2d import torque_drag2d, torque_dep2d


def electric_field(x, y):
    return np.r_[pct.func_Ex(x, y)[0], pct.func_Ey(x, y)[0]]

def flow_velocity(x, y):
    return np.r_[0., 0.]

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

    u0 = [0, (2*micro, 4*micro), (0, 0), 0, np.deg2rad(120)]
    
    # Trace ellipsoid particle
    sol1 = pct2d.trace2d(u0 = u0,
                forces  = [force_dep2d, force_drag2d, force_grav2d],
                torques = [torque_dep2d, torque_drag2d],
                params  = params,
                conditions = [check_time, check_height])
    
    # Trace sphere particle
    a, b = params.a, params.b
    params.a = params.b = (a**2*b)**(1/3)
    pct.update(params, show=False)
    
    sol2 = pct2d.trace2d(u0 = u0,
                forces  = [force_dep2d, force_drag2d, force_grav2d],
                torques = [torque_dep2d, torque_drag2d],
                params  = params,
                conditions = [check_time, check_height])
    
    params.a, params.b = a, b
    pct.update(params, show=False)
    
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,6))
    ax1 = pct2d.plot_traj2d(ax1, sol1, length=1*micro, nskip=100)
    ax1.set_title("Ellipoid (2D)", fontsize=20)
    ax2 = pct2d.plot_traj2d(ax2, sol2, length=1*micro, nskip=100)
    ax2.set_title("Sphere (2D)", fontsize=20)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plt.show()

