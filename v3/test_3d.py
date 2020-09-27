import numpy as np
import matplotlib.pyplot as plt

import pct3d_quat as pct


## User-defined electric field
from scipy.special import ellipe, legendre
from scipy.interpolate import interp2d

def potential(x, y, vol, d1, d2, nmax=50):
    c, d = np.pi*d1/(d1+d2), 2*np.pi/(d1+d2)
    sum_phi, sum_an = 0, 0

    for n in range(1, nmax+1):
        ln, an = n-0.5, legendre(n-1)(np.cos(c))/ellipe(np.cos(c/2.))
        sum_phi += an*np.cos(ln*x*d)*np.exp(-ln*y*1E6)/ln
        sum_an += an/ln

    return vol*sum_phi/np.abs(sum_an)

x = np.linspace(-4, 14, 1001)*1e-6
y = np.linspace(0, 8, 501)*1e-6
gridx, gridy = np.meshgrid(x, y)

grid_phi = potential(gridx, gridy, vol=10, d1=8e-6, d2=2e-6)
grid_Ey, grid_Ex = np.gradient(-grid_phi, y, x)
grid_dyEx, grid_dxEx = np.gradient(grid_Ex, y, x)
grid_dyEy, grid_dxEy = np.gradient(grid_Ey, y, x)

func_Ex = interp2d(x, y, grid_Ex)
func_Ey = interp2d(x, y, grid_Ey)


## Test DEP
if 1:

    def electric_field(x, y, z):
        return np.r_[func_Ex(x, y)[0], func_Ey(x, y)[0], 0.]


    pm_dep = pct.default_params()

    pm_dep.elec = electric_field
    pm_dep.eps_p = 9.5
    pm_dep.eps_f = 8.3
    pm_dep.sig_p = 600
    pm_dep.sig_f = 2e-7
    pm_dep.rho_p = 6150
    pm_dep.rho_f = 967
    pm_dep.mu_f = 1.1e-3
    pm_dep.freq = 1e6
    pm_dep.a = 0.5e-6/2.
    pm_dep.b = 3.0e-6/2.
    pm_dep.dt = 0.1e-6

    pct.update(pm_dep, show=True)

    def check_time(u):
        return True if pct.time_(u) <= 0.001 else False

    def check_height(u):
        return True if pct.pos_(u)[1] > 0 else False

    pos0 = 6e-6, 4e-6, 0.
    ang0 = pct.euler2quat(np.deg2rad(0), 0, 0.)

    u0 = [0, pos0, (0, 0, 0), ang0, (0, 0, 0)]
    forces  = [pct.force_dep, pct.force_drag, pct.force_grav]
    torques = [pct.torque_dep, pct.torque_drag]
    conditions = [check_time, check_height]

    # Trace ellipsoid particle
    sol1 = pct.trace(u0, forces, torques, pm_dep, conditions)

    # Trace sphere particle
    a, b = pm_dep.a, pm_dep.b
    pm_dep.a = pm_dep.b = (a**2*b)**(1/3)
    pct.update(pm_dep)

    sol2 = pct.trace(u0, forces, torques, pm_dep, conditions)

    pm_dep.a, pm_dep.b = a, b
    pct.update(pm_dep)

    # Plot results:
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(6,6))
    ax1 = pct.plot_traj(ax1, sol1, length=1e-6, nskip=50)
    ax1.set_title("Ellipoid (3D)", fontsize=20)
    ax2 = pct.plot_traj(ax2, sol2, length=1e-6, nskip=50)
    ax2.set_title("Sphere (3D)", fontsize=20)
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plt.show()

## Test flow
if 1:
    def flow_velocity(x, y, z, U0=0.6, R=2.1e-3):
        return np.r_[2*U0*(1 - (y/R)**2), 0, 0]

    pm_flow = pct.default_params()

    pm_flow.vf = flow_velocity
    pm_flow.rho_p = 2560
    pm_flow.rho_f = 1.225
    pm_flow.mu_f = 1.225*0.6*4.2e-3/169
    pm_flow.a = 0.5e-6
    pm_flow.b = pm_flow.a*14
    pm_flow.dt = 10e-6

    pct.update(pm_flow, show=True)

    def check_time(u):
        return True if pct.time_(u) <= 0.2 else False

    def check_height(u):
        return True if pct.pos_(u)[1] > -2.1e-3 else False

    pos0 = 0, -1.65e-3, 0
    ang0 = pct.euler2quat(np.pi/2., 0., 0.)

    u0 = [0, pos0, (0, 0, 0), ang0, (0, 0, 0)]
    forces  = [pct.force_drag, pct.force_grav]
    torques = [pct.torque_drag]
    conditions = [check_time, check_height]

    # Trace ellipsoid particle
    sol1 = pct.trace(u0, forces, torques, pm_flow, conditions)

    # Trace sphere particle
    a, b = pm_flow.a, pm_flow.b
    pm_flow.a = pm_flow.b = (a**2*b)**(1/3)
    pct.update(pm_flow)

    sol2 = pct.trace(u0, forces, torques, pm_flow, conditions)

    pm_flow.a, pm_flow.b = a, b
    pct.update(pm_flow)

    # Plot results:
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ax1 = pct.plot_traj(ax1, sol1, length=20e-6, nskip=100)
    ax1.set_title("Ellipoid (3D)", fontsize=20)
    ax1.set_ylabel("y [m]", fontsize=15)
    ax2 = pct.plot_traj(ax2, sol2, length=20e-6, nskip=100)
    ax2.set_title("Sphere (3D)", fontsize=20)

    for ax in (ax1, ax2):
        ax.set_xlabel("x [m]", fontsize=15)
        ax.grid()

    fig.tight_layout()
    plt.show()