import numpy as np
import matplotlib.pyplot as plt

import pct


## Variables [2D]: u = [t, [x, y], [vx, vy], omega, theta]

def time_(u): return u[0]   if u.ndim < 2 else u[:,0]
def pos_(u):  return u[1:3] if u.ndim < 2 else u[:,1:3]
def vel_(u):  return u[3:5] if u.ndim < 2 else u[:,3:5]
def omg_(u):  return u[5]   if u.ndim < 2 else u[:,5]
def tht_(u):  return u[6]   if u.ndim < 2 else u[:,6]


## Force & torque equations [2D]

def _total(funcs, u, params):
    return np.array([func(u, params) for func in funcs]).sum(axis=0)

def eqns2d(u, t, forces, torques, params):
    total_force  = _total(forces, u, params) if len(forces) else np.zeros(2)
    d_vel = total_force/params.mass

    total_torque = _total(torques, u, params) if len(torques) else 0.0
    d_omg = total_torque/params.inertia_short

    return np.r_[1.0, vel_(u), d_vel, d_omg, omg_(u)]

def trace2d(u0, forces, torques, params, conditions):
    return pct.trace(eqns2d, u0, forces, torques, params, conditions)


## 중력 [2D]
from scipy.constants import g

def force_grav2d(u, params):
    return np.r_[0., -(params.rho_p - params.rho_f)*params.vol*g]


## Drag [2D]

from numpy import log, sqrt

def force_drag2d(u, params):
    r, r2 = params.r, params.r2
    if r > 1:
        K22 = 16*(r2-1)/((2*r2-3)*log(r+sqrt(r2-1))/sqrt(r2-1) + r)
        K11 = 8*(r2-1)/((2*r2-1)*log(r+sqrt(r2-1))/sqrt(r2-1) - r)
    else:
        K11 = K22 = 6

    K_12 = params.mu_f*np.pi*params.a*np.diag([K11, K22])
    R, (x, y) = pct.rot2d(tht_(u)), pos_(u)

    return (R.T).dot(K_12).dot(R).dot(params.vf(x, y) - vel_(u))

def torque_drag2d(u, params):
    R, (x, y) = pct.rot2d(tht_(u)), pos_(u)
    L_xy = pct.jac2d(params.vf, x, y)
    L_12 = R.dot(L_xy).dot(R.T)  # velocity gradient
    D_12 = (L_12 + L_12.T)/2     # rate of deformation
    W_12 = (L_12 - L_12.T)/2     # rate of rotation

    r, r2 = params.r, params.r2
    if r > 1:
        a0 = -2/(r2-1) - r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5
        b0 = r2/(r2-1) + r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5/2
    else:
        a0 = b0 = 2/3.

    k3 = 16*np.pi*params.mu_f*params.a**3*r/(r2*a0 + b0)/3.
    t3 = (r**2 - 1)*D_12[0,1] + (1 + r**2)*(W_12[1,0] - omg_(u))

    return k3*t3


## DEP [2D]

def force_dep2d(u, params):
    R, (x, y) = pct.rot2d(tht_(u)), pos_(u)
    Kcm = np.diag([params.cmf_long, params.cmf_short])
    dip = params.vol*params.eps_f*Kcm.dot(R).dot(params.elecf(x, y))

    return pct.jac2d(params.elecf, x, y).dot(R.T).dot(dip)

def torque_dep2d(u, params):
    R, (x, y) = pct.rot2d(tht_(u)), pos_(u)
    Kcm = np.diag([params.cmf_long, params.cmf_short])
    E_12 = R.dot(params.elecf(x, y))
    dip = params.vol*params.eps_f*Kcm.dot(E_12)

    return np.cross(dip, E_12)


## Post-pocessing

def plot_traj2d(ax, sol, length, nskip=100):
    sol = sol[::nskip]
    x, y = pos_(sol).T
    b = length/2.
    pt0 = pos_(sol)
    pt1 = np.array([pos_(s) + pct.rot2d(tht_(s)).dot([ b, 0]) for s in sol])
    pt2 = np.array([pos_(s) + pct.rot2d(tht_(s)).dot([-b, 0]) for s in sol])

    for (x0, y0), (x1, y1), (x2, y2) in zip(pt0, pt1, pt2):
        ax.plot([x0, x1], [y0, y1], 'r-', lw=2)
        ax.plot([x0, x2], [y0, y2], 'g-', lw=2)

    ax.plot(x, y, 'k:', lw=2)
    return ax


if __name__ == "__main__":

    pass