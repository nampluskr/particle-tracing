import numpy as np
import matplotlib.pyplot as plt

import pct


## Variables [3D]: u = [t, [x,y,z], [vx,vy,vz], [w1,w2,w3], [e1,e2,e3,e0]]

def time_(u): return u[0]    if u.ndim < 2 else u[:,0]
def pos_(u):  return u[1:4]  if u.ndim < 2 else u[:,1:4].T
def vel_(u):  return u[4:7]  if u.ndim < 2 else u[:,4:7].T
def omg_(u):  return u[7:10] if u.ndim < 2 else u[:,7:10].T
def quat_(u): return u[10:]  if u.ndim < 2 else u[:,10:].T


## 운동방정식/토크방정식 [3D]

def _total(funcs, u, params):
    return np.array([func(u, params) for func in funcs]).sum(axis=0)

def eqns3d(u, t, forces, torques, params):
    total_force  = _total(forces, u, params) if len(forces) else np.zeros(3)
    d_vel = total_force/params.mass

    total_torque = _total(torques, u, params) if len(torques) else np.zeros(3)
    omg = omg_(u)
    I11 = params.inertia_long
    I22 = I33 = params.inertia_short
    I, I_inv = np.diag([I11, I22, I33]), np.diag([1/I11, 1/I22, 1/I33])
    d_omg = I_inv.dot(total_torque - np.cross(omg, I.dot(omg)))

    e1, e2, e3, e0 = quat_(u)
    mat = np.array([[e0,-e3,e2], [e3,e0,-e1], [-e2,e1,e0], [-e1,-e2,-e3]])
    d_quat = mat.dot(omg)/2.

    return np.r_[1.0, vel_(u), d_vel, d_omg, d_quat]

def trace3d(u0, forces, torques, params, conditions):
    return pct.trace(eqns3d, u0, forces, torques, params, conditions)


## 중력 [3D]
from scipy.constants import g

def force_grav3d(u, params):
    return np.r_[0., -(params.rho_p - params.rho_f)*params.vol*g, 0.]


## Drag - 구형체(Sphere) [3D]

def force_drag3d_sph(u, params):
    x, y, z = pos_(u)
    return 6*np.pi*params.mu_f*params.a*(params.vf(x, y, z) - vel_(u))

def torque_drag3d_sph(u, params):
    R, (x, y, z) = pct.rot3d(*quat_(u)), pos_(u)
    L_xyz = pct.jac3d(params.vf, x, y, z)
    L_123 = R.dot(L_xyz).dot(R.T)   # velocity gradient
    W_123 = (L_123 - L_123.T)/2     # rate of rotation

    a0 = b0 = c0 = 2/3.
    k1 = k2 = k3 = 4*np.pi*params.mu_f*params.a**3

    omg1, omg2, omg3 = omg_(u)
    t1 = 2*(W_123[2,1] - omg1)
    t2 = 2*(W_123[0,2] - omg2)
    t3 = 2*(W_123[1,0] - omg3)

    return np.r_[k1*t1, k2*t2, k3*t3]


## Drag - 타원체(Spheroid) [3D]

from numpy import log, sqrt

def force_drag3d(u, params):
    r, r2 = params.r, params.r2
    if r > 1:
        K11 = 8*(r2-1)/((2*r2-1)*log(r+sqrt(r2-1))/sqrt(r2-1) - r)
        K22 = K33 = 16*(r2-1)/((2*r2-3)*log(r+sqrt(r2-1))/sqrt(r2-1) + r)

    else:
        K11 = K22 = K33 = 6

    K_123 = params.mu_f*np.pi*params.a*np.diag([K11, K22, K33])
    R, (x, y, z) = pct.rot3d(*quat_(u)), pos_(u)

    return (R.T).dot(K_123).dot(R).dot(params.vf(x, y, z) - vel_(u))

def torque_drag3d(u, params):
    R, (x, y, z) = pct.rot3d(*quat_(u)), pos_(u)
    L_xyz = pct.jac3d(params.vf, x, y, z)
    L_123 = R.dot(L_xyz).dot(R.T)   # velocity gradient
    D_123 = (L_123 + L_123.T)/2     # rate of deformation
    W_123 = (L_123 - L_123.T)/2     # rate of rotation

    r, r2 = params.r, params.r2
    if r > 1:
        a0 = -2/(r2-1) - r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5
        b0 = c0 = r2/(r2-1) + r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5/2
    else:
        a0 = b0 = c0 = 2/3.

    omg1, omg2, omg3 = omg_(u)
    k1 = 16*np.pi*params.mu_f*params.a**3*r/(b0 + c0)/3.
    k2 = 16*np.pi*params.mu_f*params.a**3*r/(c0 + r2*a0)/3.
    k3 = 16*np.pi*params.mu_f*params.a**3*r/(r2*a0 + b0)/3.

    t1 = 2*(W_123[2,1] - omg1)
    t2 = (1 - r**2)*D_123[0,2] + (1 + r**2)*(W_123[0,2] - omg2)
    t3 = (r**2 - 1)*D_123[1,0] + (r**2 + 1)*(W_123[1,0] - omg3)

    return np.r_[k1*t1, k2*t2, k3*t3]


## DEP - 구형체(Sphere) [3D]

def force_dep3d_sph(u, params):
    x, y, z = pos_(u)
    eps_p = params.eps_p - 1J*params.sig_p/np.pi/params.freq/2.
    eps_f = params.eps_f - 1J*params.sig_f/np.pi/params.freq/2.
    cmf = (eps_p - eps_f)/(eps_p + 2*eps_f)
    grad_E2 = np.r_[pct.func_dxE2(x, y)[0], pct.func_dyE2(x, y)[0], 0.]

    return 4*np.pi*params.a**3*params.eps_f*cmf.real*grad_E2/2.

def torque_dep3d_sph(u, params):
    return np.r_[0., 0., 0.]


## DEP - 타원체(Spheroid) [3D]

def force_dep3d(u, params):
    R, (x, y, z) = pct.rot3d(*quat_(u)), pos_(u)
    Kcm = np.diag([params.cmf_long, params.cmf_short, params.cmf_short])
    dip = params.vol*params.eps_f*Kcm.dot(R).dot(params.elecf(x, y, z))

    return pct.jac3d(params.elecf, x, y, z).dot(R.T).dot(dip)

def torque_dep3d(u, params):
    R, (x, y, z) = pct.rot3d(*quat_(u)), pos_(u)
    Kcm = np.diag([params.cmf_long, params.cmf_short, params.cmf_short])
    E_123 = R.dot(params.elecf(x, y, z))
    dip = params.vol*params.eps_f*Kcm.dot(E_123)

    return np.cross(dip, E_123)


## Post-pocessing

def plot_traj3d(ax, sol, length, nskip=100):
    sol = sol[::nskip]
    x, y, z = pos_(sol)
    b = length/2.
    pt0 = pos_(sol).T
    pt1 = np.array([pos_(s) + pct.rot3d(*quat_(s)).dot([ b, 0, 0]) for s in sol])
    pt2 = np.array([pos_(s) + pct.rot3d(*quat_(s)).dot([-b, 0, 0]) for s in sol])

    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in zip(pt0, pt1, pt2):
        ax.plot([x0, x1], [y0, y1], 'r-', lw=2)
        ax.plot([x0, x2], [y0, y2], 'g-', lw=2)

    ax.plot(x, y, 'k:', lw=2)
    return ax


if __name__ == "__main__":

    pass