import numpy as np
from common import solve, func_dxE2, func_dyE2


## Variables [2D]: u = [t, [x,y], [vx,vy], ang, omg]

def time_(u): return u[0]   if u.ndim < 2 else u[:,0]
def pos_(u):  return u[1:3] if u.ndim < 2 else u[:,1:3]
def vel_(u):  return u[3:5] if u.ndim < 2 else u[:,3:5]
def ang_(u):  return u[5]   if u.ndim < 2 else u[:,5]
def omg_(u):  return u[6]   if u.ndim < 2 else u[:,6]


## Functions [2D]

def rotation(ang):
    return np.array([[ np.cos(ang), np.sin(ang)],
                     [-np.sin(ang), np.cos(ang)]]).T

def jacobian(f, pos, h=1e-7):
    x, y = pos
    dfdx = (f(x+h, y) - f(x-h, y))/h/2.
    dfdy = (f(x, y+h) - f(x, y-h))/h/2.
    return np.array([dfdx, dfdy]).T
 

## 운동방정식/토크방정식 [2D]

def _total(funcs, u, params):
    return np.array([func(u, params) for func in funcs]).sum(axis=0)

def eqns(u, t, forces, torques, params):
    total_force  = _total(forces, u, params) if len(forces) else np.zeros(2)
    total_torque = _total(torques, u, params) if len(torques) else 0.0

    d_time = 1
    d_pos = vel_(u)
    d_vel = total_force/params.mass
    d_ang = omg_(u)
    d_omg = total_torque/params.inertia_short

    return np.r_[d_time, d_pos, d_vel, d_ang, d_omg]

def trace(u0, forces, torques, params, conditions):
    return solve(eqns, u0, forces, torques, params, conditions)


## 중력 [2D]
from scipy.constants import g

def force_grav(u, params):
    return np.r_[0., -(params.rho_p - params.rho_f)*params.vol*g]


## Drag - 구형체(Sphere) [2D]

def force_drag_sph(u, params):
    x, y = pos_(u)
    Kdrag = 6*np.pi*params.mu_f*params.a
    return Kdrag*(params.vf(x, y) - vel_(u))

def torque_drag_sph(u, params):
    R = rotation(ang_(u))
    L_xy = jacobian(params.vf3d, pos_(u))
    L= R.dot(L_xy).dot(R.T)   # velocity gradient
    W= (L - L.T)/2             # rate of rotation
    w3 = omg_(u)
    Ktorq = 8*np.pi*params.mu_f*params.a**3
    return Ktorq*(W[1,0] - w3)


## Drag - 타원체(Spheroid) [2D]

def force_drag(u, params):
    K11 = params.kdrag_long
    K22 = params.kdrag_short
    Kdrag = params.mu_f*np.pi*params.a*np.diag([K11, K22])
    R, (x, y) = rotation(ang_(u)), pos_(u)
    return (R.T).dot(Kdrag).dot(R).dot(params.vf(x, y) - vel_(u))

def torque_drag(u, params):
    R = rotation(ang_(u))
    L_xy = jacobian(params.vf, pos_(u))
    L = R.dot(L_xy).dot(R.T)   # velocity gradient
    D = (L + L.T)/2            # rate of deformation
    W = (L - L.T)/2            # rate of rotation

    Ktorq = params.mu_f*np.pi*params.a**3*params.ktorq_short
    r, w3 = params.r, omg_(u)
    t3 = (r**2 - 1)*D[1,0]/2. + (r**2 + 1)*(W[1,0] - w3)/2.
    return Ktorq*t3


## DEP - 구형체(Sphere) [2D]

def force_dep_sph(u, params):
    x, y = pos_(u)
    eps_p = params.eps_p - 1J*params.sig_p/np.pi/params.freq/2.
    eps_f = params.eps_f - 1J*params.sig_f/np.pi/params.freq/2.
    cmf = 3*(eps_p - eps_f)/(eps_p + 2*eps_f)
    grad_E2 = np.r_[func_dxE2(x, y)[0], func_dyE2(x, y)[0]]
    return params.vol*params.eps_f*cmf.real*grad_E2/2.

def torque_dep_sph(u, params):
    return np.r_[0., 0.]


## DEP - 타원체(Spheroid) [2D]

def force_dep(u, params):
    R, (x, y) = rotation(ang_(u)), pos_(u)
    Kcm = np.diag([params.kcm_long, params.kcm_short])
    Elec = R.dot(params.elec(x, y))
    dip = params.vol*params.eps_f*Kcm.dot(Elec)
    return jacobian(params.elec, pos_(u)).dot(R.T).dot(dip)

def torque_dep(u, params):
    R, (x, y) = rotation(ang_(u)), pos_(u)
    Kcm = np.diag([params.kcm_long, params.kcm_short])
    Elec = R.dot(params.elec(x, y))
    dip = params.vol*params.eps_f*Kcm.dot(Elec)
    return np.cross(dip, Elec)


## Post-pocessing

def plot_traj(ax, sol, length, nskip=100):
    sol = sol[::nskip]
    x, y = pos_(sol).T
    b = length/2.
    pt0 = pos_(sol)
    pt1 = np.array([pos_(s) + rotation(ang_(s)).dot([ b, 0]) for s in sol])
    pt2 = np.array([pos_(s) + rotation(ang_(s)).dot([-b, 0]) for s in sol])

    for (x0, y0), (x1, y1), (x2, y2) in zip(pt0, pt1, pt2):
        ax.plot([x0, x1], [y0, y1], 'r-', lw=2)
        ax.plot([x0, x2], [y0, y2], 'g-', lw=2)

    ax.plot(x, y, 'k:', lw=2)
    return ax


if __name__ == "__main__":

    pass
