import numpy as np
from common import solve, func_dxE2, func_dyE2


## Variables [3D]: u = [t, [x,y,z], [vx,vy,vz], [w1,w2,w3], [e1,e2,e3,e0]]

def time_(u): return u[0]    if u.ndim < 2 else u[:,0]
def pos_(u):  return u[1:4]  if u.ndim < 2 else u[:,1:4]
def vel_(u):  return u[4:7]  if u.ndim < 2 else u[:,4:7]
def ang_(u):  return u[7:11] if u.ndim < 2 else u[:,7:11]
def omg_(u):  return u[11:]  if u.ndim < 2 else u[:,11:]


## Functions [3D]

def rotation(ang):
    e1, e2, e3, e0 = ang
    return np.array([[1-2*(e2**2+e3**2), 2*(e1*e2+e3*e0), 2*(e1*e3-e2*e0)],
                    [2*(e2*e1-e3*e0), 1-2*(e3**2+e1**2), 2*(e2*e3+e1*e0)],
                    [2*(e3*e1+e2*e0), 2*(e3*e2-e1*e0), 1-2*(e1**2+e2**2)]]).T

def jacobian(f, pos, h=1e-7):
    x, y, z = pos
    dfdx = (f(x+h, y, z) - f(x-h, y, z))/h/2.
    dfdy = (f(x, y+h, z) - f(x, y-h, z))/h/2.
    dfdz = (f(x, y, z+h) - f(x, y, z-h))/h/2.
    return np.array([dfdx, dfdy, dfdz]).T

def euler2quat(psi, theta, phi):
    """ Euler angles in Z-X-Z rotation """
    e1 = np.cos((phi - psi)/2)*np.sin(theta/2)
    e2 = np.sin((phi - psi)/2)*np.sin(theta/2)
    e3 = np.sin((phi + psi)/2)*np.cos(theta/2)
    e0 = np.cos((phi + psi)/2)*np.cos(theta/2)
    return np.r_[e1, e2, e3, e0]


## 운동방정식/토크방정식 [3D]

def _total(funcs, u, params):
    return np.array([func(u, params) for func in funcs]).sum(axis=0)

def eqns(u, t, forces, torques, params):
    total_force = _total(forces, u, params) if len(forces) else np.zeros(3)
    total_torque = _total(torques, u, params) if len(torques) else np.zeros(3)

    I11 = params.inertia_long
    I22 = I33 = params.inertia_short
    I, I_inv = np.diag([I11, I22, I33]), np.diag([1/I11, 1/I22, 1/I33])
    e1, e2, e3, e0 = ang_(u)
    mat = np.array([[e0,-e3,e2], [e3,e0,-e1], [-e2,e1,e0], [-e1,-e2,-e3]])

    d_time = 1.0
    d_pos = vel_(u)
    d_vel = total_force/params.mass
    d_ang = mat.dot(omg_(u))/2.
    d_omg = I_inv.dot(total_torque - np.cross(omg_(u), I.dot(omg_(u))))
    return np.r_[d_time, d_pos, d_vel, d_ang, d_omg]

def trace(u0, forces, torques, params, conditions):
    return solve(eqns, u0, forces, torques, params, conditions)


## 중력 [3D]
from scipy.constants import g

def force_grav(u, params):
    return np.r_[0., -(params.rho_p - params.rho_f)*params.vol*g, 0.]


## Drag - 구형체(Sphere) [3D]

def force_drag_sph(u, params):
    x, y, z = pos_(u)
    Kdrag = 6*np.pi*params.mu_f*params.a
    return Kdrag*(params.vf(x, y, z) - vel_(u))

def torque_drag_sph(u, params):
    R = rotation(ang_(u))
    L_xyz = jacobian(params.vf3d, pos_(u))
    L= R.dot(L_xyz).dot(R.T)   # velocity gradient
    W= (L - L.T)/2             # rate of rotation
    w1, w2, w3 = omg_(u)
    Ktorq = 8*np.pi*params.mu_f*params.a**3
    return Ktorq*np.r_[W[2,1] - w1, W[0,2] - w2, W[1,0] - w3]


## Drag - 타원체(Spheroid) [3D]

def force_drag(u, params):
    K11 = params.kdrag_long
    K22 = K33 = params.kdrag_short
    Kdrag = params.mu_f*np.pi*params.a*np.diag([K11, K22, K33])
    R, (x, y, z) = rotation(ang_(u)), pos_(u)
    return (R.T).dot(Kdrag).dot(R).dot(params.vf(x, y, z) - vel_(u))

def torque_drag(u, params):
    R = rotation(ang_(u))
    L_xyz = jacobian(params.vf, pos_(u))
    L = R.dot(L_xyz).dot(R.T)   # velocity gradient
    D = (L + L.T)/2             # rate of deformation
    W = (L - L.T)/2             # rate of rotation

    K11 = params.ktorq_long
    K22 = K33 = params.ktorq_short
    Ktorq = params.mu_f*np.pi*params.a**3*np.diag([K11, K22, K33])

    r, (w1, w2, w3) = params.r, omg_(u)
    t1 = W[2,1] - w1
    t2 = (1 - r**2)*D[0,2]/2. + (1 + r**2)*(W[0,2] - w2)/2.
    t3 = (r**2 - 1)*D[1,0]/2. + (r**2 + 1)*(W[1,0] - w3)/2.
    return Ktorq.dot([t1, t2, t3])


## DEP - 구형체(Sphere) [3D]

def force_dep_sph(u, params):
    x, y, z = pos_(u)
    eps_p = params.eps_p - 1J*params.sig_p/np.pi/params.freq/2.
    eps_f = params.eps_f - 1J*params.sig_f/np.pi/params.freq/2.
    cmf = 3*(eps_p - eps_f)/(eps_p + 2*eps_f)
    grad_E2 = np.r_[func_dxE2(x, y)[0], func_dyE2(x, y)[0], 0.]
    return params.vol*params.eps_f*cmf.real*grad_E2/2.

def torque_dep_sph(u, params):
    return np.r_[0., 0., 0.]


## DEP - 타원체(Spheroid) [3D]

def force_dep(u, params):
    R, (x, y, z) = rotation(ang_(u)), pos_(u)
    Kcm = np.diag([params.kcm_long, params.kcm_short, params.kcm_short])
    Elec = R.dot(params.elec(x, y, z))
    dip = params.vol*params.eps_f*Kcm.dot(Elec)
    return jacobian(params.elec, pos_(u)).dot(R.T).dot(dip)

def torque_dep(u, params):
    R, (x, y, z) = rotation(ang_(u)), pos_(u)
    Kcm = np.diag([params.kcm_long, params.kcm_short, params.kcm_short])
    Elec = R.dot(params.elec(x, y, z))
    dip = params.vol*params.eps_f*Kcm.dot(Elec)
    return np.cross(dip, Elec)


## Post-pocessing

def plot_traj(ax, sol, length, nskip=100):
    sol = sol[::nskip]
    x, y, z = pos_(sol).T
    b = length/2.
    pt0 = pos_(sol)
    pt1 = np.array([pos_(s) + rotation(ang_(s)).dot([ b, 0, 0]) for s in sol])
    pt2 = np.array([pos_(s) + rotation(ang_(s)).dot([-b, 0, 0]) for s in sol])

    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in zip(pt0, pt1, pt2):
        ax.plot([x0, x1], [y0, y1], 'r-', lw=2)
        ax.plot([x0, x2], [y0, y2], 'g-', lw=2)

    ax.plot(x, y, 'k:', lw=2)
    return ax


if __name__ == "__main__":

    pass