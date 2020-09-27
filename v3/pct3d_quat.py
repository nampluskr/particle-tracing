import numpy as np
import time
from argparse import Namespace

from numpy import pi, log, sqrt, sin, cos
from scipy.constants import epsilon_0, g


def default_params():
    param = {}
    param = Namespace(**param)

    ## Particle properties (default)
    param.a = 1e-6         # [m] semi-major axis of particle
    param.b = 1e-6         # [m] semi-minor axis of particle
    param.rho_p = 1e3      # [Kg/m^3] density of particle
    param.eps_p = 1.0      # relative permitivity of particle
    param.sig_p = 100      # [S/m] conductivty of particle

    ## Fluid properties (default)
    param.rho_f = 1e3      # [Kg/m^3] density of fluid
    param.eps_f = 1.0      # relative permitivity of fluid
    param.sig_f = 1e-6     # [S/m] conductivity of fluid

    ## Simulation parameters (default)
    param.freq = 1e6       # [Hz] AC frequency of electric field
    param.dt = 0.1e-6      # [s] time step

    # Vector fields
    param.vf   = lambda x, y, z: np.zeros(3)
    param.elec = lambda x, y, z: np.zeros(3)

    return param


def update(param, show=False):
    param.r  = param.b/param.a
    param.vol = (4/3)*pi*param.a**2*param.b
    param.mass = param.vol*param.rho_p

    # Moments of inertia
    inertia_major = 0.4*param.mass*param.a**2
    inertia_minor = 0.2*param.mass*(param.a**2 + param.b**2)
    param.I = np.diag([inertia_major, inertia_minor, inertia_minor])
    param.I_inv = np.diag(1/np.diag(param.I))

    r, r2 = param.r, param.r**2

    # Depolarization factors (= 1/3 for a sphere)
    depol_major = 1/(1 + 1.6*r + 0.4*r**2)
    depol_minor = (1 - depol_major)/2.

    # Clausius-Mosostti factors
    eps_p = epsilon_0*param.eps_p - 1J*param.sig_p/pi/param.freq/2.
    eps_f = epsilon_0*param.eps_f - 1J*param.sig_f/pi/param.freq/2.

    cm_major = (eps_p-eps_f)/((eps_p-eps_f)*depol_major + eps_f)
    cm_minor = (eps_p-eps_f)/((eps_p-eps_f)*depol_minor + eps_f)

    Kpol = np.diag([cm_major.real, cm_minor.real, cm_minor.real])
    param.Kpol = param.vol*epsilon_0*param.eps_f*Kpol

    # Drag force coefficients (= 6 for a sphere)
    if r > 1:
        drag_major =  8*(r2-1)/((2*r2-1)*log(r+sqrt(r2-1))/sqrt(r2-1) - r)
        drag_minor = 16*(r2-1)/((2*r2-3)*log(r+sqrt(r2-1))/sqrt(r2-1) + r)
    else:
        drag_major = drag_minor = 6

    Kdrag = np.diag([drag_major, drag_minor, drag_minor])
    param.Kdrag = param.mu_f*pi*param.a*Kdrag

    # Drag torque coeffients (= 8 for a sphere)
    if r > 1:
        a0 = -2/(r2-1) - r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5
        b0 = r2/(r2-1) + r*log((r-sqrt(r2-1))/(r+sqrt(r2-1)))/(r2-1)**1.5/2
    else:
        a0 = b0 = 2/3.

    torq_major = 16*r/b0/3.
    torq_minor = 32*r/(r2*a0 + b0)/3.

    Ktorq = np.diag([torq_major, torq_minor, torq_minor])
    param.Ktorq = param.mu_f*pi*param.a**3*Ktorq

    if show:
        print('\n' + '='*60)
        print("# Simulation parameters:\n" + '='*60)
        for key, value in sorted(param.__dict__.items()):
            print("{:13s} = {}".format(key, value))
        print('='*60)


## Variables [3D]: u = [t, [x,y,z], [vx,vy,vz], [q0,q1,q2,q3], [w1,w2,w3]]

def time_(u): return u[0]    if u.ndim < 2 else u[:,0]
def pos_(u):  return u[1:4]  if u.ndim < 2 else u[:,1:4]
def vel_(u):  return u[4:7]  if u.ndim < 2 else u[:,4:7]
def ang_(u):  return u[7:11] if u.ndim < 2 else u[:,7:11]
def omg_(u):  return u[11:]  if u.ndim < 2 else u[:,11:]


## Functions

def rotation(ang):
    q0, q1, q2, q3 = ang
    return np.array([[1-2*(q2**2+q3**2), 2*(q1*q2+q3*q0), 2*(q1*q3-q2*q0)],
                    [2*(q2*q1-q3*q0), 1-2*(q3**2+q1**2), 2*(q2*q3+q1*q0)],
                    [2*(q3*q1+q2*q0), 2*(q3*q2-q1*q0), 1-2*(q1**2+q2**2)]])


def jacobian(f, pos, h=1e-7):
    x, y, z = pos
    dfdx = (f(x+h, y, z) - f(x-h, y, z))/h/2. # [dfx_dx, dfy_dx, dfz_dx]
    dfdy = (f(x, y+h, z) - f(x, y-h, z))/h/2. # [dfx_dy, dfy_dy, dfz_dy]
    dfdz = (f(x, y, z+h) - f(x, y, z-h))/h/2. # [dfx_dz, dfy_dz, dfz_dz]
    return np.array([dfdx, dfdy, dfdz]).T


def euler2quat(ang1, ang2, ang3):
    q0 = cos((ang1 + ang3)/2)*cos(ang2/2)
    q1 = cos((ang1 - ang3)/2)*sin(ang2/2)
    q2 = sin((ang1 - ang3)/2)*sin(ang2/2)
    q3 = sin((ang1 + ang3)/2)*cos(ang2/2)
    return np.r_[q0, q1, q2, q3]


## Equations for translation and rotation

def _sum(funcs, u, param):
    return np.array([func(u, param) for func in funcs]).sum(axis=0)


def eqns(u, t, forces, torques, param):
    force = _sum(forces, u, param) if len(forces) else np.zeros(3)
    torque = _sum(torques, u, param) if len(torques) else np.zeros(3)

    I, I_inv = param.I, param.I_inv
    q0, q1, q2, q3 = ang_(u)
    mat = np.array([[-q1,-q2,-q3], [q0,-q3,q2], [q3,q0,-q1], [-q2,q1,q0]])

    d_time = 1.0
    d_pos = vel_(u)
    d_vel = force/param.mass
    d_ang = mat.dot(omg_(u))/2.
    d_omg = I_inv.dot(torque - np.cross(omg_(u), I.dot(omg_(u))))
    return np.r_[d_time, d_pos, d_vel, d_ang, d_omg]


def _onestep(f, u, dt, args=()):
    k1 = dt*f(u, 0., *args)
    k2 = dt*f(u + k1/2., dt/2., *args)
    k3 = dt*f(u + k2/2., dt/2., *args)
    k4 = dt*f(u + k3, dt, *args)
    return u + (k1 + 2*k2 + 2*k3 + k4)/6.


def trace(u0, forces, torques, param, conditions, nmax=100000):
    print("\n>>> u0 =", u0)
    print(">>> Wait ...")

    tstart = time.time()
    u0 = np.concatenate([np.array(u0_).flatten() for u0_ in u0])
    sol = [u0]

    for i in range(nmax):
        u = _onestep(eqns, u0, param.dt, args=(forces, torques, param))
        if all(cond(u) == True for cond in conditions):
            sol.append(u)
            u0 = u
        else: break

    template = ">>> Calculation time: {:.2f} [s] /w {:d} [steps]"
    print(template.format(time.time() - tstart, i+1))
    return np.array(sol)


## Gravition
def force_grav(u, param):
    return np.r_[0., -(param.rho_p - param.rho_f)*param.vol*g, 0.]


## Drag force and torque

def force_drag(u, param):
    R = rotation(ang_(u))
    K = (R.T).dot(param.Kdrag).dot(R)
    v = param.vf(*pos_(u)) - vel_(u)
    return K.dot(v)


def torque_drag(u, param):
    R = rotation(ang_(u))
    L = R.dot(jacobian(param.vf, pos_(u))).dot(R.T)
    D = (L + L.T)/2   # rate of deformation
    W = (L - L.T)/2   # rate of rotation

    r, w = param.r, omg_(u)
    K, t = param.Ktorq, np.zeros_like(w)
    t[0] = W[2,1] - w[0]
    t[1] = (1 - r**2)*D[0,2]/2. + (1 + r**2)*(W[0,2] - w[1])/2.
    t[2] = (r**2 - 1)*D[1,0]/2. + (r**2 + 1)*(W[1,0] - w[2])/2.
    return K.dot(t)


## Dielectrophoresis force and torque

def force_dep(u, param):
    R = rotation(ang_(u))
    K = (R.T).dot(param.Kpol).dot(R)
    p = K.dot(param.elec(*pos_(u)))
    J = jacobian(param.elec, pos_(u))
    return J.dot(p)


def torque_dep(u, param):
    R = rotation(ang_(u))
    E = R.dot(param.elec(*pos_(u)))
    p = param.Kpol.dot(E)
    return np.cross(p, E)


## Post-pocessing

def plot_traj(ax, sol, length, nskip=100):
    sol = sol[::nskip]
    x, y, z = pos_(sol).T
    b = length/2.
    pt0 = pos_(sol)
    pt1 = np.array([pos_(s) + rotation(ang_(s)).T.dot([ b, 0, 0]) for s in sol])
    pt2 = np.array([pos_(s) + rotation(ang_(s)).T.dot([-b, 0, 0]) for s in sol])

    for (x0, y0, z0), (x1, y1, z1), (x2, y2, z2) in zip(pt0, pt1, pt2):
        ax.plot([x0, x1], [y0, y1], 'r-', lw=2)
        ax.plot([x0, x2], [y0, y2], 'g-', lw=2)

    ax.plot(x, y, 'k:', lw=2)
    return ax


if __name__ == "__main__":

    pass