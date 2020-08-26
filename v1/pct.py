import numpy as np
import time
from argparse import Namespace


## Material properties and simulation conditions

params = {'a': 1.0,         # semi-minor axis of particle
          'b': 1.0,         # semi-minor axis of particle
          'rho_p': 1.0,     # density of particle
          'rho_f': 1.0,     # density of fluid
          'dt': 1.0,        # calculation time step
          'eps_p': 1.0,     # permitivity of particle
          'eps_f': 1.0,     # permitivity of fluid
          'sig_p': 1.0,     # conductivty of particle
          'sig_f': 1.0,     # conductivity of fluid
          'freq': 1e6,      # AC frequency of electric field
}
params = Namespace(**params)

def update(params, show=True):
    params.r  = params.b/params.a
    params.r2 = (params.b/params.a)**2
    params.vol  = (4/3)*np.pi*params.a**2*params.b
    params.mass = params.vol*params.rho_p
    params.inertia_long  = 0.4*params.mass*params.a**2
    params.inertia_short = 0.2*params.mass*(params.a**2 + params.b**2)

    depol_long  = 1/(1 + 1.6*params.r + 0.4*params.r**2)
    depol_short = (1 - depol_long)/2.
    eps_p = params.eps_p - 1J*params.sig_p/np.pi/params.freq/2.
    eps_f = params.eps_f - 1J*params.sig_f/np.pi/params.freq/2.
    cmf_long  = (eps_p-eps_f)/((eps_p-eps_f)*depol_long  + eps_f)
    cmf_short = (eps_p-eps_f)/((eps_p-eps_f)*depol_short + eps_f)

    params.cmf_long  = cmf_long.real
    params.cmf_short = cmf_short.real

    if show:
        print('\n' + '='*60)
        print("# Simulation parameters:\n" + '='*60)
        for key, value in sorted(params.__dict__.items()):
            print("{:13s} = {}".format(key, value))
        print('='*60)


## Functions

def rot2d(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta),  np.cos(theta)]])

def rot3d(e1, e2, e3, e0):
    return np.array([[1-2*(e2**2+e3**2), 2*(e1*e2+e3*e0), 2*(e1*e3-e2*e0)],
                     [2*(e2*e1-e3*e0), 1-2*(e3**2+e1**2), 2*(e2*e3+e1*e0)],
                     [2*(e3*e1+e2*e0), 2*(e3*e2-e1*e0), 1-2*(e1**2+e2**2)]]).T

def euler2quat(psi, theta, phi):
    """ Euler angles in Z-X-Z rotation """
    e1 = np.cos((phi - psi)/2)*np.sin(theta/2)
    e2 = np.sin((phi - psi)/2)*np.sin(theta/2)
    e3 = np.sin((phi + psi)/2)*np.cos(theta/2)
    e0 = np.cos((phi + psi)/2)*np.cos(theta/2)
    return np.r_[e1, e2, e3, e0]

def jac2d(f, x, y, h=1e-7):
    dfdx = (f(x+h, y) - f(x-h, y))/h/2.
    dfdy = (f(x, y+h) - f(x, y-h))/h/2.
    return np.array([dfdx, dfdy]).T

def jac3d(f, x, y, z, h=1e-7):
    dfdx = (f(x+h, y, z) - f(x-h, y, z))/h/2.
    dfdy = (f(x, y+h, z) - f(x, y-h, z))/h/2.
    dfdz = (f(x, y, z+h) - f(x, y, z-h))/h/2.
    return np.array([dfdx, dfdy, dfdz]).T


## Particle tracing solver [2D/3D]

def _onestep(f, u, dt, args=()):
    k1 = dt*f(u, 0., *args)
    k2 = dt*f(u + k1/2., dt/2., *args)
    k3 = dt*f(u + k2/2., dt/2., *args)
    k4 = dt*f(u + k3, dt, *args)
    return u + (k1 + 2*k2 + 2*k3 + k4)/6.

def trace(eqns, u0, forces, torques, params, conditions, nmax=100000):
    print("\n>>> u0 =", u0)
    print(">>> Wait ...")

    u0 = np.concatenate([np.array(u0_).flatten() for u0_ in u0])
    sol = [u0]
    tstart = time.time()

    for i in range(nmax):
        u = _onestep(eqns, u0, params.dt, args=(forces, torques, params))
        if all(check(u) == True for check in conditions):
            sol.append(u)
            u0 = u
        else: break

    template = ">>> Calculation time: {:.2f} [s] /w {:d} [steps]"
    print(template.format(time.time() - tstart, i+1))

    return np.array(sol)


## Reference electric field

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

grid_E2 = grid_Ex**2 + grid_Ey**2
grid_dyE2, grid_dxE2 = np.gradient(grid_E2, y, x)

func_dxE2 = interp2d(x, y, grid_dxE2)
func_dyE2 = interp2d(x, y, grid_dyE2)


if __name__ == "__main__":

    update(params)