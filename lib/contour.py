# contour.py
# Contour velocity field and lidar obstacle field for a differential-drive rover
import numpy as np

# Contour error function (ellipse): g < 0 inside, g = 0 on the contour, g > 0 outside
def g(params, x, y):
    a, b, x0, y0 = params
    return ((x - x0) / a) ** 2 + ((y - y0) / b) ** 2 - 1

# Numeric unit gradient of the contour error function
def unit_grad(params, x, y):
    dx = 1e-4
    dg_dx = (g(params, x + dx, y) - g(params, x, y)) / dx
    dg_dy = (g(params, x, y + dx) - g(params, x, y)) / dx
    Grad = np.array([dg_dx, dg_dy])
    norm = np.linalg.norm(Grad)
    if norm < 1e-9:  # at the center the gradient is zero
        return np.array([1.0, 0.0])
    u = Grad / norm
    return u

# Membership functions for the contour error
def mu(gval, nu):
    mun = np.tanh(nu * gval ** 2)  # normal: large far from the contour (g^2: smooth)
    mut = 1 - mun                  # tangent: large near the contour
    return mut, mun

# Velocity field
def VF(params, x, y, nu, direction=1):  # direction: +1 counterclockwise, -1 clockwise
    ug = unit_grad(params, x, y)
    gval = g(params, x, y)
    # Normal Field (points toward the contour)
    N = - np.sign(gval) * ug
    # Tangential Field (along the contour)
    T = direction * np.array([-ug[1], ug[0]])
    # Membership functions
    mut, mun = mu(gval, nu)
    # Velocity Field
    V = mun * N + mut * T
    return V

# Obstacle field from the nearest lidar return: distance d, unit vector u (rover -> object)
def Vobs(d, u, Ds, nu, sigma):  # Ds: standoff distance, sigma: +1 or -1 (side)
    grad_d = -u                  # away from the object
    g_o = d - Ds                 # g_o < 0 inside the standoff
    # Normal Field (pushes out, only inside the standoff)
    N = np.tanh(nu * max(0, -g_o) ** 2) * grad_d
    # Tangential Field (goes around the object)
    T = sigma * np.array([-grad_d[1], grad_d[0]])
    # Membership functions
    mut, _ = mu(g_o, nu)
    # Velocity Field
    V = mut * T + N
    return V, g_o

# Side to go around: the one that agrees with the contour field
def side(V, u, mu_obs, sigma):
    grad_d = -u
    T = np.array([-grad_d[1], grad_d[0]])
    if mu_obs < 0.1 or np.dot(V, grad_d) > 0:  # far, or V already points away
        if np.dot(sigma * T, V) < -0.1:        # switch only if the side clearly disagrees
            sigma = -sigma
    return sigma                               # otherwise keep the same side

# Merging with obstacle field (u: rover -> object, for the release)
def Merge(g_o, nu_b, V, Vo, u=None):
    mu_obs = 1 - np.tanh(nu_b * max(0, g_o) ** 2)  # 1 inside the standoff, -> 0 far away
    if u is not None:                    # release: V already points away from the object
        away = max(0, np.dot(V, -u))
        mu_obs = mu_obs * (1 - np.tanh(5 * away) ** 2)
    Vmerged = (1 - mu_obs) * V + mu_obs * Vo
    return Vmerged, mu_obs
