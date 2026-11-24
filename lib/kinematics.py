# kinematics.py
# kinematic functions for rover motion
import numpy as np # it is needed here as well, because the functions use numpy functions

# Foward kinematics: convert wheel speeds to linear and angular velocities
def u1u2_to_vw(u1, u2, R=0.15, L=0.33):
    v = R * (u2 + u1) / 2
    w = R * (u2 - u1) / L
    return v, w

# Inverse kinematics: convert linear and angular velocities to wheel speeds
def vw_to_u1u2(v, w, R=0.15, L=0.33):
    u1 = (v - (L / 2) * w) / R
    u2 = (v + (L / 2) * w) / R
    return u1, u2

def vw_to_dxdydtht(v, w, tht):
    dx = v * np.cos(tht)
    dy = v * np.sin(tht)
    dtht = w
    return dx, dy, dtht

def xcyc_to_xryr(xc, yc, tht, ell = 0.1):
    xr = xc + ell * np.cos(tht)
    yr = yc + ell * np.sin(tht)
    return xr, yr

def VF_to_vw(V, tht, ell=0.1):
    A = np.array([
        [np.cos(tht), -ell * np.sin(tht)],
        [np.sin(tht), ell * np.cos(tht)]
    ])
    aux = np.linalg.solve(A, V)
    v = aux[0]
    w = aux[1]
    return v, w
