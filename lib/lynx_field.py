#!/usr/bin/python3
# ================================================================
# lynx_field.py
# The field of the paper on the rover: contour field + lidar obstacle field,
# merged with the release (contour.py, squared memberships as in the paper), then the heading point -> (v, w) (kinematics.py),
# then (v, w) -> PWM with the W10 calibration (wheels_pwm, as in W10).
# Units: m, rad, s.
# ================================================================

import math

import numpy as np

import contour as con
import kinematics as kin

R_EFF = 0.060325             # m, wheel radius
L_EFF = 0.287                # m, track width
W_EPS = 1.5                  # rad/s: wheel speed over which the deadband ramps in


class Field:
    def __init__(self, params, ell=0.10, speed=0.15, nu=30.0,
                 Ds=0.35, nu_o=30.0, nu_b=100.0, k_o=2.0, release=True):
        self.params = params            # contour (a, b, x0, y0), m
        self.ell = ell                  # heading point = where the lidar is
        self.speed = speed              # m/s along the field
        self.nu, self.Ds, self.nu_o, self.nu_b, self.k_o = nu, Ds, nu_o, nu_b, k_o
        self.sigma, self.mu_obs = 1, 0.0
        self.release = release          # let go of the wall when Vc already leaves it

    def command(self, x, y, th, d, psi):
        """Pose of the axle center (x, y, th), nearest return (d, psi in the
        rover frame) -> (v, w) and a dict for the log."""
        xr, yr = kin.xcyc_to_xryr(x, y, th, self.ell)
        Vc = con.VF(self.params, xr, yr, self.nu)
        if math.isfinite(d):
            u = np.array([math.cos(th + psi), math.sin(th + psi)])   # rover -> wall
            self.sigma = con.side(Vc, u, self.mu_obs, self.sigma)
            Vo, g_o = con.Vobs(d, u, self.Ds, self.nu_o, self.sigma)
            # release (u): let go of the wall when Vc already points away from it
            V, self.mu_obs = con.Merge(g_o, self.nu_b, Vc, self.k_o * Vo,
                                       u if self.release else None)
        else:                                   # nothing in range: contour only
            V, self.mu_obs = Vc, 0.0
        Vref = self.speed * V / max(np.linalg.norm(V), 0.25)
        v, w = kin.VF_to_vw(Vref, th, self.ell)
        g = con.g(self.params, xr, yr)
        return v, w, dict(xr=xr, yr=yr, g=g, mu_obs=self.mu_obs, sigma=self.sigma)


def wheels_pwm(v, w, k_mot, pwm0, pwm_turn=0.0):
    """PWM (uL, uR) for v (m/s) and w (rad/s): inverse kinematics, then the
    motor line of the W10 calibration, |wheel| = k (|PWM| - PWM0).
    Turning drags the four wheels sideways (scrub): the deadband grows by up
    to pwm_turn, all of it in a pivot (wheels opposite), none going straight."""
    wL = (v - w * L_EFF / 2) / R_EFF
    wR = (v + w * L_EFF / 2) / R_EFF
    turn = abs(wR - wL) / max(abs(wL) + abs(wR), 1e-9)    # 0 straight, 1 pivot
    p0 = pwm0 + pwm_turn * turn
    # smooth deadband: ramps in over ~W_EPS rad/s instead of jumping 0 -> p0
    # (a jump makes the wheels shiver when their speed crosses zero)
    uL = math.copysign(p0 * math.tanh(abs(wL) / W_EPS) + abs(wL) / k_mot, wL)
    uR = math.copysign(p0 * math.tanh(abs(wR) / W_EPS) + abs(wR) / k_mot, wR)
    return uL, uR


def cap(uL, uR, pwm_max):
    """Scale both PWMs so the larger is at most pwm_max: keeps the turn (W10)."""
    s = max(1.0, abs(uL) / pwm_max, abs(uR) / pwm_max)
    return round(uL / s), round(uR / s)
