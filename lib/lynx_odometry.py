#!/usr/bin/python3
# ================================================================
# lynx_odometry.py
# Dead reckoning from the wheel angles (Lecture 17, as in LYNX_HO_W10_Aut.py).
# Units: m and rad (W10 uses mm; only the units changed).
# ================================================================

import math

R_EFF = 0.060325             # m, wheel radius (Lynxmotion: 120.65 mm wheel)
L_EFF = 0.287                # m, track width (Lynxmotion)


class Odometry:
    def __init__(self, x=0.0, y=0.0, th=0.0):
        self.x, self.y, self.th = x, y, th
        self.prev = None                    # last wheel angles

    def zero(self, x=0.0, y=0.0, th=0.0):
        self.x, self.y, self.th = x, y, th

    def update(self, phiL, phiR):
        """New wheel angles (rad) -> pose (x, y, th) of the wheel axle center."""
        if self.prev is not None:
            dL, dR = phiL - self.prev[0], phiR - self.prev[1]
            ds = R_EFF * (dL + dR) / 2
            self.x += ds * math.cos(self.th)
            self.y += ds * math.sin(self.th)
            self.th += R_EFF * (dR - dL) / L_EFF
        self.prev = (phiL, phiR)
        return self.x, self.y, self.th
