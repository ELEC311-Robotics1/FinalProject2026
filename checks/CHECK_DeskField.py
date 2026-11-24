#!/usr/bin/python3
# ================================================================
# CHECK_DeskField.py
# No motors, no lidar: does the field ask for the right motion?
#     python3 checks/CHECK_DeskField.py k PWM0      (your calibration)
# Each line is a situation, what the rover should do, and what it asks.
# ================================================================

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from lynx_field import Field, wheels_pwm, cap

try:
    k_mot, pwm0 = float(sys.argv[1]), float(sys.argv[2])
except (ValueError, IndexError):
    k_mot, pwm0 = 0.12, 8.0
    print(f"  (no calibration given: k = {k_mot}, PWM0 = {pwm0})")

CONTOUR = (0.6, 0.6, 2.5, 0.0)       # as in LYNX_FINAL_Field.py
INF = math.inf
cases = [
    # x, y, th, d, psi, what it should do, test on (v, w)
    (0.0, 0.0, 0.0, INF, 0.0, "far from the circle, facing it: forward",
     lambda v, w: v > 0 and abs(w) < 0.5),
    (0.0, 0.0, math.pi / 2, INF, 0.0, "far, facing left of it: forward and turn right",
     lambda v, w: w < 0),
    (2.4, -0.6, 0.0, INF, 0.0, "heading point on the circle (bottom), heading +x: forward, counterclockwise",
     lambda v, w: v > 0 and w > -0.05),
    (0.0, 0.0, 0.0, 0.20, 0.0, "wall 0.20 m straight ahead (inside the standoff): back off or turn",
     lambda v, w: v < 0.05 or abs(w) > 0.3),
    (0.0, 0.0, 0.0, 0.25, math.pi / 2, "wall 0.25 m on the left: turn right or push away",
     lambda v, w: w <= 0.05),
]
ok = True
for x, y, th, d, psi, name, good in cases:
    f = Field(CONTOUR)
    v, w, info = f.command(x, y, th, d, psi)
    uL, uR = cap(*wheels_pwm(v, w, k_mot, pwm0), 70)
    mark = "ok " if good(v, w) else "LOOK"
    ok = ok and mark == "ok "
    print(f"  {mark} {name}\n       v {v:+.3f} m/s  w {w:+.3f} rad/s  ->  uL {uL:+4d} uR {uR:+4d}"
          f"   (mu_obs {info['mu_obs']:.2f})")
print("\n  All as expected." if ok else "\n  Check the LOOK lines before driving.")
