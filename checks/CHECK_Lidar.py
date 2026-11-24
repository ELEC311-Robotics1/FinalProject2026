#!/usr/bin/python3
# ================================================================
# CHECK_Lidar.py
# Before driving: is the lidar's 0 deg straight ahead, and where does it
# see the rover itself? Rover on the floor, about 1 m of free space around
# it, ONE box straight ahead at about 0.5 m.
#     python3 checks/CHECK_Lidar.py /dev/ttyUSB1      (or --fake)
# It prints the nearest return (should be the box: psi near 0 deg), the
# ranges ahead/left/back/right, and every return closer than 0.45 m by
# angle: those that are not the box are the chassis or the wires -> put
# them in MASK of LYNX_FINAL_Field.py. Saves lidar_check.csv and a plot.
# ================================================================

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lib"))
from lynx_lidar import Lidar, nearest

MOUNT_DEG = 0.0              # as in LYNX_FINAL_Field.py
CLOSE = 0.45                 # m: list everything closer than this


def sector(psi, d, center_deg, half=5.0):
    dd = (np.degrees(psi) - center_deg + 180.0) % 360.0 - 180.0
    r = d[np.abs(dd) <= half]
    return float(np.median(r)) if r.size else float("nan")


if "--fake" in sys.argv:
    import lynx_sim as sim
    rover = sim.SimRover(1.5, 1.0, math.pi / 2)
    lidar = sim.FakeLidar(rover)
else:
    if len(sys.argv) < 2:
        sys.exit("  usage: CHECK_Lidar.py LIDAR_PORT   (or --fake)")
    lidar = Lidar(sys.argv[1], MOUNT_DEG, mask=())      # no mask: see everything

try:
    lidar.start()
    t0 = time.monotonic()
    psi, d, t = lidar.latest()
    while len(d) == 0 and time.monotonic() - t0 < 10:   # wait for a full turn
        time.sleep(0.2)
        psi, d, t = lidar.latest()
    time.sleep(0.5)
    psi, d, t = lidar.latest()
    if len(d) == 0:
        sys.exit(f"  No full turn in 10 s. {getattr(lidar, 'fault', '')}")
    dmin, pmin = nearest(psi, d)
    st = getattr(lidar, "stats", None)
    print(f"\n  {len(d)} returns in the last turn")
    if st:
        print(f"  A1 sent {st['raw']} measurements: {st['zero']} with no echo, "
              f"{st['lowq']} too weak, {lidar.bad} bytes out of step, "
              f"{lidar.turns / max(time.monotonic() - t0, 1e-3):.1f} turns/s")
    print(f"  nearest: {dmin:.3f} m at psi = {math.degrees(pmin):+.1f} deg "
          f"(the box ahead should be near 0)")
    print(f"  ahead {sector(psi, d, 0):.2f}  left {sector(psi, d, 90):.2f}  "
          f"back {sector(psi, d, 180):.2f}  right {sector(psi, d, -90):.2f} m")
    close = np.where(d < CLOSE)[0]
    print(f"\n  returns closer than {CLOSE} m (box, chassis, wires):")
    for i in close[np.argsort(psi[close])]:
        print(f"    psi {math.degrees(psi[i]):+7.1f} deg   d {d[i]:.3f} m")
    np.savetxt("lidar_check.csv", np.c_[np.degrees(psi), d], delimiter=",",
               header="psi_deg_ccw_rover,d_m", comments="")
    try:
        import matplotlib
        if "--fake" in sys.argv:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 5))
        plt.plot(d * np.cos(psi), d * np.sin(psi), '.', ms=3)
        plt.plot(0, 0, 'r>', ms=12)
        plt.gca().add_patch(plt.Circle((0, 0), CLOSE, fill=False, ls='--'))
        plt.xlabel('x (m), ahead'); plt.ylabel('y (m), left')
        plt.gca().set_aspect('equal'); plt.grid(alpha=0.3)
        plt.xlim(-1.5, 1.5); plt.ylim(-1.5, 1.5)
        plt.title('Rover frame: the box should be on +x')
        plt.savefig("lidar_check.png", dpi=120)
        if "--fake" not in sys.argv:
            plt.show()
    except Exception:
        pass
    print("\n  Saved lidar_check.csv and lidar_check.png")
finally:
    lidar.stop()
