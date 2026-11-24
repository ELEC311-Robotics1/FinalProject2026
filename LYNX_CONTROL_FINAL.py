#!/usr/bin/python3
# ================================================================
# LYNX_CONTROL_FINAL.py
# ELEC 311 final project: the rover follows the lidar-informed fuzzy
# velocity field, and the wheels follow it with the model-free law of the
# paper, PWM = -K1 tanh(NU1 S), S = e + BETA int(e), e = wheel speed minus
# wheel reference. No motor calibration: the integral builds the PWM.
# The walls act ONLY through the lidar. ON THE FLOOR, clear space, start slow.
#
# On the rover's Pi, from the LYNX_Final folder:
#     python3 LYNX_CONTROL_FINAL.py LIDAR_PORT
#         LIDAR_PORT  the A1's port, e.g. /dev/ttyUSB1
# Autonomous (no keys, no laptop): add --auto SECONDS. It zeros the pose
# where the rover sits, waits WAIT s for you to step away, drives SECONDS,
# stops, saves the log.
#     python3 LYNX_CONTROL_FINAL.py LIDAR_PORT --auto 60
#     pkill -f LYNX_CONTROL     (emergency stop from another terminal)
# Target: add --target. Before driving, the rover's camera finds the RED
# object (bearing, as in L16), the lidar gives its distance along that
# bearing (else its size), and the contour becomes a circle of TARGET_R
# around it. While driving, the camera keeps looking (as the W10 chase):
# each sighting moves the centre toward what the camera sees, which
# corrects the heading drift of the odometry.
#     nohup python3 LYNX_CONTROL_FINAL.py LIDAR_PORT --auto 150 --target
# Stop in front instead of circling: add --stop (with --target). It drives to
# STOP_R from the target, avoiding what is on the way, and stops there (or
# when the lidar sees something closer than STOP_D straight ahead, near it).
# On a laptop, no rover (same code, simulated rover, lidar and maze):
#     python3 LYNX_CONTROL_FINAL.py --fake
#
# Keys, in this terminal (rover only):
#     g       go: follow the field
#     z       pose = (0, 0, 0) here (the contour is defined in this frame)
#     space   stop                       q   quit
# The motors stop on q, space, Ctrl+C, a closed SSH session, a lost USB
# link, a lidar fault, and when no new lidar turn arrives for STALE s.
# Log: control_log_<time>.csv in the folder you ran it from.
# ================================================================

# ── Task: the contour, in the frame where you pressed z (m) ─────
CONTOUR  = (0.6, 0.6, 2.5, 0.0)   # circle of 0.6 m, 2.5 m straight ahead
SPEED    = 0.15                   # m/s along the field (0.3 m/s in the paper)
ELL      = 0.10                   # m: the lidar (and heading point) ahead of the axle
DS       = 0.35                   # m: standoff from the walls

# ── Lidar mounting (measure with checks/CHECK_Lidar.py) ─────────
MOUNT_DEG = 0.0                   # where the A1's 0 deg points (motor end forward: 0)
MASK = [(150.0, 210.0, 0.25)]     # (psi_lo, psi_hi deg, r_max m): the chassis behind
STALE = 1.0                       # s without a new lidar turn -> motors stop

# ── Loop and motors ─────────────────────────────────────────────
LOOP_HZ = 20                      # Mega: 20 Hz (as W10). Teensy: try 50
WAIT = 5.0                        # s, --auto: time to step away before it drives

# ── The law: PWM = -K1 tanh(NU1 S), S = e + BETA int(e), per side ──
TS    = 1.0 / LOOP_HZ             # s, sampling time
K1    = 80                        # PWM: the bound, K1 > xi_bar (cruise ~26, pivot ~46)
                                  # (Sabertooth max 127)
NU1   = 1.0 / (4.0 * TS)          # s/rad: 5 at 20 Hz
BETA  = 2.0                       # 1/s, keep below 5
TAU_W = 0.05                      # s: filter of the encoder speeds

# ── Target (--target): the red object, seen by the rover's camera ─
CAM      = 0                      # the rover's webcam
RED_LO   = [(0, 120, 70), (170, 120, 70)]    # red wraps around H = 0
RED_HI   = [(10, 255, 255), (179, 255, 255)]
HFOV_DEG = 31.0                   # webcam horizontal field of view (deg)
OBJ_W    = 0.20                   # m, width of the red book (camera fallback)
TARGET_R = 0.6                    # m, radius of the circle around the target
Z_MAX    = 6.0                    # m, the camera's distance is capped at the lidar range
NEAR_FREEZE = 1.5                 # m: no updates closer than this to the centre
STOP_R   = 0.5                    # m, --stop: stop when the heading point is this far from the target
STOP_D   = 0.40                   # m, --stop: or when the lidar sees something this close ahead,
                                  #    within 2 m of the target (the book itself, or what hides it)
GATE     = 1.0                    # m: sightings farther than this from the first
                                  #    estimate are other red things: ignored

import math
import os
import signal

import numpy as np
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))
import lynx_io as io
from lynx_field import Field, R_EFF, L_EFF
from lynx_lidar import nearest
from lynx_odometry import Odometry

FAKE = "--fake" in sys.argv


# ── Desk check: the signs of the law, before any motor turns ────
def desk_check():
    """Wheels at rest, field asks (v, w): the first PWM must push the
    right way (S = -reference, so PWM = K1 tanh(NU1 reference))."""
    tests = [((0.1, 0.0), lambda L, R: L > 0 and R > 0 and L == R, "straight"),
             ((0.0, 0.5), lambda L, R: R > 0 > L, "turn left"),
             ((0.0, -0.5), lambda L, R: L > 0 > R, "turn right")]
    ok = True
    for (v, w), good, name in tests:
        qr = ((v - w * L_EFF / 2) / R_EFF, (v + w * L_EFF / 2) / R_EFF)
        uL, uR = (int(round(K1 * math.tanh(NU1 * q))) for q in qr)
        mark = "ok " if good(uL, uR) else "WRONG"
        ok = ok and mark == "ok "
        print(f"  {mark} {name:10s} v={v}, w={w}:  uL={uL:+4d}  uR={uR:+4d}")
    return ok


# ── One control step: the only code that talks to the wheels ────
LAST = [0, 0]                     # PWM sent at the previous step
SM = dict(prev=None, t=None, qd=[0.0, 0.0], z=[0.0, 0.0], S=[0.0, 0.0])


def smc_pwm(a, t, v, w):
    """Wheel angles a (rad) at time t, field command (v, w) ->
    PWM (uL, uR) = -K1 tanh(NU1 S), with S = e + BETA int(e) per side."""
    dt = None
    if SM["prev"] is not None:
        dt = max(t - SM["t"], 1e-3)
        for i in range(2):                    # encoder speeds, filtered
            raw = (a[i] - SM["prev"][i]) / dt
            SM["qd"][i] += dt / (TAU_W + dt) * (raw - SM["qd"][i])
    SM["prev"], SM["t"] = a, t
    if (v == 0.0 and w == 0.0) or dt is None:  # stopped: no wind-up
        SM["z"], SM["S"] = [0.0, 0.0], [0.0, 0.0]
        return 0, 0
    qr = ((v - w * L_EFF / 2) / R_EFF, (v + w * L_EFF / 2) / R_EFF)   # rad/s
    u = []
    for i in range(2):
        e = SM["qd"][i] - qr[i]
        SM["z"][i] += e * dt
        SM["S"][i] = e + BETA * SM["z"][i]
        u.append(int(round(-K1 * math.tanh(NU1 * SM["S"][i]))))
    return u[0], u[1]


def control_step(ser, lidar, odo, field, go, logf, t):
    a = io.read_angles(ser)
    if a is None:
        raise IOError("no encoder reply")
    x, y, th = odo.update(*a)
    psi, d, t_scan = lidar.latest()
    dmin, pmin = nearest(psi, d)
    stale = (time.monotonic() - t_scan) > STALE
    v = w = 0.0
    info = dict(g=float("nan"), mu_obs=0.0)
    if go and not stale and not lidar.fault:
        v, w, info = field.command(x, y, th, dmin, pmin)
    uL, uR = smc_pwm(a, t, v, w)              # bounded by K1 by construction
    if v == 0.0 and w == 0.0:                 # stop: at once, no ramp
        uL = uR = 0
    LAST[:] = [uL, uR]
    io.send_motor(ser, 1, uL)
    io.send_motor(ser, 2, uR)
    logf.write(f"{t:.3f},{int(go)},{uL},{uR},{a[0]:.5f},{a[1]:.5f},"
               f"{x:.4f},{y:.4f},{th:.4f},{dmin:.3f},{pmin:.3f},"
               f"{info['g']:.5f},{info['mu_obs']:.3f},{SM['S'][0]:.4f},{SM['S'][1]:.4f}\n")
    return dict(x=x, y=y, th=th, d=dmin, psi=pmin, uL=uL, uR=uR, stale=stale, **info)


def open_log():
    logf = open(time.strftime("control_log_%Y%m%d_%H%M%S.csv"), "w", buffering=1)
    logf.write("t_s,go,uL_pwm,uR_pwm,phiL_rad,phiR_rad,x_m,y_m,th_rad,"
               "d_m,psi_rad,g,mu_obs,SL,SR\n")
    return logf


# ── Laptop: the same steps on the simulated rover, lidar and maze ─
def run_fake(T=60.0, start=(2.7, 0.8, math.pi / 2), contour=(0.6, 0.6, 3.0, 5.4),
             speed=0.3, k_mot=0.1822, pwm0=12.5, verbose=True):   # sim motor only
    import lynx_sim as sim
    rover = sim.SimRover(*start, k_mot=k_mot, pwm0=pwm0)
    ser, lidar = sim.FakeSerial(rover), sim.FakeLidar(rover, ell=ELL)
    odo = Odometry(*start)                      # the maze frame
    field = Field(contour, ell=ELL, speed=speed, Ds=DS)
    logf = open_log()
    path, dmins, gs = [], [], []
    dt, sub = 1.0 / LOOP_HZ, 10
    for n in range(int(T * LOOP_HZ)):
        s = control_step(ser, lidar, odo, field, True, logf, n * dt)
        for _ in range(sub):                    # the world between two steps
            rover.step(dt / sub)
        path.append((rover.x, rover.y, rover.th))
        dmins.append(s["d"]); gs.append(s["g"])
    logf.close()
    if verbose:
        reached = [i for i, g in enumerate(gs) if abs(g) < 0.05]
        print(f"  closest wall (lidar) {min(dmins):.2f} m;  "
              + (f"on the contour at t = {reached[0] / LOOP_HZ:.1f} s"
                 if reached else "did not reach the contour"))
        print(f"  Log: {logf.name}")
    return path, dmins, gs, contour


# ── Target: camera bearing + lidar range -> centre of the contour ─
def find_red(img, save=None):
    """Largest red blob: (cx, width px), or None. save: write a picture
    with the blob boxed and the centre line, to check what it found."""
    import cv2
    hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, RED_LO[0], RED_HI[0]) | cv2.inRange(hsv, RED_LO[1], RED_HI[1])
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) > 20]
    if save:
        out = img.copy()
        cv2.line(out, (img.shape[1] // 2, 0), (img.shape[1] // 2, img.shape[0]), (255, 255, 0), 1)
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 1)
        if cs:
            x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 255), 3)
        cv2.imwrite(save, out)
    if not cs:
        return None
    x, y, w, h = cv2.boundingRect(max(cs, key=cv2.contourArea))
    H, W = img.shape[:2]
    edge = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2   # cut by the frame
    return x + w / 2.0, float(w), float(edge)


def camera_look(seconds, save=None):
    """Rover's camera for `seconds`: (bearing rad, distance m from size), or None."""
    import cv2
    cam = cv2.VideoCapture(CAM)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    seen, t0, width, n = [], time.monotonic(), 640, 0
    while (time.monotonic() - t0 < seconds
           or (n < 8 and time.monotonic() - t0 < 3 * seconds)):   # at least 8 frames
        ok, img = cam.read()
        n += 1
        if ok:
            width, last = img.shape[1], img
            b = find_red(img)
            if b:
                seen.append(b)
    cam.release()
    if save and "last" in locals():
        find_red(last, save=save)                 # the last frame: exposure settled
    if len(seen) < 3:
        return None
    cx, wpx = np.median(np.array(seen)[:, :2], axis=0)
    f = (width / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))   # px
    beta = math.atan2(width / 2.0 - cx, f)        # rad, positive to the LEFT
    return beta, f * OBJ_W / max(wpx, 1.0)


def locate_target(lidar, seconds=3.0):
    """Rover at rest. Returns (x, y) of the target in the rover frame, or None."""
    look = camera_look(seconds, save="target_view.jpg")
    if look is None:
        print("  TARGET: no red object seen by the camera.", flush=True)
        return None
    beta, z_cam = look
    z_cam = min(z_cam, Z_MAX)                     # a sliver of red looks very far
    psi, d, _ = lidar.latest()
    near = np.abs((psi - beta + math.pi) % (2 * math.pi) - math.pi) < math.radians(3)
    near &= d < 1.4 * z_cam                       # along the bearing, nearer than the camera
                                                  # says: the target, or what hides it
    if near.sum():
        z, how = float(np.min(d[near])), "lidar"
    else:
        z, how = z_cam, "camera size"
    xt = ELL + z * math.cos(beta)                 # the camera/lidar is ELL ahead of the axle
    yt = z * math.sin(beta)
    print(f"  TARGET: bearing {math.degrees(beta):+.1f} deg, distance {z:.2f} m "
          f"({how}; camera says {z_cam:.2f} m) -> centre ({xt:.2f}, {yt:.2f}) m"
          "  [what it saw: target_view.jpg]", flush=True)
    return xt, yt


class Tracker:
    """As the W10 chase: the camera keeps running in a thread; whenever the
    red target is in view, its centre is measured again from the pixels and
    the circle moves toward it (a low-pass, GAIN per sighting). This corrects
    the heading drift of the odometry while the target is visible."""
    GAIN, FPS = 0.3, 5.0

    def __init__(self, lidar=None):
        import threading
        self.lidar = lidar
        self.lock = threading.Lock()
        self.look, self.n, self.used, self.first = None, 0, 0, None
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import cv2
        cam = cv2.VideoCapture(CAM)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while self.running:
            ok, img = cam.read()
            if ok:
                b = find_red(img)
                if b and not b[2]:                       # whole target in view
                    cx, wpx = b[0], b[1]
                    width = img.shape[1]
                    f = (width / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
                    look = (math.atan2(width / 2.0 - cx, f), f * OBJ_W / max(wpx, 1.0))
                    with self.lock:
                        self.look, self.n = look, self.n + 1
            time.sleep(1.0 / self.FPS)
        cam.release()

    def stop(self):
        self.running = False

    def step(self, field, x, y, th):
        with self.lock:
            look, n = self.look, self.n
        if look is None or n == self.used:
            return
        self.used = n
        beta, z = look
        px, py = x + ELL * math.cos(th), y + ELL * math.sin(th)
        ox, oy = field.params[2], field.params[3]
        if math.hypot(ox - px, oy - py) < NEAR_FREEZE:
            return                                       # up close: size fails, keep the centre
        z = min(z, Z_MAX)
        if self.lidar is not None:                       # lidar range along the bearing
            psi, d, _ = self.lidar.latest()
            near = np.abs((psi - beta + math.pi) % (2 * math.pi) - math.pi) < math.radians(3)
            near &= (d > 0.7 * z) & (d < 1.4 * z)
            if near.sum():
                z = float(np.min(d[near]))
        nx, ny = px + z * math.cos(th + beta), py + z * math.sin(th + beta)
        if self.first is None:
            self.first = (ox, oy)                        # the estimate at the start
        if math.hypot(nx - self.first[0], ny - self.first[1]) > GATE:
            return                                       # another red thing: ignore
        cx, cy = ox + self.GAIN * (nx - ox), oy + self.GAIN * (ny - oy)
        field.params = (field.params[0], field.params[1], cx, cy)
        self.last = (cx, cy)


# ── Rover: keys in the terminal, motors always stopped on the way out ─
def run_rover():
    import select
    import termios
    import tty
    from lynx_lidar import Lidar
    try:
        port = sys.argv[1]
        assert not port.startswith("-")
        auto = float(sys.argv[sys.argv.index("--auto") + 1]) if "--auto" in sys.argv else None
    except (ValueError, IndexError, AssertionError):
        print("  usage: LYNX_CONTROL_FINAL.py LIDAR_PORT [--auto SECONDS] [--target [--stop]]   (or --fake)")
        sys.exit(1)
    print(f"  contour {CONTOUR}, speed {SPEED} m/s, {LOOP_HZ} Hz; "
          f"PWM = -{K1} tanh({NU1:g} S), S = e + {BETA:g} int(e)")
    if not desk_check():
        print("  the signs of the law are not right: fix them, then run again.")
        sys.exit(1)

    def shutdown(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGHUP, shutdown)       # SSH session closed (keys mode)

    mega = None
    if auto is not None:                          # no one to answer "Select option"
        import serial.tools.list_ports
        acm = [p.device for p in serial.tools.list_ports.comports()
               if p.device != port and ("ACM" in p.device or "USB" in p.device)]
        acm.sort(key=lambda d: "ACM" not in d)       # the Mega is usually ttyACM
        mega = acm[0] if acm else None
        signal.signal(signal.SIGHUP, signal.SIG_IGN)   # survive a closed SSH
    ser = io.open_rover(mega)
    lidar = Lidar(port, MOUNT_DEG, MASK)
    odo, field, logf = Odometry(), Field(CONTOUR, ell=ELL, speed=SPEED, Ds=DS), open_log()
    go, reason, t0 = False, "Done.", time.monotonic()
    refine = None
    keys = auto is None and sys.stdin.isatty()
    if keys:
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
    try:
        lidar.start()
        if "--target" in sys.argv:
            time.sleep(1.0)                       # a few lidar turns first
            tgt = locate_target(lidar)
            if tgt is None:
                raise KeyboardInterrupt           # no target: do not drive
            r0 = STOP_R if "--stop" in sys.argv else TARGET_R
            field.params = (r0, r0, tgt[0], tgt[1])
            t0 = time.monotonic()                 # the wait starts now
        refine = Tracker(lidar) if "--target" in sys.argv else None
        if keys:
            print("  g go  z zero  space stop  q quit\n")
            tty.setcbreak(fd)                     # one key at a time
        else:
            print(f"  AUTO: drives in {WAIT:.0f} s for {auto:.0f} s. Step away.", flush=True)
        t_go = t0 + WAIT
        while True:
            t1 = time.monotonic()
            if keys:
                r, _, _ = select.select([fd], [], [], 0)
                if r:
                    for ch in os.read(fd, 64).decode(errors="ignore").lower():
                        if ch == "q":
                            raise KeyboardInterrupt
                        if ch == " ":
                            go = False
                        if ch == "g":
                            go = True
                        if ch == "z":
                            odo.zero()
            else:
                if not go and t1 >= t_go:
                    odo.zero()                    # the frame of the contour: here
                    go = True
                    print("  GO", flush=True)
                if t1 >= t_go + auto:
                    break
            s = control_step(ser, lidar, odo, field, go, logf, t1 - t0)
            if refine and go:
                refine.step(field, s["x"], s["y"], s["th"])
                if "--stop" in sys.argv:          # arrive and stop in front of the target
                    cx, cy = field.params[2], field.params[3]
                    px = s["x"] + ELL * math.cos(s["th"])
                    py = s["y"] + ELL * math.sin(s["th"])
                    dc = math.hypot(cx - px, cy - py)
                    ahead = math.isfinite(s["d"]) and abs(s["psi"]) < math.radians(25)
                    if dc < STOP_R + 0.03 or (dc < 2.0 and ahead and s["d"] < STOP_D):
                        io.stop_all(ser)
                        print(f"  ARRIVED: {dc:.2f} m from the target centre, "
                              f"nearest ahead {s['d']:.2f} m", flush=True)
                        break
                if int(t1 * LOOP_HZ) % (5 * LOOP_HZ) == 0:    # every 5 s
                    print(f"  TARGET now ({field.params[2]:.2f}, {field.params[3]:.2f}) m", flush=True)
            flag = lidar.fault or ("LIDAR STALE" if s["stale"] else "")
            line = (f"  {'GO  ' if go else 'stop'}  L {s['uL']:+4d} R {s['uR']:+4d}  "
                    f"x {s['x']:+5.2f} y {s['y']:+5.2f} th {math.degrees(s['th']):+5.0f}  "
                    f"wall {s['d']:4.2f} m  g {s['g']:+7.3f}  {flag:14s}")
            if keys:
                print("\r" + line, end="", flush=True)
            elif int(t1 * LOOP_HZ) % LOOP_HZ == 0:  # about once a second
                print(line, flush=True)
            time.sleep(max(0.0, 1.0 / LOOP_HZ - (time.monotonic() - t1)))
    except KeyboardInterrupt:
        pass
    except Exception as e:                        # USB unplugged, Mega reset...
        reason = f"LINK LOST: {e.__class__.__name__}: {e}"
    finally:
        io.close_rover(ser)                       # motors first, always
        if refine:
            refine.stop()
        lidar.stop()
        logf.close()
        if keys:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except Exception:
                pass
        print(f"\n\n  {reason} Motors stopped. Log: {logf.name}")
        print("=^..^=")


if __name__ == "__main__":
    if FAKE:
        run_fake()
    else:
        run_rover()
