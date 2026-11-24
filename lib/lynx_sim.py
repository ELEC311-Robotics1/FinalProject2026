#!/usr/bin/python3
# ================================================================
# lynx_sim.py
# A laptop stand-in for the rover, so the SAME programs run without it:
#   FakeSerial: speaks the protocol of LYNX_COMM_RPI5.ino (sel 1..7),
#               drives a simulated rover (motor line + lag + encoders)
#   FakeLidar:  same interface as lynx_lidar.Lidar (latest(), stop()),
#               ray-casts the maze from the lidar position
#   The maze:   6 x 6.6 m room, four regions, 1.2 m doors (Lecture 20)
# ================================================================

import math
import time

import numpy as np

R_EFF, L_EFF, N = 0.060325, 0.287, 12000

# ── The maze ─────────────────────────────────────────────────────
W, H = 6.0, 6.6
YA, YB, XM, DOOR = 1.6, 4.2, 3.0, 1.2


def walls():
    segs = [(0, 0, W, 0), (W, 0, W, H), (W, H, 0, H), (0, H, 0, 0), (XM, YA, XM, YB)]
    for y, doors in [(YA, [1.4, 4.6]), (YB, [2.2, 3.8])]:
        x = 0.0
        for c in doors:
            segs.append((x, y, c - DOOR / 2, y))
            x = c + DOOR / 2
        segs.append((x, y, W, y))
    return np.array(segs, float)


SEGS = walls()


def raycast(x, y, th, segs=SEGS, n=360, rmin=0.15, rmax=6.0):
    """One turn from (x, y), heading th: psi (rover frame, rad) and d (m)."""
    psi = np.radians(np.arange(n)) - math.pi
    phi = th + psi
    dx, dy = np.cos(phi)[:, None], np.sin(phi)[:, None]
    x1, y1, x2, y2 = segs.T
    ex, ey = x2 - x1, y2 - y1
    det = ex * dy - ey * dx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (ex * (y1 - y) - ey * (x1 - x)) / det
        s = (dx * (y1 - y) - dy * (x1 - x)) / det
    hit = (np.abs(det) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1)
    d = np.where(hit, t, np.inf).min(axis=1)
    keep = (d >= rmin) & (d <= rmax)
    return psi[keep], d[keep]


# ── The simulated rover ──────────────────────────────────────────
class SimRover:
    """Wheel speed follows the motor line with a lag; pose integrates it."""
    def __init__(self, x, y, th, k_mot=0.12, pwm0=8.0, lag=0.1):
        self.x, self.y, self.th = x, y, th
        self.k, self.pwm0, self.lag = k_mot, pwm0, lag
        self.pwm = [0, 0]                   # left, right (signed)
        self.w = [0.0, 0.0]                 # wheel speeds, rad/s
        self.phi = [0.0, 0.0]               # wheel angles, rad
        self.t = 0.0

    def step(self, dt):
        for i in range(2):
            p = self.pwm[i]
            target = math.copysign(self.k * (abs(p) - self.pwm0), p) if abs(p) > self.pwm0 else 0.0
            self.w[i] += dt / self.lag * (target - self.w[i])
            self.phi[i] += self.w[i] * dt
        v = R_EFF * (self.w[0] + self.w[1]) / 2
        om = R_EFF * (self.w[1] - self.w[0]) / L_EFF
        self.x += v * math.cos(self.th) * dt
        self.y += v * math.sin(self.th) * dt
        self.th += om * dt
        self.t += dt

    def ticks(self, i):
        return int(round(self.phi[i] * N / (2 * math.pi)))


class FakeSerial:
    """Byte-for-byte the protocol of LYNX_COMM_RPI5.ino, on a SimRover."""
    def __init__(self, rover):
        self.rover, self.out, self.pending = rover, b"", b""

    def reset_input_buffer(self):
        self.out = b""

    def write(self, data):
        self.pending += bytes(data)
        while self.pending:
            sel = self.pending[0]
            if sel in (1, 2, 3, 4):
                if len(self.pending) < 2:
                    return
                pwm = self.pending[1]
                self.pending = self.pending[2:]
                side = 0 if sel in (1, 2) else 1        # 1,2 left; 3,4 right
                self.rover.pwm[side] = pwm if sel in (1, 3) else -pwm
            elif sel in (5, 6):
                self.pending = self.pending[1:]
                t = self.rover.ticks(0 if sel == 5 else 1)
                self.out += f"{t} {t}\r\n".encode()     # front and rear encoders
            elif sel == 7:
                self.pending = self.pending[1:]
                self.rover.pwm = [0, 0]
            else:
                self.pending = self.pending[1:]

    def readline(self):
        line, sep, rest = self.out.partition(b"\n")
        self.out = rest
        return line + sep

    def close(self):
        pass


class FakeLidar:
    """Same interface as lynx_lidar.Lidar; a full turn every 1/hz s of sim time."""
    def __init__(self, rover, ell=0.10, hz=5.5):
        self.rover, self.ell, self.dt = rover, ell, 1.0 / hz
        self.psi, self.d, self.t = np.zeros(0), np.zeros(0), -1.0
        self.fault, self.turns = "", 0

    def start(self):
        pass

    def latest(self):
        r = self.rover
        if r.t - self.t >= self.dt:         # a new turn
            xl = r.x + self.ell * math.cos(r.th)
            yl = r.y + self.ell * math.sin(r.th)
            self.psi, self.d = raycast(xl, yl, r.th)
            self.t = r.t
            self.turns += 1
        return self.psi, self.d, time.monotonic()

    def stop(self):
        pass
