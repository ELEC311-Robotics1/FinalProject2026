#!/usr/bin/python3
# ================================================================
# lynx_lidar.py
# RPLIDAR A1 in a thread (the protocol of RPLidar_Live.py, pyserial only).
# latest() gives the last full turn IN THE ROVER FRAME:
#   psi: angle in rad, counterclockwise, 0 = straight ahead
#   d:   range in m
# The A1 reports angles CLOCKWISE, 0 deg = toward its motor (flat end).
#   MOUNT_DEG: where the A1's 0 deg points, counterclockwise from the
#   rover's forward direction (0 if the motor end faces forward).
#   psi = MOUNT_DEG - angle
# MASK: (psi_lo_deg, psi_hi_deg, r_max_m) sectors where the A1 sees the
#   rover itself (chassis, wires); returns closer than r_max are dropped.
#   Measure them with checks/CHECK_Lidar.py.
# ================================================================

import math
import threading
import time

import numpy as np

SCAN  = b"\xa5\x20"          # start scanning
STOP  = b"\xa5\x25"          # stop scanning
RESET = b"\xa5\x40"          # restart the A1 core
REPLY = b"\xa5\x5a\x05\x00\x00\x40\x81"
MIN_QUALITY = 10             # drop weak returns
MIN_R, MAX_R = 0.15, 6.0     # m, A1 limits


def measure(p):
    """5 bytes -> (new turn?, quality, angle deg, range mm), or None."""
    s, ns = p[0] & 1, (p[0] >> 1) & 1
    if s == ns or not (p[1] & 1):
        return None
    angle = ((p[1] >> 1) | (p[2] << 7)) / 64.0
    dist = (p[3] | (p[4] << 8)) / 4.0
    return s == 1, p[0] >> 2, angle, dist


def to_rover(angle_deg, dist_mm, mount_deg=0.0, mask=()):
    """A1 angles (deg, clockwise) and ranges (mm) -> rover frame (rad, m)."""
    a = np.asarray(angle_deg, float)
    d = np.asarray(dist_mm, float) / 1000.0
    psi = np.radians((mount_deg - a + 180.0) % 360.0 - 180.0)
    keep = (d >= MIN_R) & (d <= MAX_R)
    psid = np.degrees(psi)
    for lo, hi, rmax in mask:
        inside = ((psid - lo) % 360.0) <= ((hi - lo) % 360.0)
        keep &= ~(inside & (d < rmax))
    return psi[keep], d[keep]


def nearest(psi, d, deg=5.0, gap=0.05):
    """Nearest return that has a neighbour within deg and gap (no lone points).
    Returns (d, psi) or (inf, 0) if there is none."""
    if len(d) == 0:
        return math.inf, 0.0
    order = np.argsort(d)
    for i in order[:20]:
        dpsi = np.abs((psi - psi[i] + math.pi) % (2 * math.pi) - math.pi)
        close = (dpsi < math.radians(deg)) & (np.abs(d - d[i]) < gap)
        if close.sum() >= 2:                # itself and at least one more
            return float(d[i]), float(psi[i])
    i = order[0]
    return float(d[i]), float(psi[i])


class Lidar:
    def __init__(self, port, mount_deg=0.0, mask=()):
        self.port, self.mount, self.mask = port, mount_deg, mask
        self.lock = threading.Lock()
        self.psi, self.d, self.t = np.zeros(0), np.zeros(0), 0.0
        self.turns, self.bad, self.fault = 0, 0, ""
        self.stats = dict(raw=0, zero=0, lowq=0)
        self.running = False

    def start(self):
        import serial
        self.ser = serial.Serial(self.port, 115200, timeout=1)
        self._start_scan()
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()

    def latest(self):
        """(psi rad, d m, t) of the last full turn."""
        with self.lock:
            return self.psi, self.d, self.t

    def stop(self):
        self.running = False
        try:
            self.ser.write(STOP)
            self.ser.dtr = True             # motor off
        except Exception:
            pass
        try:
            self.ser.close()
        except Exception:
            pass

    def _start_scan(self):
        """Spin up, start a scan; 3 tries, reset on the last (as RPLidar_Live)."""
        ser = self.ser
        ser.dtr = False                     # motor on
        for attempt in range(1, 4):
            if attempt == 3:
                ser.write(RESET)
                time.sleep(2.0)
            ser.write(STOP)
            time.sleep(0.1)
            if attempt == 1:
                time.sleep(1.5)             # motor up to speed
            ser.reset_input_buffer()
            ser.write(SCAN)
            if ser.read(7) == REPLY:
                return
        raise IOError("A1: no reply after 3 tries (port, TX/RX wires, motor)")

    def _run(self):
        # A turn ends on the A1's start flag, as in RPLidar_Live.py
        turn, buf = [], b""
        raw = zero = lowq = 0                       # this turn, for the check
        try:
            while self.running:
                buf += self.ser.read(max(5, self.ser.in_waiting))
                while len(buf) >= 5:
                    m = measure(buf[:5])
                    if m is None:           # out of step: slide 1
                        buf, self.bad = buf[1:], self.bad + 1
                        continue
                    buf = buf[5:]
                    new, q, a, dist = m
                    if new and raw > 10:            # one full turn done
                        arr = np.array(turn) if turn else np.zeros((0, 2))
                        psi, d = to_rover(arr[:, 0], arr[:, 1], self.mount, self.mask)
                        with self.lock:
                            self.psi, self.d, self.t = psi, d, time.monotonic()
                            self.stats = dict(raw=raw, zero=zero, lowq=lowq)
                        self.turns += 1
                        turn, raw, zero, lowq = [], 0, 0, 0
                    raw += 1
                    if dist <= 0:
                        zero += 1
                    elif q < MIN_QUALITY:
                        lowq += 1
                    else:
                        turn.append((a, dist))
        except Exception as e:              # loose cable, unplugged
            self.fault = f"LIDAR LOST: {e.__class__.__name__}"
