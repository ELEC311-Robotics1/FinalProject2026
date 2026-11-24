#!/usr/bin/python3
# ================================================================
# lynx_io.py
# The serial helpers of LYNX_HO_W10_Aut.py and LYNX_HO_W10_Calibration.py,
# in one place. Same protocol as LYNX_COMM_RPI5.ino (Mega now, Teensy later):
#   sel=1,pwm -> left forward      sel=2,pwm -> left backward
#   sel=3,pwm -> right forward     sel=4,pwm -> right backward
#   sel=5     -> left encoders (FL, RL)
#   sel=6     -> right encoders (FR, RR)
#   sel=7     -> watchdog reset
# The only change: the functions take the port as their first argument,
# so the same code drives the rover or the simulator (lynx_sim.FakeSerial).
# ================================================================

import math
import time

N = 12000                    # ticks per wheel turn


def open_rover(port=None):
    """Open the rover's serial port, as in every LYNX script (port=None: ask)."""
    import serial
    from lynx_port import get_port
    ser = serial.Serial(port=port or get_port(), baudrate=115200, timeout=0.05)
    time.sleep(2)
    ser.readline()
    return ser


def read_pair(ser, sel):
    ser.reset_input_buffer()
    ser.write(bytes([sel]))
    try:
        a, b = ser.readline().decode().strip().split()
        return int(a), int(b)
    except (ValueError, UnicodeDecodeError):
        return None, None


_prev = None                 # last raw counts (FL, RL, FR, RR)
_phi = [0.0, 0.0]            # wheel angles (left, right), rad


def read_angles(ser):
    """Wheel angles (phiL, phiR) in rad, or None if a read failed.
    Each side is the average of its front and rear encoders; if one of the
    two did not move while the other did (a dead or loose encoder), that
    step uses the one that moved."""
    global _prev
    fl, rl = read_pair(ser, 5)
    fr, rr = read_pair(ser, 6)
    if fl is None or fr is None:
        return None
    c = (fl, rl, fr, rr)
    if _prev is not None:
        for side, (a, b) in enumerate(((0, 1), (2, 3))):
            da, db = c[a] - _prev[a], c[b] - _prev[b]
            if da == 0 and abs(db) > 2:
                d = db                          # front dead: use the rear
            elif db == 0 and abs(da) > 2:
                d = da                          # rear dead: use the front
            else:
                d = (da + db) / 2.0
            _phi[side] += 2 * math.pi * d / N
    _prev = c
    return _phi[0], _phi[1]


def send_motor(ser, channel, u):
    pwm = min(int(abs(u)), 127)
    sel = (1 if u >= 0 else 2) if channel == 1 else (3 if u >= 0 else 4)
    ser.write(bytes([sel, pwm]))


def stop_all(ser):
    ser.write(bytes([1, 0]))
    ser.write(bytes([3, 0]))


def close_rover(ser):
    """Motors off, watchdog reset, port closed (as in W10)."""
    try:
        stop_all(ser)
        time.sleep(0.1)
        ser.write(bytes([7]))
        ser.close()
    except Exception:
        pass
