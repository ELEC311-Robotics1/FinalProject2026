# ELEC 311 Final Project 2026: Lidar-Informed Fuzzy Velocity Field

The Lynxmotion rover follows a fuzzy velocity field toward a contour, and avoids
obstacles sensed only by its RPLIDAR A1. Embry-Riddle Aeronautical University, Prescott.

## What is here

| File | What it does |
|---|---|
| `LYNX_FINAL_Field.py` | The field, with the wheels driven by the calibrated motor map (needs `k` and `PWM0` from the W10 calibration) |
| `LYNX_CONTROL_FINAL.py` | The same field, with the wheels driven by the model-free law of the paper, `PWM = -K1 tanh(NU1 S)`; no calibration |
| `lib/` | Field, contour, kinematics, odometry, lidar driver, serial protocol, simulator |
| `checks/CHECK_Lidar.py` | Live lidar check: mounting angle and mask |
| `checks/CHECK_DeskField.py` | The field on the desk, wheels off the ground |
| `firmware/LYNX_COMM_RPI5/` | Arduino Mega sketch (needs the `Encoder` and `Sabertooth` libraries) |

## On the rover

```bash
sudo iw wlan0 set power_save off
cd ~
rm -rf FinalProject2026
git clone https://github.com/ELEC311-Robotics1/FinalProject2026.git
cd FinalProject2026
```

Find the lidar port (`ls /dev/ttyUSB*`), then, on the floor, in clear space:

```bash
python3 LYNX_FINAL_Field.py k PWM0 /dev/ttyUSB1 --auto 60          # motor map
python3 LYNX_CONTROL_FINAL.py /dev/ttyUSB1 --auto 60               # model-free law
```

Add `--target` to put the contour around the red object seen by the camera,
and `--stop` (with `--target`) to stop in front of it:

```bash
python3 LYNX_CONTROL_FINAL.py /dev/ttyUSB1 --auto 150 --target --stop
```

| Option | Meaning |
|---|---|
| `--auto S` | No keys: waits 5 s, zeros the pose where the rover sits, drives S seconds, stops |
| `--target` | Camera bearing + lidar range locate the red object; the contour is a circle around it |
| `--stop` | With `--target`: drive to the object and stop 0.5 m from it |
| `--fake` | Laptop, no rover: the same code on a simulated rover, lidar and maze |

Without `--auto`, the keys are: `g` go, `space` stop, `z` zero the pose here, `q` quit.
Emergency stop from another terminal: `pkill -f LYNX_`.

Each run saves `field_log_<time>.csv` or `control_log_<time>.csv` in the folder you ran it from.
Copy it off the rover before deleting the clone, from the laptop:

```bash
scp elec@<rover address>:~/FinalProject2026/*_log_*.csv .
```

## On a laptop

```bash
pip install -r requirements.txt
python3 LYNX_CONTROL_FINAL.py --fake
```
