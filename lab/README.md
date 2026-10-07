# Assignment 5 lab: drawing KDC with the xArm7

The xArm7 holds a marker and writes on a sheet of paper. Your Assignment 4
`Robot` does the inverse kinematics; you find the paper by touching its four
corners. See Question 5 in `assignment5.pdf` for the full procedure, the
written part and the safety rules.

## Setup

The lab uses the course robot library, the same one as the Assignment 3 lab:

```bash
git clone https://github.com/CMU-16384-2026-Fall/16384-robot-lib
cd 16384-robot-lib
git pull                        # if you already had it cloned
conda env update -f environment.yml
conda activate 16384
```

Run everything below from this `lab` folder. With `ROBOT_IP` unset the scripts
drive a MuJoCo simulation, shown in your browser; with it set, the real arm.

## Files

- `robot.py`: **paste in your Assignment 4 `Robot`** (FK, Jacobians, IK).
- `plane.py`: **you edit and submit this.** The paper frame from four corners.
- `check_plane.py`: checks your `plane.py`. Run it before going to the robot.
- `calibrate.py`: touch the four corners (the simulator does it for you).
- `sketch.py`: draw the strokes you want traced, and save them.
- `draw.py`: plan the strokes with your IK, then draw them.
- `drawing_arm.py`: everything between your code and the arm (provided).
- `strokes/example.json`: a square and a triangle, to try things out.
- `handin/`: your drawing video goes here.

## The order to do it in

```bash
python check_plane.py                    # until it reports no failures
python calibrate.py                      # simulator: finds its (slightly crooked) sheet
python draw.py strokes/example.json      # try the pipeline
python sketch.py strokes/kdc.json        # draw K, D and C
python draw.py strokes/kdc.json          # -> results/kdc-sim.png
```

Then in lab, with `ROBOT_IP` set:

```bash
python check_plane.py
python calibrate.py                      # you push the marker onto each corner
python draw.py strokes/kdc.json --plan-only
python draw.py strokes/kdc.json          # -> results/kdc-robot.png; film it
```

`calibrate.py` keeps the simulator's calibration and the robot's apart
(`calibration-sim.json`, `calibration-robot.json`), and `draw.py` uses the one
for the arm it is driving. `sketch.py` sizes its sheet from the latest
calibration, so draw after calibrating if your real sheet isn't US letter.

## Sketch controls

| | |
|---|---|
| left click | add a point to the current stroke |
| right click, enter | finish the stroke |
| backspace, u | take back the last point |
| g | 5 mm snap on/off |
| c | clear everything |
| s, or close the window | save |

## Lab staff

The constants at the top of `drawing_arm.py` are the ones to check at a
station: `MARKER_LENGTH` (flange face to marker tip) has to be measured for the
holder in use, and the holder has to give at least `PRESS` (1 mm).
