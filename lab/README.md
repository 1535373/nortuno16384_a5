# Assignment 5 lab: drawing RKD with the xArm7

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
python calibrate.py --manual             # practise jogging onto the corners yourself
python draw.py strokes/example.json      # try the pipeline
python sketch.py strokes/rkd.json        # draw R, K and D
python draw.py strokes/rkd.json          # -> results/rkd-sim.png
```

Then in lab, with `ROBOT_IP` set:

```bash
python check_plane.py
python calibrate.py                      # swing it over each corner, then jog it down onto it
python draw.py strokes/rkd.json --plan-only
python draw.py strokes/rkd.json          # -> results/rkd-robot.png; film it
```

`calibrate.py` keeps the simulator's calibration and the robot's apart
(`calibration-sim.json`, `calibration-robot.json`), and `draw.py` uses the one
for the arm it is driving. `sketch.py` sizes its sheet from the latest
calibration, so draw after calibrating if your real sheet isn't US letter.

## Calibrating on the real arm

1. The arm goes to the home pose of the Assignment 3 lab: marker down, a few
   centimetres over the table, joints 1, 4 and 7 free. For each corner, swing
   the arm by hand until the marker is over it (or as close as it goes) and
   press enter. Push a watched joint out of place, or the marker out of the
   area the arm can draw on, and the arm locks, asks you to let go, and moves
   back.
2. The arm moves to its drawing pose above corner 1, raising its shoulder on
   the way so the marker stays clear of the table.
3. At each corner, jog the tip down until it just touches, and press enter:

| | |
|---|---|
| w / s | away from you / towards you |
| a / d | left / right |
| r / f | raise / lower |
| 1 2 3 | steps of 0.5, 2, 10 mm |
| enter | the tip is on the corner |
| q | give up |

The marker keeps one orientation the whole time, while touching and while
drawing, so where its tip really sits on the flange doesn't matter: the offset
is the same everywhere and cancels. That is also why you don't push it onto
the corners by hand.

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
station. `MARKER_LENGTH` (flange face to marker tip) only needs to be roughly
right, within a centimetre or two: calibration cancels the rest, and the
simulator's own marker is 18 mm longer and 5 mm off-axis to prove it. The
holder has to give at least `PRESS` (1 mm).
