# Assignment 5 lab: drawing RKD with the xArm7

The xArm7 holds a marker and writes on a sheet of paper. Your Assignment 4
`Robot` does the inverse kinematics; you find the paper by touching its four
corners. See Question 2 in `assignment5.pdf` for the full procedure, the
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

- `robot.py`: **copy your Assignment 4 `robot.py` over this one**, whole and
  unchanged: `cp ../../Assignment4/ex2/robot.py robot.py` (ex1, ex2 and ex3
  hold the same class; take the one with your IK in it).
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
python calibrate.py --manual             # practise touching the corners (with the keyboard)
python draw.py strokes/example.json      # try the pipeline
python sketch.py strokes/rkd.json        # draw R, K and D
python draw.py strokes/rkd.json          # -> results/rkd-sim.png
```

Then in lab, with `ROBOT_IP` set (the sheet is already taped down, about 50 cm
straight out from the base; leave it there):

```bash
python check_plane.py
python calibrate.py                      # marker in; bring it over each corner, then touch each by hand
python draw.py strokes/rkd.json --plan-only
python draw.py strokes/rkd.json          # -> results/rkd-robot.png; film it
```

`calibrate.py` keeps the simulator's calibration and the robot's apart
(`calibration-sim.json`, `calibration-robot.json`), and `draw.py` uses the one
for the arm it is driving. `sketch.py` sizes its sheet from the latest
calibration, so draw after calibrating if your real sheet isn't US letter.

## Calibrating on the real arm

0. Before the arm moves, the gripper opens (hold any marker already in it).
   Put the marker between the fingers, tip down and straight, press enter,
   hold it until they close on it, then let go. To calibrate again with the
   same marker still in, use `python calibrate.py --keep-marker`; a marker
   put back in needs a new calibration, since it never sits exactly where
   the last one did.
1. The arm goes to the home pose of the Assignment 3 lab and every joint is
   let go. For each corner, bring the marker over it, down to a centimetre or
   two above it, and press enter. Take the marker out of the area the arm can
   draw on and the arm locks, asks you to let go, and moves back.
2. For each corner, the arm puts the marker upright a little over it and lets
   go again. Put the tip on the corner, keeping the marker upright, and press
   enter. The script says how far the marker leant; more than 5 degrees and
   it suggests calibrating again. The arm then takes the marker to the next
   corner, never lower than the paper.

Nothing that goes wrong ends the run, short of Ctrl-C. A fault the controller
latches (a bump, the e-stop, "abnormal movement in manual mode") is cleared,
after it asks you to release the e-stop if that's what it was. An arm pushed
past the edge of where it may be moves itself back. The script also starts
from an arm left faulted or out of place. Each step is saved as it is done,
so a run stopped part-way offers to pick up where it stopped (as long as the
marker hasn't moved in the gripper).

Any order round the sheet works; the corners are put in order for you, with
corner 1 to 2 along the long edge. The controller's own safety fence is off
while calibrating (it stops the flange short of the sheet's near edge) and
back on when it ends; the library's safety box stays on throughout.

The marker is upright at every touch and at every point of the drawing, so
where its tip really sits on the flange doesn't matter: the offset is the same
everywhere and cancels. That is why the marker has to be upright when you
touch a corner.

On the real arm, `draw.py` solves the drawing with a 7-joint IK of ours rather
than your planar one: the real sheet can sit lower than the planar chain
reaches. It keeps joints 3 and 5 at zero wherever it can, so where your IK
reaches, it finds the same answer; the plan says how far they left the plane.

In the simulator, `python calibrate.py --manual` practises the touches with
the keyboard instead:

| | |
|---|---|
| w / s | away from you / towards you |
| a / d | left / right |
| r / f | raise / lower |
| 1 2 3 | steps of 0.5, 2, 10 mm |
| enter | the tip is on the corner |
| q | give up |

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
marker is held in the xArm Gripper, so measure from the flange face to the tip
with a marker in. The holder has to give at least `PRESS` (1 mm), and the tip
must be less than about 16 cm from the flange, since the flange is 17 cm over
the table in the home pose. Every pose with joints 1, 4 and 7 vertical puts
the flange there, so a short marker sits well above the paper during the
swing; calibrate.py takes it down to `JOG_START` (3 cm, by `MARKER_LENGTH`)
for the jog, which is why `MARKER_LENGTH` should be right to a centimetre.

Tape a US-letter sheet in landscape, centred 50 cm straight out from the base
(edges 39 and 61 cm out), where the simulator's sheet is. From there every
corner can be reached in the drawing pose.
