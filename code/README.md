# Assignment 5: Double Pendulum Simulation

Simulate the double pendulum with the equations of motion you derived
previously. See `assignment5.pdf` for the full writeup.

Question 2, the hands-on lab (drawing with the xArm7), lives in `../lab`; see
`../lab/README.md`. Its `plane.py` is graded by the same autograder as this
folder's code (`python local_autograder.py -p plane`).

## Files

- `double_pendulum.py`: **the file you edit and submit.** Fill in
  `total_energy`, `dynamics` and `simulate`.
- `local_autograder.py`: checks your `total_energy`, `dynamics` and `simulate`,
  and the lab's `plane.py`, against the test cases in `tests/`.
- `visualize.py`: animates your simulation in real time (provided).
- `energy_check.py`: plots how much your simulation's total energy drifts,
  which should be almost none (provided).
- `create_submission.py`: packages `double_pendulum.py` and the lab's files
  (`../lab/robot.py`, `../lab/plane.py`, strokes, calibrations, results and
  `../lab/handin`) into the zip you upload.

You need `numpy`, `scipy` and `matplotlib`.

## Conventions

- `q = [theta_1, theta_2]`. Both angles are measured from the downward vertical,
  and counterclockwise is positive. `q = [0, 0]` hangs straight down.
- There is no applied torque: the pendulum swings freely under gravity.
- Use the constants `M1`, `M2`, `L1`, `L2`, `G` in `double_pendulum.py` instead
  of writing the numbers into your code. The autograder changes them.

## Checking your work

All of these run from this folder:

```bash
python local_autograder.py                 # everything
python local_autograder.py -p dynamics     # one part: energy, dynamics, simulate or plane
python local_autograder.py -v              # list every failing case, not just the first few
```

This runs your code on the test cases in `tests/inK.txt` and compares the
results with `tests/outK.txt` (`in4.txt` is the lab's `plane.py`). Many
pendulum cases use a different pendulum (other masses, lengths and gravity),
which is why your code has to use `M1`, `M2`, `L1`, `L2` and `G`. Each failure shows the input and what was expected.

Gradescope runs the same tests on randomly perturbed versions of these cases,
with the answers computed fresh each time. Code that is correct in general
scores the same on both; code that only works for these exact numbers (or only
for the default pendulum) does not.

`simulate` is compared loosely, so a simple integrator you write yourself
(Euler with `dt = 0.01`, for instance) can pass as well as `solve_ivp` does.
`total_energy` and `dynamics` have exact answers and are compared closely.

## Seeing it

```bash
python visualize.py
python visualize.py --q0 120 -30 --T 20
python visualize.py --chaos 5 --plot
python energy_check.py
```

`visualize.py` takes its initial angles in degrees (`--q0`) and its initial
velocities in degrees per second (`--qd0`). `--chaos N` overlays N pendulums
whose starting `theta_2` values differ by `--eps` radians (0.001 by default).
Run `python visualize.py --help` for the other options.

## Submitting

```bash
python create_submission.py
```

This writes `submission.zip`. Rename it to `<andrew_id>.zip` and upload it to
Gradescope.
