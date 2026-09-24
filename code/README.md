# Assignment 5: Double Pendulum Simulation

Simulate the double pendulum from the written section with the equations of
motion you derived there. See `assignment5.pdf` for the full writeup.

## Files

- `double_pendulum.py`: **the file you edit and submit.** Fill in
  `total_energy`, `dynamics` and `simulate`.
- `visualize.py`: animates your simulation in real time (provided).
- `energy_check.py`: plots how much your simulation's total energy drifts,
  which should be almost none (provided).
- `autograder.py`: checks your `total_energy`, `dynamics` and `simulate`
  against the reference answers in `tests/expected.npz`.
- `create_submission.py`: packages `double_pendulum.py` into the zip you upload.

You need `numpy`, `scipy` and `matplotlib`.

## Conventions

- `q = [theta_1, theta_2]`. Both angles are measured from the downward vertical,
  and counterclockwise is positive. `q = [0, 0]` hangs straight down.
- There is no applied torque: the pendulum swings freely under gravity.
- Use the constants `M1`, `M2`, `L1`, `L2`, `G` in `double_pendulum.py` instead
  of writing the numbers into your code. The autograder changes them.

## Running things

All of these run from this folder:

```bash
python autograder.py
python energy_check.py
python visualize.py
python visualize.py --q0 120 -30 --T 20
python visualize.py --chaos 5 --plot
```

`visualize.py` takes its initial angles in degrees (`--q0`) and its initial
velocities in degrees per second (`--qd0`). `--chaos N` overlays N pendulums
whose starting `theta_2` values differ by `--eps` radians (0.001 by default).
Run `python visualize.py --help` for the other options.

