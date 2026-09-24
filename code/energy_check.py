"""Energy sanity check for your simulation.

The double pendulum conserves its total energy T + V, so a correct `dynamics`
and `simulate` keep the energy flat.  This script simulates the pendulum for 20
seconds and plots its total energy over time.

    python energy_check.py

It uses your `simulate` and `total_energy` from double_pendulum.py.  Provided
-- you do not need to edit this file.  Put the plot in your writeup (part c).
"""

import sys

import matplotlib.pyplot as plt
import numpy as np

import double_pendulum as dp


def main():
    q0 = np.deg2rad([120.0, -20.0])
    qd0 = np.zeros(2)

    try:
        t, q, qd = dp.simulate(q0, qd0, T=20.0, dt=0.01)
        energy = dp.total_energy(q, qd)
    except NotImplementedError as e:
        sys.exit(f'double_pendulum.{e} is not implemented yet -- fill in the TODO first.')
    print(f'starting energy {energy[0]:.6g} J, '
          f'largest change {np.max(np.abs(energy - energy[0])):.3g} J')

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, energy)
    # Show at least 1 J of range, so tiny numerical wiggles in a conserved
    # energy are not blown up to fill the whole plot.
    mid = 0.5 * (energy.max() + energy.min())
    half = max(0.5, 0.55 * (energy.max() - energy.min()))
    ax.set_ylim(mid - half, mid + half)
    ax.ticklabel_format(axis='y', useOffset=False)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('total energy E [J]')
    ax.set_title('Total energy of the double pendulum')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
