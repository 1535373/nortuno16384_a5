"""Real-time animation of your double pendulum simulation.  Provided -- you do
not need to edit this file.

    python visualize.py                          # default initial condition
    python visualize.py --q0 120 -30 --T 20      # angles in degrees
    python visualize.py --q0 90 0 --qd0 0 200    # velocities in degrees / s
    python visualize.py --chaos 5                # 5 pendulums, 0.001 rad apart
    python visualize.py --plot                   # also plot the joint angles

The animation plays at real time: one second of simulation per second on the
wall clock (change it with --speed).  It loops when it reaches the end.
"""

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import double_pendulum as dp


def bob_positions(q):
    """(N, 3, 2) positions of pivot, bob 1, bob 2 for a (N, 2) trajectory."""
    return np.array([dp.forward_kinematics(qi) for qi in q])


def setup_axes(ax):
    reach = 1.1 * (dp.L1 + dp.L2)
    ax.set_xlim(-reach, reach)
    ax.set_ylim(-reach, reach)
    ax.set_aspect('equal')
    ax.axhline(0, color='0.6', lw=0.8, zorder=0)
    ax.plot([0], [0], 's', color='0.3', ms=8, zorder=5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')


def animate(t, qs, labels=None, trail=1.0, speed=1.0, title=None):
    """Animate one or more trajectories at real time.

    Args:
        t:      sample times, shape (N,), shared by every trajectory.
        qs:     one (N, 2) joint-angle trajectory, or a list of them.
        labels: optional legend label per trajectory.
        trail:  seconds of bob-2 path to draw behind each pendulum.
        speed:  playback rate (1.0 = real time).
    """
    t = np.asarray(t, dtype=float)
    if np.ndim(qs) == 2:
        qs = [qs]
    pts = [bob_positions(np.asarray(q, dtype=float)) for q in qs]
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(pts))) if len(pts) > 1 else ['C0']
    trail_len = max(1, int(round(trail / (t[1] - t[0])))) if len(t) > 1 else 1

    fig, ax = plt.subplots(figsize=(6, 6))
    setup_axes(ax)
    if title:
        ax.set_title(title)

    arms, trails = [], []
    for i, color in enumerate(colors):
        label = labels[i] if labels else None
        arm, = ax.plot([], [], 'o-', color=color, lw=2.5, ms=10, label=label, zorder=3)
        tr, = ax.plot([], [], '-', color=color, lw=1, alpha=0.6, zorder=2)
        arms.append(arm)
        trails.append(tr)
    clock = ax.text(0.02, 0.97, '', transform=ax.transAxes, va='top', family='monospace')
    if labels:
        ax.legend(loc='upper right', fontsize=8)

    # Pick the frame from the wall clock rather than counting frames, so the
    # playback stays real-time even when drawing is slower than dt.
    start = [time.perf_counter()]
    loop_pause = 1.0  # seconds to hold the last frame before looping

    def update(_):
        sim_time = (time.perf_counter() - start[0]) * speed
        if sim_time > t[-1] + loop_pause * speed:
            start[0] = time.perf_counter()
            sim_time = 0.0
        k = min(int(np.searchsorted(t, sim_time)), len(t) - 1)
        lo = max(0, k - trail_len)
        for p, arm, tr in zip(pts, arms, trails):
            arm.set_data(p[k, :, 0], p[k, :, 1])
            tr.set_data(p[lo:k + 1, 2, 0], p[lo:k + 1, 2, 1])
        clock.set_text(f't = {t[k]:6.2f} s')
        return arms + trails + [clock]

    # Keep a reference to the animation, or it is garbage-collected.
    fig._anim = FuncAnimation(fig, update, interval=1000 / 60, blit=True,
                              cache_frame_data=False)
    return fig


def plot_angles(t, qs, labels=None):
    """Static plot of theta_1 and theta_2 against time."""
    if np.ndim(qs) == 2:
        qs = [qs]
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(qs))) if len(qs) > 1 else ['C0']
    for i, (q, color) in enumerate(zip(qs, colors)):
        label = labels[i] if labels else None
        axes[0].plot(t, q[:, 0], color=color, lw=1, label=label)
        axes[1].plot(t, q[:, 1], color=color, lw=1)
    axes[0].set_ylabel(r'$\theta_1$ [rad]')
    axes[1].set_ylabel(r'$\theta_2$ [rad]')
    axes[1].set_xlabel('t [s]')
    if labels:
        axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--q0', type=float, nargs=2, default=[120.0, -20.0],
                        metavar=('TH1', 'TH2'), help='initial angles [deg]')
    parser.add_argument('--qd0', type=float, nargs=2, default=[0.0, 0.0],
                        metavar=('TH1D', 'TH2D'), help='initial velocities [deg/s]')
    parser.add_argument('--T', type=float, default=15.0, help='duration [s]')
    parser.add_argument('--dt', type=float, default=1 / 60, help='sample spacing [s]')
    parser.add_argument('--speed', type=float, default=1.0, help='playback rate')
    parser.add_argument('--trail', type=float, default=1.0, help='trail length [s]')
    parser.add_argument('--chaos', type=int, default=0, metavar='N',
                        help='simulate N pendulums whose theta_2 differ by --eps')
    parser.add_argument('--eps', type=float, default=1e-3,
                        help='theta_2 spacing for --chaos [rad]')
    parser.add_argument('--plot', action='store_true', help='also plot the angles')
    args = parser.parse_args()

    q0 = np.deg2rad(args.q0)
    qd0 = np.deg2rad(args.qd0)
    n = max(1, args.chaos)
    offsets = args.eps * np.arange(n)

    qs, labels = [], None
    try:
        for d in offsets:
            t, q, _ = dp.simulate(q0 + np.array([0.0, d]), qd0, T=args.T, dt=args.dt)
            qs.append(q)
    except NotImplementedError as e:
        sys.exit(f'double_pendulum.{e} is not implemented yet -- fill in the TODO first.')
    if args.chaos:
        labels = [f'theta_2 + {d:.0e}' if d else 'theta_2' for d in offsets]

    title = (f'q0 = ({args.q0[0]:g}, {args.q0[1]:g}) deg'
             + (f', {n} pendulums {args.eps:g} rad apart' if args.chaos else ''))
    animate(t, qs, labels=labels, trail=args.trail, speed=args.speed, title=title)
    if args.plot:
        plot_angles(t, qs, labels=labels)
    plt.show()


if __name__ == '__main__':
    main()
