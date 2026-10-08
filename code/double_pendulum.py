"""Double pendulum dynamics and simulation.  THIS IS THE FILE YOU EDIT AND SUBMIT.

The system is the double pendulum you derived previously: point masses M1 and
M2 on massless rods of length L1 and L2.  Both angles are measured from the
downward vertical (absolute angles: theta_2 is link 2's angle, not its angle
relative to link 1).  There is no
applied torque: the pendulum swings freely under gravity.

    q  = [theta_1, theta_2]            (radians)
    qd = [theta_1_dot, theta_2_dot]    (radians / second)

Fill in the three TODOs: `total_energy`, `dynamics` and `simulate`.
"""

import numpy as np
from scipy.integrate import solve_ivp

# Physical parameters.  Use these names in your code -- do not hard-code the
# numbers.  The autograder changes them to test other pendulums.
M1 = 1.0    # mass of bob 1 [kg]
M2 = 1.0    # mass of bob 2 [kg]
L1 = 1.0    # length of link 1 [m]
L2 = 1.0    # length of link 2 [m]
G = 9.81    # gravitational acceleration [m/s^2]

# Integrator tolerances.  If you use solve_ivp, pass these as rtol= and atol=.
# Loose tolerances let small errors pile up, and the energy drifts.
RTOL = 1e-9
ATOL = 1e-9


def forward_kinematics(q):
    """Positions of the pivot, bob 1 and bob 2, as a (3, 2) array of (x, y).

    Provided for you (the visualizers use it).  The pivot is at the origin, x
    points right and y points up, so the hanging rest configuration q = [0, 0]
    puts both bobs straight below the pivot.
    """
    th1, th2 = q
    p1 = np.array([L1 * np.sin(th1), -L1 * np.cos(th1)])
    p2 = p1 + np.array([L2 * np.sin(th2), -L2 * np.cos(th2)])
    return np.array([[0.0, 0.0], p1, p2])


def total_energy(q, qd):
    """Total energy T + V of the double pendulum.

    Args:
        q:  joint angles, shape (N, 2): column 0 is theta_1, column 1 theta_2.
        qd: joint velocities, shape (N, 2).

    Returns:
        energy at each of the N samples, shape (N,).

    Use your kinetic and potential energy from your derivation.  Measure
    the potential energy from the pivot height (y = 0), with y pointing up.
    """
    q, qd = np.asarray(q, dtype=float), np.asarray(qd, dtype=float)
    th1, th2 = q[:, 0], q[:, 1]
    th1d, th2d = qd[:, 0], qd[:, 1]

    # TODO: compute the kinetic energy T and potential energy V.
    raise NotImplementedError("total_energy")


def dynamics(q, qd):
    """Joint accelerations of the double pendulum.

    Args:
        q:  joint angles [theta_1, theta_2], shape (2,).
        qd: joint velocities [theta_1_dot, theta_2_dot], shape (2,).

    Returns:
        qdd: joint accelerations [theta_1_ddot, theta_2_ddot], a numpy array of
             shape (2,).

    Hint: your Euler-Lagrange equations are linear in
    qdd.  Write them as  M(q) qdd = b(q, qd)  with a 2x2 matrix M, and
    solve for qdd with np.linalg.solve.
    """
    # TODO: implement the equations of motion.
    raise NotImplementedError("dynamics")


def simulate(q0, qd0, T=10.0, dt=0.01):
    """Simulate the double pendulum from an initial state.

    Args:
        q0:     initial joint angles, shape (2,).
        qd0:    initial joint velocities, shape (2,).
        T:      duration of the simulation [s].
        dt:     spacing of the returned samples [s].

    Returns:
        (t, q, qd):
            t:  sample times 0, dt, 2 dt, ..., T, shape (N,).  Use
                np.arange(0, T + dt / 2, dt) so T itself is included.
            q:  joint angles at those times, shape (N, 2).
            qd: joint velocities at those times, shape (N, 2).

    We recommend scipy's solve_ivp, but you may write your own integrator.
    With solve_ivp:
        1. solve_ivp integrates a first-order system  x_dot = f(t, x).  Stack
           the state as x = [theta_1, theta_2, theta_1_dot, theta_2_dot] and
           write f(t, x) using your `dynamics` function.
        2. Call solve_ivp with t_eval set to the sample times, and pass
           rtol=RTOL, atol=ATOL.  Use t_span = (0, t_eval[-1]).
        3. Unpack the solution (sol.t and sol.y, which has shape (4, N)) into
           t, q and qd.
    """
    # TODO: simulate the double pendulum.
    raise NotImplementedError("simulate")
