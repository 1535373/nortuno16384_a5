"""Self-check for Assignment 5.  Run this before you submit:

    python autograder.py

It runs your `total_energy`, `dynamics` and `simulate` from double_pendulum.py
on the cases in tests/expected.npz and compares the results against the
reference answers.  For anything that does not match it prints the input, what
was expected, and what you gave.

Some cases change the pendulum's masses, lengths and gravity (by setting
double_pendulum.M1, .M2, .L1, .L2 and .G), so use those names in your code
rather than the numbers.

This is a convenience tool, not the grader.  The Gradescope autograder runs the
same comparisons plus additional randomly generated cases.
"""

import importlib
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, 'tests', 'expected.npz')
ENERGY_TOL = 1e-6   # total energy [J]
DYN_TOL = 1e-6      # joint accelerations [rad/s^2]
# Loose enough for simple fixed-step integrators (explicit or semi-implicit
# Euler with dt = 0.01) over the 1 s test horizon.
SIM_Q_TOL = 0.25    # simulated angles [rad]
SIM_QD_TOL = 1.5    # simulated velocities [rad/s]


def vec(values):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        return f'{values: .6f}'
    return '[' + ', '.join(f'{v: .6f}' for v in values.reshape(-1)) + ']'


def block(rows, indent='    '):
    width = max(len(label) for label, _ in rows)
    return '\n'.join(f'{indent}{label:<{width}} = {vec(value)}' for label, value in rows)


class Checker:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def report_pass(self, name):
        self.passed += 1
        print(f'[ OK ] {name}')

    def report_fail(self, name, message):
        self.failed += 1
        print(f'[FAIL] {name}')
        print(message)

    def run(self, name, fn):
        """Call ``fn``; turn any exception from student code into a failure."""
        try:
            fn()
        except Exception:
            self.report_fail(name, '    your code raised an exception:\n' + '\n'.join(
                '    ' + line for line in traceback.format_exc().strip().splitlines()))

    def summary(self):
        print()
        print(f'{self.passed} passed, {self.failed} failed')
        if self.failed == 0:
            print('Everything matches.  Run create_submission.py to package your handin.')
        return self.failed == 0


def set_params(dp, names, values):
    for name, value in zip(names, values):
        setattr(dp, name, float(value))


def params_rows(names, values):
    return [(str(n), v) for n, v in zip(names, values)]


def check_energy(checker, dp, data):
    names = [str(n) for n in data['param_names']]
    for i, params in enumerate(data['dyn_params']):
        name = f'total_energy, parameter set {i}' + (' (default)' if i == 0 else '')

        def run():
            set_params(dp, names, params)
            q, qd, expected = data['dyn_q'][i], data['dyn_qd'][i], data['dyn_energy'][i]
            got = np.asarray(dp.total_energy(q.copy(), qd.copy()), dtype=float)
            if got.shape != expected.shape:
                checker.report_fail(name, f'    for q and qd of shape {q.shape}, expected '
                                          f'shape {expected.shape}, got shape {got.shape}')
                return
            err = np.abs(got - expected)
            err[~np.isfinite(err)] = np.inf
            if err.max() <= ENERGY_TOL:
                checker.report_pass(name)
                return
            k = int(np.argmax(err))
            checker.report_fail(name, '\n'.join([
                '  parameters:', block(params_rows(names, params)),
                '  input (the sample that is furthest off):',
                block([('q', q[k]), ('qd', qd[k])]),
                '  expected:', block([('energy', expected[k])]),
                '  you gave:', block([('energy', got[k])])]))

        checker.run(name, run)


def check_dynamics(checker, dp, data):
    names = [str(n) for n in data['param_names']]
    for i, params in enumerate(data['dyn_params']):
        name = f'dynamics, parameter set {i}' + (' (default)' if i == 0 else '')

        def run():
            set_params(dp, names, params)
            worst, worst_k, worst_got = -1.0, None, None
            for k in range(data['dyn_q'].shape[1]):
                q, qd = data['dyn_q'][i, k], data['dyn_qd'][i, k]
                got = np.asarray(dp.dynamics(q.copy(), qd.copy()), dtype=float)
                if got.shape != (2,):
                    checker.report_fail(name, f'    dynamics must return shape (2,), '
                                              f'got shape {got.shape}')
                    return
                err = np.max(np.abs(got - data['dyn_qdd'][i, k]))
                err = np.inf if not np.isfinite(err) else err
                if err > worst:
                    worst, worst_k, worst_got = err, k, got
            if worst <= DYN_TOL:
                checker.report_pass(name)
                return
            k = worst_k
            checker.report_fail(name, '\n'.join([
                '  parameters:', block(params_rows(names, params)),
                '  input (the case that is furthest off):',
                block([('q', data['dyn_q'][i, k]), ('qd', data['dyn_qd'][i, k])]),
                '  expected:', block([('qdd', data['dyn_qdd'][i, k])]),
                '  you gave:', block([('qdd', worst_got)])]))

        checker.run(name, run)


def check_simulate(checker, dp, data):
    names = [str(n) for n in data['param_names']]
    T, dt = float(data['sim_T']), float(data['sim_dt'])
    for c in range(len(data['sim_q0'])):
        name = f'simulate, case {c}'

        def run():
            set_params(dp, names, data['sim_params'][c])
            out = dp.simulate(data['sim_q0'][c].copy(), data['sim_qd0'][c].copy(), T=T, dt=dt)
            t, q, qd = (np.asarray(x, dtype=float) for x in out)
            exp_t, exp_q, exp_qd = data['sim_t'][c], data['sim_q'][c], data['sim_qd'][c]
            inputs = block(params_rows(names, data['sim_params'][c])
                           + [('q0', data['sim_q0'][c]), ('qd0', data['sim_qd0'][c]),
                              ('T', T), ('dt', dt)])
            for label, got, exp in (('t', t, exp_t), ('q', q, exp_q), ('qd', qd, exp_qd)):
                if got.shape != exp.shape:
                    checker.report_fail(name, f'  input:\n{inputs}\n'
                                              f'    expected {label} with shape {exp.shape}, '
                                              f'got shape {got.shape}')
                    return
            if np.max(np.abs(t - exp_t)) > 1e-9:
                checker.report_fail(name, f'  input:\n{inputs}\n'
                                          f'    t should be 0, dt, 2 dt, ..., T; you gave '
                                          f't[:3] = {vec(t[:3])}, t[-1] = {t[-1]:.6f}')
                return
            bad = ((np.abs(q - exp_q).max(axis=1) > SIM_Q_TOL)
                   | (np.abs(qd - exp_qd).max(axis=1) > SIM_QD_TOL)
                   | ~np.isfinite(q).all(axis=1) | ~np.isfinite(qd).all(axis=1))
            if not bad.any():
                checker.report_pass(name)
                return
            k = int(np.argmax(bad))
            checker.report_fail(name, '\n'.join([
                f'  input:\n{inputs}',
                f'  first disagreement at t = {exp_t[k]:.2f} s '
                f'(tolerance {SIM_Q_TOL:g} rad, {SIM_QD_TOL:g} rad/s):',
                '  expected:', block([('q', exp_q[k]), ('qd', exp_qd[k])]),
                '  you gave:', block([('q', q[k]), ('qd', qd[k])])]))

        checker.run(name, run)


def main():
    data = dict(np.load(DATA))
    checker = Checker()
    try:
        dp = importlib.import_module('double_pendulum')
    except Exception:
        checker.report_fail('import double_pendulum.py', '\n'.join(
            '    ' + line for line in traceback.format_exc().strip().splitlines()))
        checker.summary()
        return 1

    names = [str(n) for n in data['param_names']]
    defaults = [getattr(dp, n) for n in names]
    try:
        check_energy(checker, dp, data)
        check_dynamics(checker, dp, data)
        check_simulate(checker, dp, data)
    finally:
        set_params(dp, names, defaults)
    return 0 if checker.summary() else 1


if __name__ == '__main__':
    sys.exit(main())
