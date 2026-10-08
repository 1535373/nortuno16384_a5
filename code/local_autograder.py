"""Autograder for HW5: the double pendulum (Q1) and the drawing lab's paper frame (Q2).

    python local_autograder.py                 # grade everything
    python local_autograder.py -p dynamics     # one part: energy, dynamics, simulate or plane
    python local_autograder.py -v              # list every failing case

It runs your `total_energy`, `dynamics` and `simulate` from double_pendulum.py,
and your `plane_frame`, `paper_to_robot` and `robot_to_paper` from
../lab/plane.py, on the cases in tests/inK.txt, and compares the results with
tests/outK.txt.

Many pendulum cases use other masses, lengths and gravity, which they set by
assigning double_pendulum.M1, .M2, .L1, .L2 and .G, so use those names in your
code rather than the numbers.

Needs only numpy (plus whatever double_pendulum.py imports).
"""

import argparse
import contextlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "tests"
LAB = ROOT.parent / "lab"

# total_energy and dynamics are closed-form, so any correct rearrangement of
# them agrees to rounding. Relative to the answer's size, since a short,
# heavy pendulum can have accelerations in the hundreds.
TOL = 1e-6
# simulate is compared loosely enough that a fixed-step Euler integrator
# (explicit or semi-implicit) at dt = 0.01 passes over the 1 s test horizon:
# the handout invites students to write their own. The case generator keeps
# only cases where Euler stays well inside these.
SIM_Q_TOL = 0.25    # rad
SIM_QD_TOL = 1.5    # rad/s
SIM_T_TOL = 1e-9    # s
# The paper frame is closed-form too: metres and unit vectors, to rounding.
PLANE_TOL = 1e-6

PARAM_NAMES = ("M1", "M2", "L1", "L2", "G")
DEFAULTS = {"M1": 1.0, "M2": 1.0, "L1": 1.0, "L2": 1.0, "G": 9.81}

MAX_FAILURES_SHOWN = 10
MAX_DETAIL_CHARS = 200


def _load_module(path):
    """Import a student file by path, under its own name."""
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_jsonl(path):
    """Stream one JSON object per line."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _close(actual, expected, tol=TOL):
    """Equal up to `tol`, relative to the answer's size (absolute below 1).

    Fails rather than raising on garbage, and shapes must match exactly:
    allclose alone would broadcast a scalar against a (2,) and call it a match.
    """
    try:
        actual = np.asarray(actual, dtype=float)
        expected = np.asarray(expected, dtype=float)
        scale = max(1.0, float(np.max(np.abs(expected)))) if expected.size else 1.0
        return actual.shape == expected.shape and bool(
            np.allclose(actual, expected, atol=tol * scale, rtol=0)
        )
    except (TypeError, ValueError):
        return False


def _array(value, name, shape):
    """`value` as a float array of `shape`, or a ValueError saying what it was."""
    if value is None:
        raise ValueError(f"{name} is None")
    array = np.asarray(value, dtype=float)
    if array.shape != tuple(shape):
        raise ValueError(f"{name} has shape {array.shape}, expected {tuple(shape)}")
    return array


def _describe(err):
    return f"{type(err).__name__}: {err}"


def _fmt(values):
    values = np.atleast_1d(np.asarray(values, dtype=float))
    return "[" + ", ".join(f"{v:.4g}" for v in values) + "]"


@contextlib.contextmanager
def parameters(module, params):
    """Set the pendulum's M1, M2, L1, L2, G on `module` for one case, then
    put back whatever was there."""
    saved = {name: getattr(module, name, None) for name in params}
    for name, value in params.items():
        setattr(module, name, float(value))
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(module, name, value)


def _params_label(params):
    if all(abs(params[n] - DEFAULTS[n]) < 1e-12 for n in PARAM_NAMES):
        return "default pendulum"
    return ", ".join(f"{n}={params[n]:.3g}" for n in PARAM_NAMES)


class Grader:
    """Counts cases and keeps only the failures it is going to print."""

    def __init__(self, name, keep=MAX_FAILURES_SHOWN):
        self.name = name
        self.passed = 0
        self.total = 0
        self.keep = keep
        self.failures = []  # (label, detail), capped at `keep`
        self.stopped = False  # the rest of the cases were failed in one go

    def record(self, label, passed, detail=""):
        self.total += 1
        if passed:
            self.passed += 1
        elif len(self.failures) < self.keep:
            self.failures.append((label, detail))

    def run(self, cases, label, check):
        """Record `check(case, expected)` -> (ok, detail) for every case.

        Anything the student's code raises is a failure of that case, not of
        the run. NotImplementedError (the stub's TODO) fails every remaining
        case with one message, instead of the same line per case.
        """
        cases = iter(cases)
        for i, (case, expected) in enumerate(cases):
            try:
                ok, detail = check(case, expected)
            except NotImplementedError as err:
                self.record(f"case {i} onward", False, f"not implemented yet ({_describe(err)})")
                self.total += sum(1 for _ in cases)
                self.stopped = True
                return self
            except Exception as err:
                ok, detail = False, _describe(err)
            self.record(label(i, case), ok, "" if ok else detail)
        return self

    def report(self, max_failures=MAX_FAILURES_SHOWN):
        """Print the result: a passing run is one line, a failing one lists the
        failures it kept and says how many it didn't."""
        print(f"\n=== {self.name}: {self.passed}/{self.total} ===")

        shown = min(max_failures, len(self.failures))
        for label, detail in self.failures[:shown]:
            line = f"  [FAIL] {label}"
            if detail:
                if len(detail) > MAX_DETAIL_CHARS:
                    detail = detail[:MAX_DETAIL_CHARS] + "..."
                line += f" — {detail}"
            print(line)

        hidden = (self.total - self.passed) - shown
        if hidden > 0 and not self.stopped:
            print(f"  ... and {hidden} more failure(s); pass --verbose to keep more")
        return self.passed, self.total


def _case_label(i, case):
    return f"case {i} ({_params_label(case['params'])})"


# ----------------------------------------------------------------------
# The tests. Each takes the student's double_pendulum module and an iterable
# of (case, expected) pairs, so Gradescope's grade.py runs exactly these on
# its perturbed cases. They take the module, not the functions, because each
# case sets the pendulum's parameters on it.
# ----------------------------------------------------------------------


def grade_energy(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        q = np.array(case["q"], dtype=float)
        qd = np.array(case["qd"], dtype=float)
        with parameters(module, case["params"]):
            energy = _array(module.total_energy(q.copy(), qd.copy()), "energy", (len(q),))
        want = np.asarray(expected["energy"])
        if _close(energy, want):
            return True, ""
        k = int(np.nanargmax(np.abs(energy - want))) if np.isfinite(energy).any() else 0
        return False, (f"at q = {_fmt(q[k])}, qd = {_fmt(qd[k])}: energy {energy[k]:.6g}, "
                       f"expected {want[k]:.6g}")

    def label(i, case):
        return f"{_case_label(i, case)}, {len(case['q'])} states"

    return Grader("total_energy", keep).run(cases, label, check)


def grade_dynamics(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        q = np.array(case["q"], dtype=float)
        qd = np.array(case["qd"], dtype=float)
        with parameters(module, case["params"]):
            qdd = _array(module.dynamics(q.copy(), qd.copy()), "qdd", (2,))
        if _close(qdd, expected["qdd"]):
            return True, ""
        return False, (f"at q = {_fmt(q)}, qd = {_fmt(qd)}: qdd = {_fmt(qdd)}, "
                       f"expected {_fmt(expected['qdd'])}")

    return Grader("dynamics", keep).run(cases, _case_label, check)


def grade_simulate(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        want_t = np.asarray(expected["t"])
        want_q = np.asarray(expected["q"])
        want_qd = np.asarray(expected["qd"])
        n = len(want_t)
        with parameters(module, case["params"]):
            result = module.simulate(np.array(case["q0"], dtype=float),
                                     np.array(case["qd0"], dtype=float),
                                     T=case["T"], dt=case["dt"])
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            return False, "expected simulate to return (t, q, qd)"
        t = _array(result[0], "t", (n,))
        q = _array(result[1], "q", (n, 2))
        qd = _array(result[2], "qd", (n, 2))
        if np.max(np.abs(t - want_t)) > SIM_T_TOL:
            return False, (f"t should be 0, dt, 2 dt, ..., T; got t[:3] = {_fmt(t[:3])}, "
                           f"t[-1] = {t[-1]:.6g}")
        bad = ((np.abs(q - want_q).max(axis=1) > SIM_Q_TOL)
               | (np.abs(qd - want_qd).max(axis=1) > SIM_QD_TOL)
               | ~np.isfinite(q).all(axis=1) | ~np.isfinite(qd).all(axis=1))
        if not bad.any():
            return True, ""
        k = int(np.argmax(bad))
        return False, (f"off from t = {want_t[k]:.2f} s: q = {_fmt(q[k])}, qd = {_fmt(qd[k])}; "
                       f"expected q = {_fmt(want_q[k])}, qd = {_fmt(want_qd[k])}")

    def label(i, case):
        return (f"case {i} (q0 = {_fmt(case['q0'])}, qd0 = {_fmt(case['qd0'])}, "
                f"{_params_label(case['params'])})")

    return Grader("simulate", keep).run(cases, label, check)


_FRAME_PARTS = (("the x axis", 0), ("the y axis", 1), ("the normal", 2), ("the origin", 3))


def _plane_label(i, case):
    corners = np.asarray(case["corners"], dtype=float)
    width = np.linalg.norm(corners[1] - corners[0])
    height = np.linalg.norm(corners[3] - corners[0])
    return f"case {i} ({width * 100:.0f} x {height * 100:.0f} cm sheet)"


def grade_plane_frame(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        corners = np.array(case["corners"], dtype=float)
        H = _array(module.plane_frame(corners.copy()), "H", (4, 4))
        want = np.asarray(expected["H"])
        if not _close(H[3], [0, 0, 0, 1], PLANE_TOL):
            return False, f"the bottom row should be [0, 0, 0, 1], got {_fmt(H[3])}"
        for name, col in _FRAME_PARTS:
            if not _close(H[:3, col], want[:3, col], PLANE_TOL):
                return False, (f"{name} (column {col}) is {_fmt(H[:3, col])}, "
                               f"expected {_fmt(want[:3, col])}")
        return True, ""

    return Grader("plane_frame", keep).run(cases, _plane_label, check)


def grade_paper_to_robot(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        H = np.array(case["H"], dtype=float)
        points = np.array(case["paper_points"], dtype=float)
        got = _array(module.paper_to_robot(H.copy(), points.copy()), "points",
                     points.shape)
        want = np.asarray(expected["robot_points"])
        if _close(got, want, PLANE_TOL):
            return True, ""
        k = int(np.argmax(np.max(np.abs(got - want), axis=1)))
        return False, (f"paper point {_fmt(points[k])} went to {_fmt(got[k])}, "
                       f"expected {_fmt(want[k])}")

    return Grader("paper_to_robot", keep).run(cases, _plane_label, check)


def grade_robot_to_paper(module, cases, keep=MAX_FAILURES_SHOWN):
    def check(case, expected):
        H = np.array(case["H"], dtype=float)
        points = np.array(case["robot_points"], dtype=float)
        got = _array(module.robot_to_paper(H.copy(), points.copy()), "points",
                     points.shape)
        want = np.asarray(expected["paper_points"])
        if _close(got, want, PLANE_TOL):
            return True, ""
        k = int(np.argmax(np.max(np.abs(got - want), axis=1)))
        return False, (f"robot point {_fmt(points[k])} went to {_fmt(got[k])}, "
                       f"expected {_fmt(want[k])}")

    return Grader("robot_to_paper", keep).run(cases, _plane_label, check)


# ----------------------------------------------------------------------
# Local runs: your code, the published fixtures
# ----------------------------------------------------------------------

# name -> (the file it grades, fixture number, test)
TESTS_BY_NAME = {
    "energy": ("double_pendulum.py", 1, grade_energy),
    "dynamics": ("double_pendulum.py", 2, grade_dynamics),
    "simulate": ("double_pendulum.py", 3, grade_simulate),
    "plane_frame": ("plane.py", 4, grade_plane_frame),
    "paper_to_robot": ("plane.py", 4, grade_paper_to_robot),
    "robot_to_paper": ("plane.py", 4, grade_robot_to_paper),
}
# what -p accepts beyond the test names
GROUPS = {"plane": ["plane_frame", "paper_to_robot", "robot_to_paper"]}
FILES = {"double_pendulum.py": ROOT / "double_pendulum.py", "plane.py": LAB / "plane.py"}


def fixture(k):
    """(case, expected) pairs from tests/inK.txt and tests/outK.txt."""
    return zip(_read_jsonl(TESTS / f"in{k}.txt"), _read_jsonl(TESTS / f"out{k}.txt"))


def run_test(name, module, keep=MAX_FAILURES_SHOWN):
    """Grade one test on the published fixture."""
    _, k, grade = TESTS_BY_NAME[name]
    missing = [str(p) for p in (TESTS / f"in{k}.txt", TESTS / f"out{k}.txt") if not p.exists()]
    if missing:
        failed = Grader(name, keep)
        failed.record("fixtures present", False, "missing " + ", ".join(missing))
        return failed
    return grade(module, fixture(k), keep)


def load_file(filename):
    """(module, None), or (None, why it would not import)."""
    path = FILES[filename]
    try:
        return _load_module(path), None
    except Exception as err:
        return None, f"{path.relative_to(ROOT.parent)} doesn't import: {_describe(err)}"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-p", "--problem", type=str, default="all",
                        choices=list(TESTS_BY_NAME) + list(GROUPS) + ["all"])
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="list every failure, not just the first few")
    parser.add_argument("--max-failures", type=int, default=MAX_FAILURES_SHOWN,
                        help="failures to list per test (default: %(default)s)")
    args = parser.parse_args()

    keep = 10 ** 9 if args.verbose else args.max_failures
    selected = (list(TESTS_BY_NAME) if args.problem == "all"
                else GROUPS.get(args.problem, [args.problem]))

    modules = {}
    total_passed = total_count = 0
    for name in selected:
        filename = TESTS_BY_NAME[name][0]
        if filename not in modules:
            modules[filename] = load_file(filename)
        module, error = modules[filename]
        if module is None:
            grader = Grader(name, keep)
            grader.record(f"{filename} imports", False, error)
        else:
            grader = run_test(name, module, keep)
        p, t = grader.report(max_failures=keep)
        total_passed += p
        total_count += t

    print(f"\n=== TOTAL: {total_passed}/{total_count} ===")
    if total_passed == total_count:
        print("Everything matches. Run create_submission.py to package your handin.")
    sys.exit(0 if total_passed == total_count else 1)


if __name__ == "__main__":
    main()
