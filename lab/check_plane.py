"""Check your plane.py before you take it to the robot.  Provided.

    python check_plane.py

Three rounds, each saying what is wrong when something is:

1. Sheets with a known answer. Perfectly flat sheets, laid square to the robot,
   turned, tilted, and stood up like a whiteboard on a wall, whose frame is
   known exactly. Your frame has to match it.
2. Touches that miss. Real touches don't lie on one plane, and then the frame
   is whatever the formula in plane.py says. Your frame is checked against
   everything that formula promises: a unit normal perpendicular to both
   diagonals, a plane through the centroid, an origin straight above or below
   corner 1, an x axis in the plane and towards corner 2, a right-handed frame.
3. The graded cases: code/tests/in4.txt, exactly as
   `python local_autograder.py -p plane` runs them from code/.

Then, if you have run calibrate.py in the simulator, it rebuilds that frame
and compares it with the simulated sheet's true one.

It needs nothing but numpy, and moves no arm. If all of it passes, your
plane.py is ready for `python calibrate.py` on the real arm.
"""

import importlib.util
import math
import sys
import traceback

import numpy as np

import drawing_arm as da

CODE = da.HERE.parent / "code"
TOL = 1e-6


class Results:
    def __init__(self):
        self.failed = 0
        self.passed = 0

    def check(self, name, fn):
        try:
            problem = fn()
        except Exception:
            problem = "raised an exception:\n" + "\n".join(
                "        " + line for line in traceback.format_exc().strip().splitlines()[-6:])
        if problem:
            self.failed += 1
            print(f"  [FAIL] {name}\n         {problem}")
        else:
            self.passed += 1
            print(f"  [ OK ] {name}")


def fmt(v):
    return "[" + ", ".join(f"{x:.4g}" for x in np.ravel(v)) + "]"


def load_plane():
    spec = importlib.util.spec_from_file_location("plane", da.HERE / "plane.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def frame_of(plane, corners):
    H = np.asarray(plane.plane_frame(np.array(corners, dtype=float)), dtype=float)
    if H.shape != (4, 4):
        raise ValueError(f"plane_frame returned shape {H.shape}, expected (4, 4)")
    if np.allclose(H, np.eye(4)):
        raise ValueError("plane_frame returned the identity (still the stub?)")
    return H


# ----------------------------------------------------------------------
# Sheets whose frame is known
# ----------------------------------------------------------------------


def rotation(x_axis, normal):
    x = np.asarray(x_axis, float) / np.linalg.norm(x_axis)
    n = np.asarray(normal, float) / np.linalg.norm(normal)
    return np.column_stack([x, np.cross(n, x), n])


def sheet(width, height, origin, x_axis, normal):
    """The four corners of an exactly flat sheet, and its true frame."""
    R = rotation(x_axis, normal)
    H = np.eye(4)
    H[:3, :3], H[:3, 3] = R, origin
    local = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], float)
    return local @ R.T + origin, H


def tilt(angle_deg, about):
    a = math.radians(angle_deg)
    about = np.asarray(about, float) / np.linalg.norm(about)
    K = np.array([[0, -about[2], about[1]], [about[2], 0, -about[0]], [-about[1], about[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K


KNOWN = [
    ("letter sheet on the table, square to the robot",
     sheet(0.279, 0.216, [0.528, -0.1395, 0.0], [0, 1, 0], [0, 0, 1])),
    ("A4 sheet, raised 5 cm and turned 30 degrees",
     sheet(0.297, 0.210, [0.5, -0.1, 0.05], tilt(30, [0, 0, 1]) @ [0, 1, 0], [0, 0, 1])),
    ("sheet tilted 20 degrees towards the robot",
     sheet(0.25, 0.2, [0.5, -0.12, 0.02], [0, 1, 0], tilt(20, [0, 1, 0]) @ [0, 0, 1])),
    ("whiteboard standing up, facing the robot",
     sheet(0.4, 0.3, [0.6, 0.2, 0.1], [0, -1, 0], [-1, 0, 0])),
]


def round_one(plane, results):
    print("\n1. sheets with a known answer")
    for name, (corners, truth) in KNOWN:
        def run(corners=corners, truth=truth):
            H = frame_of(plane, corners)
            for label, col in (("the normal (column 2)", 2), ("the x axis (column 0)", 0),
                               ("the y axis (column 1)", 1), ("the origin (column 3)", 3)):
                if not np.allclose(H[:3, col], truth[:3, col], atol=TOL):
                    hint = ""
                    if col == 2 and np.allclose(H[:3, 2], -truth[:3, 2], atol=TOL):
                        hint = " -- the right line, pointing the wrong way: check the cross product's order"
                    elif col == 3 and np.allclose(H[:3, 3], corners[0], atol=TOL):
                        hint = ""
                    return f"{label} is {fmt(H[:3, col])}, expected {fmt(truth[:3, col])}{hint}"
            if not np.allclose(H[3], [0, 0, 0, 1]):
                return f"the bottom row is {fmt(H[3])}, expected [0, 0, 0, 1]"
            return None
        results.check(name, run)


# ----------------------------------------------------------------------
# Touches that miss: the formula's promises
# ----------------------------------------------------------------------


def promises(corners, H):
    """What's wrong with H as the formula's frame for these corners, or None."""
    p1, p2, p3, p4 = corners
    x, y, n, o = H[:3, 0], H[:3, 1], H[:3, 2], H[:3, 3]
    c = corners.mean(axis=0)
    if abs(np.linalg.norm(n) - 1) > TOL:
        return f"the normal has length {np.linalg.norm(n):.6f}, not 1"
    for name, d in (("p3 - p1", p3 - p1), ("p4 - p2", p4 - p2)):
        if abs(n @ d) > TOL:
            return (f"the normal is not perpendicular to the diagonal {name} "
                    f"(dot product {n @ d:.3g}): use the cross product of the diagonals")
    if n @ np.cross(p3 - p1, p4 - p2) < 0:
        return "the normal points the wrong way: (p3 - p1) x (p4 - p2), in that order"
    if abs((o - c) @ n) > TOL:
        return (f"the origin is {(o - c) @ n * 1000:.3f} mm off the plane through the "
                "centroid of the four corners")
    if np.linalg.norm(np.cross(o - p1, n)) > TOL:
        return "the origin should be corner 1 moved along the normal onto the plane"
    if abs(np.linalg.norm(x) - 1) > TOL or abs(x @ n) > TOL:
        return "the x axis should be a unit vector in the plane (perpendicular to the normal)"
    if np.linalg.norm(np.cross(x, (p2 - p1) - ((p2 - p1) @ n) * n)) > TOL or x @ (p2 - p1) < 0:
        return "the x axis should point from corner 1 towards corner 2, within the plane"
    if not np.allclose(y, np.cross(n, x), atol=TOL):
        return f"the y axis is {fmt(y)}, expected n x x_hat = {fmt(np.cross(n, x))}"
    return None


def round_two(plane, results):
    print("\n2. touches that miss the plane by a few millimetres")
    rng = np.random.default_rng(5)
    for i, (name, (corners, _)) in enumerate(KNOWN):
        for trial in range(3):
            touched = corners + rng.uniform(-0.003, 0.003, (4, 3))
            label = f"{name}, uneven touches #{trial + 1}"
            results.check(label, lambda t=touched: promises(t, frame_of(plane, t)))
            if trial == 0:
                original = touched.copy()
                results.check(f"{name}: plane_frame leaves its input alone",
                              lambda t=touched, o=original: None if (
                                  frame_of(plane, t) is not None and np.array_equal(t, o))
                              else "plane_frame changed the corners it was given")


def round_trip(plane, results):
    print("\n   carrying points between the paper and the robot")
    corners, H = KNOWN[2][1]
    paper = np.array([[0.0, 0.0, 0.0], [0.1, 0.05, 0.0], [0.2, 0.15, 0.03], [0.05, 0.1, -0.001]])

    def forward():
        got = np.asarray(plane.paper_to_robot(H, paper), dtype=float)
        if got.shape != paper.shape:
            return f"paper_to_robot returned shape {got.shape} for {paper.shape[0]} points"
        want = paper @ H[:3, :3].T + H[:3, 3]
        k = int(np.argmax(np.max(np.abs(got - want), axis=1)))
        if not np.allclose(got, want, atol=TOL):
            return f"paper point {fmt(paper[k])} went to {fmt(got[k])}, expected {fmt(want[k])}"
        return None

    def backward():
        robot = paper @ H[:3, :3].T + H[:3, 3]
        got = np.asarray(plane.robot_to_paper(H, robot), dtype=float)
        if got.shape != robot.shape:
            return f"robot_to_paper returned shape {got.shape} for {robot.shape[0]} points"
        k = int(np.argmax(np.max(np.abs(got - paper), axis=1)))
        if not np.allclose(got, paper, atol=TOL):
            return f"robot point {fmt(robot[k])} came back as {fmt(got[k])}, expected {fmt(paper[k])}"
        return None

    def one_point():
        got = np.asarray(plane.paper_to_robot(H, paper[1:2]), dtype=float)
        return None if got.shape == (1, 3) else f"one point in, shape {got.shape} out; expected (1, 3)"

    results.check("paper_to_robot", forward)
    results.check("robot_to_paper", backward)
    results.check("a single point, as shape (1, 3)", one_point)


def round_three(results):
    print("\n3. the graded cases (code/tests/in4.txt)")
    sys.path.insert(0, str(CODE))
    try:
        import local_autograder as grader
    finally:
        sys.path.remove(str(CODE))
    module, error = grader.load_file("plane.py")
    for name in grader.GROUPS["plane"]:
        if module is None:
            results.check(name, lambda: error)
            continue
        g = grader.run_test(name, module, keep=3)
        detail = "; ".join(f"{label}: {why}" for label, why in g.failures)
        results.check(f"{name}: {g.passed}/{g.total} cases",
                      lambda g=g, d=detail: None if g.passed == g.total else d)


def against_the_sim(plane):
    path = da.calibration_path(False)
    if not path.exists():
        print("\n(no calibration-sim.json yet: run `python calibrate.py` to try your "
              "frame on the simulator's sheet)")
        return
    corners = da.load_calibration(path)
    try:
        H = frame_of(plane, corners)
    except Exception as err:
        print(f"\nyour calibration-sim.json: plane_frame fails on it ({err})")
        return
    truth = da.sim_paper_frame()
    angle = math.degrees(math.acos(min(1.0, abs(float(H[:3, 2] @ truth[:3, 2])))))
    print(f"\nyour calibration-sim.json, through this plane.py: origin "
          f"{np.linalg.norm(H[:3, 3] - truth[:3, 3]) * 1000:.1f} mm and normal "
          f"{angle:.2f} deg from the simulated sheet's (the touches themselves are "
          f"off by about {da.SIM_TOUCH_NOISE * 1000:.1f} mm)")


def main():
    try:
        plane = load_plane()
    except Exception as err:
        raise SystemExit(f"lab/plane.py doesn't import: {type(err).__name__}: {err}")
    results = Results()
    round_one(plane, results)
    round_two(plane, results)
    round_trip(plane, results)
    round_three(results)
    against_the_sim(plane)

    print(f"\n{results.passed} passed, {results.failed} failed")
    if results.failed:
        print("Fix plane.py before calibrating on the robot: a wrong frame sends the "
              "marker into the table, or draws off the paper.")
        sys.exit(1)
    print("plane.py is ready. Next: `python calibrate.py` and `python draw.py ...` in the "
          "simulator, then on the robot.")


if __name__ == "__main__":
    main()
