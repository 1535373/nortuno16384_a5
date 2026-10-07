"""The xArm7 as a drawing arm: everything between your code and the robot.

Provided -- you do not need to edit this file, but read the top of it.

The arm is driven as the planar chain your Assignment 4 `Robot` solves. With
joints 3 and 5 held at zero, every joint of the xArm7 sits in one vertical
plane, the one joint 1 turns. Joints 2, 4 and 6 are then three parallel
hinges in that plane: a planar RRR arm standing up, with the marker on the end
of its third link. So to put the marker tip on a point p:

    1. turn joint 1 to face it:      yaw = atan2(p_y, p_x)
    2. in that plane, p is (r, z):   r = sqrt(p_x^2 + p_y^2), z = p_z - shoulder height
    3. your IK solves the RRR for the goal [r, z, phi], where phi is the angle
       that keeps the marker pointing into the paper

`ArmModel` holds that map (measured off UFACTORY's own model of the arm, so it
is the same for the simulator and the real one), and `PenIK` calls your
`Robot.inverse_kinematics` with it. The rest plans a drawing and plays it on
the arm, real or simulated.

Lab staff: the constants in the first block are the ones to check at a station.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Setup. Lab staff: measure these at the stations.
# ----------------------------------------------------------------------

# Flange face to marker tip, along the flange's z axis, in metres.
MARKER_LENGTH = 0.120
# How far above the paper the marker travels between strokes.
PEN_UP = 0.030
# How far *below* the calibrated surface the marker is sent while drawing, so
# it stays in contact over small bumps and calibration error. The marker holder
# has to give at least this much, or the arm pushes the tip into the table.
PRESS = 0.001

DRAW_SPEED = 0.03     # m/s along a stroke
TRAVEL_SPEED = 0.06   # m/s between strokes, pen up
PLUNGE_SPEED = 0.015  # m/s putting the pen down and lifting it
MOVE_SPEED = 0.3      # rad/s for the planned joint move to the start
RATE = 100.0          # Hz the drawing is streamed at

# Your IK is run at points this far apart along the path; the joint angles in
# between are interpolated. Over 4 mm that bends the path by microns.
WAYPOINT_SPACING = 0.004
# How close the marker has to land, by the arm's own kinematics, for an IK
# answer to be used.
IK_TOLERANCE = 0.001   # m
PEN_TOLERANCE = 0.03   # rad of marker tilt
# Your inverse_kinematics runs 100 steps of 0.1 and returns. That was sized for
# the metre-long links of Assignment 4; the xArm's are a third of that, so one
# call gets only part of the way, and it is called again from its own answer
# until it lands, up to this many times.
IK_MAX_CALLS = 60

# Where the arm waits: marker pointing straight down, 15 cm over the table,
# 42 cm in front of the base. Joints 3 and 5 are 0, as everywhere in this lab.
READY_Q = np.array([0.0, -0.0051, 0.0, 0.7604, 0.0, 0.7656, 0.0])

# The sheet the simulator pretends is on the table. It is a US-letter sheet in
# landscape, 42 cm in front of the base, turned 3 degrees and tilted slightly,
# so a calibration that assumes the paper is square to the robot shows it.
SIM_PAPER_SIZE = (0.279, 0.216)
SIM_PAPER_CENTER = (0.42, 0.0, 0.004)
SIM_PAPER_YAW = math.radians(3.0)
SIM_PAPER_TILT = (math.radians(0.8), math.radians(-0.5))  # about x, about y
SIM_TOUCH_NOISE = 0.0015  # m, how far off each simulated corner touch lands

PAPER_MARGIN = 0.010  # m strokes must keep from the paper's edges


def is_real():
    """True when ROBOT_IP is set: the scripts talk to the real arm."""
    return bool(os.environ.get("ROBOT_IP", "").strip())


def calibration_path(real=None):
    real = is_real() if real is None else real
    return HERE / ("calibration-robot.json" if real else "calibration-sim.json")


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ----------------------------------------------------------------------
# The arm's kinematics
# ----------------------------------------------------------------------


class ArmModel:
    """Forward kinematics of the xArm7 with a marker, and the planar map.

    Built on UFACTORY's model of the arm (`xarm7_description`, the same one
    the library's safety guard uses), in the robot's base frame: z up, z = 0 at
    the mounting surface. Everything planar is measured off it here rather than
    written down, as Assignment 4's visualizer did.
    """

    PLANAR_JOINTS = (1, 3, 5)  # 0-indexed: joints 2, 4 and 6

    def __init__(self, marker_length=MARKER_LENGTH):
        import pinocchio as pin
        from robot_descriptions.loaders.pinocchio import load_robot_description

        self._pin = pin
        self.model = load_robot_description("xarm7_description").model
        self.data = self.model.createData()
        self.lower = np.array(self.model.lowerPositionLimit, dtype=float)
        self.upper = np.array(self.model.upperPositionLimit, dtype=float)
        self.marker_length = float(marker_length)
        self._probe()

    # -- forward kinematics --------------------------------------------

    def _frames(self, q):
        self._pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        return [self.data.oMi[i] for i in range(1, 8)]

    def tip(self, q):
        """Marker tip in the base frame, (3,)."""
        flange = self._frames(q)[6]
        return flange.translation + self.marker_length * flange.rotation[:, 2]

    def pen_axis(self, q):
        """Unit vector from the flange towards the marker tip, (3,)."""
        return self._frames(q)[6].rotation[:, 2].copy()

    def tips(self, qs):
        return np.array([self.tip(q) for q in qs])

    # -- the planar chain ----------------------------------------------

    def _in_plane(self, q, point):
        """`point` in the vertical plane joint 1 faces: (r, z above the shoulder)."""
        radial = np.array([math.cos(q[0]), math.sin(q[0]), 0.0])
        return np.array([radial @ point, point[2] - self.shoulder_z])

    def _headings(self, q):
        """World headings, within the arm's plane, of the three links and the pen."""
        frames = self._frames(q)
        tip = frames[6].translation + self.marker_length * frames[6].rotation[:, 2]
        o2, o4, o6, t = (self._in_plane(q, p) for p in
                         (frames[1].translation, frames[3].translation,
                          frames[5].translation, tip))
        axis = frames[6].rotation[:, 2]
        radial = np.array([math.cos(q[0]), math.sin(q[0]), 0.0])
        heading = lambda v: math.atan2(v[1], v[0])  # noqa: E731
        return (np.array([heading(o4 - o2), heading(o6 - o4), heading(t - o6)]),
                heading([radial @ axis, axis[2]]), (o2, o4, o6, t))

    def _probe(self):
        """Link lengths, and theta_k = sign_k * q_joint + offset_k for each link."""
        q = np.zeros(7)
        self.shoulder_z = float(self._frames(q)[1].translation[2])
        headings, pen, (o2, o4, o6, t) = self._headings(q)
        self.link_lengths = np.array([np.linalg.norm(o4 - o2), np.linalg.norm(o6 - o4),
                                      np.linalg.norm(t - o6)])
        thetas = np.array([headings[0], headings[1] - headings[0], headings[2] - headings[1]])
        signs = []
        for k, joint in enumerate(self.PLANAR_JOINTS):
            stepped = q.copy()
            stepped[joint] += 0.1
            h, _, _ = self._headings(stepped)
            moved = np.array([h[0], h[1] - h[0], h[2] - h[1]])
            signs.append(math.copysign(1.0, _wrap(moved[k] - thetas[k])))
        self._signs = np.array(signs)
        self._offsets = thetas.copy()
        # The marker is not along the third link (the flange is offset from
        # joint 6), so the link's heading and the marker's differ by this.
        self.pen_offset = _wrap(pen - headings[2])

    def planar(self, q):
        """(yaw, thetas) for a configuration with joints 3 and 5 at zero."""
        q = np.asarray(q, dtype=float)
        return float(q[0]), self._signs * q[list(self.PLANAR_JOINTS)] + self._offsets

    def joints(self, yaw, thetas, near):
        """The 7 joint values for (yaw, thetas), each the 2*pi-copy nearest `near`."""
        q = np.zeros(7)
        q[0] = yaw
        q[list(self.PLANAR_JOINTS)] = (np.asarray(thetas, dtype=float) - self._offsets) / self._signs
        q[6] = near[6]
        for j in (0,) + self.PLANAR_JOINTS:
            q[j] += 2.0 * math.pi * round((near[j] - q[j]) / (2.0 * math.pi))
        return q

    def planar_goal(self, point, pen_dir):
        """(yaw, [r, z, phi]): where joint 1 turns, and the goal for your IK.

        phi is the third link's heading that points the marker along `pen_dir`,
        as nearly as the plane allows: the marker can only tilt within the
        arm's plane, so the part of `pen_dir` across it is dropped.
        """
        point = np.asarray(point, dtype=float)
        yaw = math.atan2(point[1], point[0])
        radial = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        r, z = radial @ point, point[2] - self.shoulder_z
        pen = math.atan2(pen_dir[2], radial @ pen_dir)
        return yaw, np.array([r, z, _wrap(pen - self.pen_offset)])

    def within_limits(self, q):
        return bool(np.all(q >= self.lower - 1e-9) and np.all(q <= self.upper + 1e-9))


class IKError(RuntimeError):
    """Your IK could not put the marker where it was asked to go."""


class PenIK:
    """Puts the marker tip on a point, using your `Robot.inverse_kinematics`."""

    def __init__(self, model, robot_cls):
        self.model = model
        lengths = np.array(model.link_lengths, dtype=float)
        self.robot = robot_cls(lengths, np.ones(3), np.ones(3), 1.0)
        self.calls = 0
        self._self_test()

    def _self_test(self):
        """Say so now if robot.py is still the stub, rather than mid-plan."""
        _, thetas = self.model.planar(READY_Q)
        ee = np.asarray(self.robot.end_effector(thetas.copy()), dtype=float)
        if ee.shape != (3,) or not np.all(np.isfinite(ee)) or not np.any(ee[:2]):
            raise SystemExit("lab/robot.py: end_effector gives "
                             f"{ee!r} -- is forward_kinematics still the stub? Paste in "
                             "your Assignment 4 Robot.")
        goal = ee + np.array([0.02, -0.02, 0.0])
        answer = np.asarray(self.robot.inverse_kinematics(thetas.copy(), goal), dtype=float)
        if answer.shape == (3,) and np.array_equal(answer, thetas):
            raise SystemExit("lab/robot.py: inverse_kinematics returns its starting angles "
                             "unchanged -- is it still the stub? Paste in your "
                             "Assignment 4 Robot.")

    def solve(self, point, pen_dir, seed_q):
        """Joint values putting the marker tip on `point`, pointing along `pen_dir`.

        Starts your IK from `seed_q`, the previous point's answer, so a path is
        solved as a chain of small steps.
        """
        yaw, goal = self.model.planar_goal(point, pen_dir)
        _, thetas = self.model.planar(seed_q)
        for _ in range(IK_MAX_CALLS):
            answer = self.robot.inverse_kinematics(thetas.copy(), goal.copy())
            self.calls += 1
            answer = np.asarray(answer, dtype=float)
            if answer.shape != (3,) or not np.all(np.isfinite(answer)):
                raise IKError(f"robot.py's inverse_kinematics returned {answer!r}, "
                              "expected 3 finite joint angles")
            thetas = answer
            if self._at_goal(thetas, goal):
                break

        q = self.model.joints(yaw, thetas, near=seed_q)
        off = float(np.linalg.norm(self.model.tip(q) - point))
        if off > IK_TOLERANCE:
            ee = np.asarray(self.robot.end_effector(thetas), dtype=float)
            if np.linalg.norm(ee[:2] - goal[:2]) < IK_TOLERANCE:
                raise IKError(
                    f"robot.py says the marker reached the goal, but the arm's own "
                    f"kinematics put it {off * 1000:.1f} mm away. Check "
                    "forward_kinematics in lab/robot.py.")
            raise IKError(
                f"after {IK_MAX_CALLS} calls robot.py's inverse_kinematics still leaves "
                f"the marker {off * 1000:.1f} mm from {np.round(point, 3)}. The point "
                "may be out of reach, or the IK may not be converging.")
        tilt = math.acos(max(-1.0, min(1.0, float(self.model.pen_axis(q) @ pen_dir))))
        if tilt > PEN_TOLERANCE + self._unreachable_tilt(point, pen_dir):
            raise IKError(f"the marker ends up tilted {math.degrees(tilt):.1f} deg "
                          f"from where it should point at {np.round(point, 3)}")
        if not self.model.within_limits(q):
            raise IKError(f"reaching {np.round(point, 3)} needs joint angles past the "
                          f"arm's limits ({np.round(q, 2)}): it is out of reach")
        return q

    def _at_goal(self, thetas, goal):
        ee = np.asarray(self.robot.end_effector(thetas), dtype=float)
        return (np.linalg.norm(ee[:2] - goal[:2]) < 0.2 * IK_TOLERANCE
                and abs(_wrap(ee[2] - goal[2])) < 0.2 * PEN_TOLERANCE)

    @staticmethod
    def _unreachable_tilt(point, pen_dir):
        """The part of the marker's tilt the arm's plane can't take out."""
        yaw = math.atan2(point[1], point[0])
        across = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return math.asin(min(1.0, abs(float(across @ pen_dir))))


def load_robot_class():
    """Your Robot, from lab/robot.py."""
    sys.path.insert(0, str(HERE))
    try:
        from robot import Robot
    finally:
        sys.path.remove(str(HERE))
    return Robot


# ----------------------------------------------------------------------
# Calibration files
# ----------------------------------------------------------------------


def save_calibration(path, corners, source):
    data = {
        "corners": np.asarray(corners, dtype=float).tolist(),
        "source": source,
        "marker_length": MARKER_LENGTH,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def load_calibration(path=None):
    """The four touched corners, (4, 3). The frame is rebuilt from them with
    your plane.py every time, so a fix to plane.py takes effect at once."""
    path = Path(path) if path else calibration_path()
    if not path.exists():
        where = "the real arm" if "robot" in path.name else "the simulator"
        raise SystemExit(f"no calibration for {where} yet ({path.name}): "
                         "run `python calibrate.py` first.")
    return np.array(json.loads(path.read_text())["corners"], dtype=float)


def paper_size(corners):
    """(width, height) of the sheet: the bottom and left edges' lengths."""
    corners = np.asarray(corners, dtype=float)
    return (float(np.linalg.norm(corners[1] - corners[0])),
            float(np.linalg.norm(corners[3] - corners[0])))


def check_frame(plane, H, corners):
    """Catch a plane.py that would send the arm somewhere strange, before it does.

    Not the grader -- `code/local_autograder.py -p plane` is -- just the
    checks that matter for safety: a proper rotation, a normal pointing up out
    of the table, and corners that land back near where they were touched.
    """
    H = np.asarray(H, dtype=float)
    if H.shape != (4, 4) or not np.all(np.isfinite(H)):
        raise SystemExit(f"plane_frame returned {H!r}; expected a 4x4 transform.")
    R = H[:3, :3]
    if np.allclose(H, np.eye(4)):
        raise SystemExit("plane_frame returned the identity. Is lab/plane.py still the stub?")
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or np.linalg.det(R) < 0:
        raise SystemExit("plane_frame's rotation block is not a rotation: its columns "
                         "should be unit length, perpendicular, and right-handed.")
    if R[2, 2] < 0.5:
        raise SystemExit(f"plane_frame's normal {np.round(R[:, 2], 3)} does not point up "
                         "out of the table. Check the order of the cross product, and "
                         "that the corners were touched in the order plane.py shows.")
    width, height = paper_size(corners)
    expected = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]])
    landed = np.asarray(plane.paper_to_robot(H, expected), dtype=float)
    if landed.shape != (4, 3):
        raise SystemExit(f"paper_to_robot returned shape {landed.shape} for 4 points.")
    worst = np.max(np.linalg.norm(landed - corners, axis=1))
    if worst > 0.01:
        raise SystemExit(f"paper_to_robot puts the paper's corners up to {worst * 1000:.0f} mm "
                         "from where they were touched. Check plane.py.")
    back = np.asarray(plane.robot_to_paper(H, landed), dtype=float)
    if back.shape != (4, 3) or np.max(np.abs(back - expected)) > 1e-6:
        raise SystemExit("robot_to_paper does not undo paper_to_robot. Check plane.py.")


# ----------------------------------------------------------------------
# Strokes
# ----------------------------------------------------------------------


def load_strokes(path):
    """The strokes saved by sketch.py: a list of (n, 2) arrays in paper metres."""
    data = json.loads(Path(path).read_text())
    strokes = [np.asarray(s, dtype=float).reshape(-1, 2) for s in data["strokes"]]
    return [s for s in strokes if len(s)]


def check_strokes(strokes, size):
    width, height = size
    for i, stroke in enumerate(strokes):
        low, high = stroke.min(axis=0), stroke.max(axis=0)
        if (low[0] < PAPER_MARGIN or low[1] < PAPER_MARGIN
                or high[0] > width - PAPER_MARGIN or high[1] > height - PAPER_MARGIN):
            raise SystemExit(
                f"stroke {i} runs off the paper: it spans u {low[0]:.3f}-{high[0]:.3f} m, "
                f"v {low[1]:.3f}-{high[1]:.3f} m, and this sheet is {width:.3f} x "
                f"{height:.3f} m with a {PAPER_MARGIN * 100:.0f} cm margin. Redraw it "
                "with sketch.py, which shows this sheet.")


# ----------------------------------------------------------------------
# Planning
# ----------------------------------------------------------------------


class Plan:
    """A drawing as joint values over time, at RATE Hz."""

    def __init__(self, t, q, tip, pen_down):
        self.t, self.q, self.tip, self.pen_down = t, q, tip, pen_down

    @property
    def duration(self):
        return float(self.t[-1])


def _segment(a, b, speed):
    """Points from a to b (paper coordinates, (3,)) at most WAYPOINT_SPACING apart."""
    length = np.linalg.norm(b - a)
    if length < 1e-9:
        return [], speed
    n = max(1, int(math.ceil(length / WAYPOINT_SPACING)))
    return [a + (b - a) * k / n for k in range(1, n + 1)], speed


def tool_path(strokes):
    """(points, speeds, pen_down) in paper coordinates (u, v, h): every stroke
    with a plunge before it, a lift after it, and pen-up travel between."""
    points, speeds, down = [], [], []

    def add(segment, is_down):
        pts, speed = segment
        points.extend(pts)
        speeds.extend([speed] * len(pts))
        down.extend([is_down] * len(pts))

    first = np.array([*strokes[0][0], PEN_UP])
    points.append(first)
    speeds.append(TRAVEL_SPEED)
    down.append(False)
    for stroke in strokes:
        above = np.array([*stroke[0], PEN_UP])
        add(_segment(points[-1], above, TRAVEL_SPEED), False)
        add(_segment(above, np.array([*stroke[0], -PRESS]), PLUNGE_SPEED), False)
        for a, b in zip(stroke[:-1], stroke[1:]):
            add(_segment(np.array([*a, -PRESS]), np.array([*b, -PRESS]), DRAW_SPEED), True)
        add(_segment(points[-1], np.array([*stroke[-1], PEN_UP]), PLUNGE_SPEED), False)
    return np.array(points), np.array(speeds), np.array(down)


def plan_drawing(strokes, plane, H, ik, seed_q, progress=True):
    """Solve your IK along the whole drawing, before anything moves.

    The path is laid out on the paper, then carried into the robot's frame by
    your paper_to_robot, and the marker points along minus your plane normal.
    """
    paper_pts, speeds, down = tool_path(strokes)
    robot_pts = np.asarray(plane.paper_to_robot(H, paper_pts), dtype=float)
    gaps = np.linalg.norm(np.diff(robot_pts, axis=0), axis=1)
    t_way = np.concatenate([[0.0], np.cumsum(gaps / speeds[1:])])
    return plan_path(robot_pts, t_way, down, -np.asarray(H[:3, 2], dtype=float), ik,
                     seed_q, progress)


def plan_touches(targets, normal, ik, seed_q, dwell=0.8):
    """Down onto each of `targets` (robot frame) and back up, in turn.

    Returns the plan and, for each target, the sample in the middle of the
    dwell on it: the simulator's stand-in for a person holding the marker on a
    corner.
    """
    normal = np.asarray(normal, dtype=float)
    points, times, down = [ik.model.tip(seed_q)], [0.0], [False]

    def go(goal, speed, is_down=False):
        pts, _ = _segment(points[-1], goal, speed)
        for p in pts:
            times.append(times[-1] + np.linalg.norm(p - points[-1]) / speed)
            points.append(p)
            down.append(is_down)

    middles = []
    for target in targets:
        go(target + PEN_UP * normal, TRAVEL_SPEED)
        go(target, PLUNGE_SPEED)
        times.append(times[-1] + dwell)
        points.append(target.copy())
        down.append(True)
        middles.append(times[-1] - dwell / 2)
        go(target + PEN_UP * normal, PLUNGE_SPEED)
    plan = plan_path(np.array(points), np.array(times), np.array(down), -normal, ik,
                     seed_q, progress=False)
    return plan, [int(round(t * RATE)) for t in middles]


def plan_path(robot_pts, t_way, down, pen_dir, ik, seed_q, progress=True):
    """Your IK at every waypoint, then the joints sampled at RATE Hz."""
    q_way = []
    q = np.asarray(seed_q, dtype=float)
    n = len(robot_pts)
    started = time.perf_counter()
    for i, point in enumerate(robot_pts):
        try:
            q = ik.solve(point, pen_dir, q)
        except IKError as err:
            raise SystemExit(f"\nplanning stopped at waypoint {i} of {n} "
                             f"(marker at {np.round(point, 3)} m): {err}")
        if q_way and np.max(np.abs(q - q_way[-1])) > 0.1:
            raise SystemExit(f"\nplanning stopped at waypoint {i}: the arm would jump "
                             f"{np.max(np.abs(q - q_way[-1])):.2f} rad between two points "
                             f"{WAYPOINT_SPACING * 1000:.0f} mm apart. Your IK may have "
                             "flipped to another solution.")
        q_way.append(q)
        if progress and (i % 25 == 0 or i == n - 1):
            print(f"\r[plan] solving your IK along the path: {i + 1}/{n} "
                  f"({time.perf_counter() - started:.0f} s)", end="", flush=True)
    if progress:
        print()
    q_way = np.array(q_way)

    t = np.arange(0.0, t_way[-1] + 0.5 / RATE, 1.0 / RATE)
    q = np.column_stack([np.interp(t, t_way, q_way[:, j]) for j in range(7)])
    tip = np.column_stack([np.interp(t, t_way, robot_pts[:, j]) for j in range(3)])
    pen = np.interp(t, t_way, np.asarray(down, dtype=float)) > 0.5
    return Plan(t, q, tip, pen)


def check_plan(plan, every=5):
    """The library's own safety check (self-collision, safety box) on the plan,
    before it runs. The real arm would refuse these setpoints mid-drawing."""
    from xarm7_lib import SafetyGuard

    guard = SafetyGuard()
    for k in list(range(0, len(plan.q), every)) + [len(plan.q) - 1]:
        violation = guard.check(plan.q[k])
        if violation is not None:
            raise SystemExit(f"the drawing would take the arm somewhere it may not go, "
                             f"{plan.t[k]:.1f} s in: {violation}")


def joint_move_clear(model, plane, H, q_from, q_to, clearance=0.01):
    """Whether the marker stays above the paper on a straight joint move."""
    for s in np.linspace(0.0, 1.0, 40):
        q = (1.0 - s) * q_from + s * q_to
        h = np.asarray(plane.robot_to_paper(H, model.tip(q)[None]), dtype=float)[0, 2]
        if h < clearance:
            return False
    return True


# ----------------------------------------------------------------------
# The arm
# ----------------------------------------------------------------------


def open_arm(start_q=READY_Q):
    """The real arm when ROBOT_IP is set, the simulator (in meshcat) otherwise."""
    if is_real():
        from xarm7_lib import RealXArm7
        print(f"[arm] connecting to the xArm7 at {os.environ['ROBOT_IP'].strip()}")
        return RealXArm7(os.environ["ROBOT_IP"].strip())
    from xarm7_lib import SimulatedXArm7
    arm = SimulatedXArm7(visualize=True, q0=start_q)
    if arm.viz is not None:
        print(f"[arm] simulator: {arm.viz.url}")
    return arm


def hands_clear(message="the arm is about to move"):
    """Wait for the person at the arm to say it is safe. The simulator doesn't ask."""
    if not is_real():
        return
    input(f"[arm] {message}. Stand clear, hand on the e-stop, and press enter: ")


def stream(arm, plan, on_tick=None):
    """Play a plan already started at plan.q[0]; returns the measured joints."""
    measured = np.zeros_like(plan.q)
    measured[0] = arm.joint_values
    period = 1.0 / RATE
    next_tick = time.perf_counter()
    for k in range(1, len(plan.q)):
        arm.servo_joints(plan.q[k])
        measured[k] = arm.joint_values
        if on_tick is not None:
            on_tick(k, measured[k])
        next_tick += period
        time.sleep(max(0.0, next_tick - time.perf_counter()))
    return measured


def run(arm, plan, model, plane=None, H=None, scene=None, label="drawing"):
    """Move to the plan's start, then stream it. Returns the measured joints.

    With a paper frame, the move to the start is refused if it would bring the
    marker near the paper on the way.
    """
    if H is not None and not joint_move_clear(model, plane, H, arm.joint_values, plan.q[0]):
        raise SystemExit(
            "the move to the start of the drawing would bring the marker within 1 cm "
            "of the paper on the way. Move the arm up and over the sheet first "
            "(`python -m xarm7_lib.free_drive`), then run this again.")
    hands_clear(f"moving to the start of the {label}")
    if not arm.set_joint_targets(plan.q[0], speed=MOVE_SPEED):
        raise SystemExit("the arm never reached the start of the drawing")
    print(f"[arm] {label}: {plan.duration:.0f} s")

    def on_tick(k, q):
        if scene is not None:
            scene.update(q)
        if k % int(RATE) == 0:
            print(f"\r[arm] {plan.t[k]:5.1f} / {plan.duration:.1f} s", end="", flush=True)

    try:
        measured = stream(arm, plan, on_tick)
    finally:
        print()
        arm.stop()
    return measured


# ----------------------------------------------------------------------
# The simulator's sheet of paper
# ----------------------------------------------------------------------


def _rotation(axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def sim_paper_frame():
    """The simulated sheet's true frame, in the base frame, (4, 4).

    Laid out like the real one: x along the near edge to your right as you
    face the robot (the robot's +y), y away from you (the robot's -x).
    """
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    R = (_rotation("z", SIM_PAPER_YAW) @ _rotation("x", SIM_PAPER_TILT[0])
         @ _rotation("y", SIM_PAPER_TILT[1]) @ R)
    width, height = SIM_PAPER_SIZE
    H = np.eye(4)
    H[:3, :3] = R
    H[:3, 3] = np.array(SIM_PAPER_CENTER) - R @ [width / 2, height / 2, 0.0]
    return H


def sim_paper_corners():
    width, height = SIM_PAPER_SIZE
    H = sim_paper_frame()
    local = np.array([[0, 0, 0], [width, 0, 0], [width, height, 0], [0, height, 0]], float)
    return local @ H[:3, :3].T + H[:3, 3]


class SimScene:
    """The paper, the marker and the ink, drawn into the simulator's meshcat view.

    Ink is wherever the marker tip is at or below the *true* simulated sheet,
    so a calibration that is off shows up as ink that skips or a pen that
    dives: the drawing is planned on your calibrated sheet, and inked on the
    real one.
    """

    INK_HEIGHT = 0.0005  # m above the sheet that still leaves a mark

    def __init__(self, arm, model, H_cal=None):
        import meshcat.geometry as g

        self.g, self.arm, self.model = g, arm, model
        self.viz = getattr(arm, "viz", None)
        frame = getattr(arm, "_base_frame", None)
        self.world = frame() if frame is not None else np.eye(4)
        self.H = sim_paper_frame()
        self.ink, self.strokes, self._last_draw = [], 0, 0.0
        if self.viz is None:
            return
        width, height = SIM_PAPER_SIZE
        center = self.H.copy()
        center[:3, 3] = self.H[:3, :3] @ [width / 2, height / 2, -0.0006] + self.H[:3, 3]
        with self.viz.lock:
            node = self.viz.viewer["lab/paper"]
            node.set_object(g.Box([width, height, 0.001]),
                            g.MeshLambertMaterial(color=0xF7F5EE))
            node.set_transform(self.world @ center)
            self.viz.viewer["lab/marker"].set_object(
                g.Cylinder(self.model.marker_length, 0.007),
                g.MeshLambertMaterial(color=0x2B59C3))
        if H_cal is not None:
            self.outline("calibrated", H_cal, 0xE0A030)
        self.update(arm.joint_values, force=True)

    def _line(self, path, points, color, width=2.0):
        pts = (np.asarray(points, float) @ self.world[:3, :3].T + self.world[:3, 3])
        with self.viz.lock:
            self.viz.viewer[path].set_object(self.g.Line(
                self.g.PointsGeometry(pts.T.astype(np.float32)),
                self.g.LineBasicMaterial(color=color, linewidth=width)))

    def outline(self, name, H, color):
        """A sheet-sized rectangle on the frame H: where calibration thinks the paper is."""
        if self.viz is None:
            return
        width, height = SIM_PAPER_SIZE
        local = np.array([[0, 0, 0.001], [width, 0, 0.001], [width, height, 0.001],
                          [0, height, 0.001], [0, 0, 0.001]])
        self._line(f"lab/{name}", local @ H[:3, :3].T + H[:3, 3], color)

    def planned(self, plan):
        """The plan's pen-down stretches, faint, on the sheet."""
        if self.viz is None:
            return
        pieces, current = [], []
        for p, down in zip(plan.tip, plan.pen_down):
            if down:
                current.append(p)
            elif current:
                pieces.append(current)
                current = []
        if current:
            pieces.append(current)
        with self.viz.lock:
            self.viz.viewer["lab/plan"].delete()
        for i, piece in enumerate(pieces):
            if len(piece) > 1:
                self._line(f"lab/plan/{i}", np.array(piece) + [0, 0, 0.0015], 0xB0B8C8)

    def update(self, q, force=False):
        tip = self.model.tip(q)
        local = (tip - self.H[:3, 3]) @ self.H[:3, :3]
        if local[2] <= self.INK_HEIGHT:
            if not self.ink or self.ink[-1] is None:
                self.ink.append([])
            self.ink[-1].append(tip.copy())
        elif self.ink and self.ink[-1] is not None:
            self.ink.append(None)  # pen lifted: the next mark starts a new stroke
        now = time.perf_counter()
        if self.viz is None or (not force and now - self._last_draw < 0.08):
            return
        self._last_draw = now
        axis = self.model.pen_axis(q)
        T = np.eye(4)
        T[:3, :3] = self._frame_for(axis)
        T[:3, 3] = tip - 0.5 * self.model.marker_length * axis
        with self.viz.lock:
            self.viz.viewer["lab/marker"].set_transform(self.world @ T)
        marks = [m for m in self.ink if m]
        for i, mark in enumerate(marks):
            if len(mark) > 1 and (i == len(marks) - 1 or i >= self.strokes):
                pts = np.array(mark)
                pts = pts - np.outer((pts - self.H[:3, 3]) @ self.H[:3, 2] - 0.001, self.H[:3, 2])
                self._line(f"lab/ink/{i}", pts, 0x111111, 3.0)
        self.strokes = len(marks) - 1

    @staticmethod
    def _frame_for(axis):
        """A rotation taking three.js's cylinder axis (+y) onto `axis`."""
        y = axis / np.linalg.norm(axis)
        helper = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(helper, y)
        x /= np.linalg.norm(x)
        z = np.cross(x, y)
        return np.column_stack([x, y, z])
