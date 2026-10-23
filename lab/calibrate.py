"""Find the paper: touch the marker to its four corners.  Provided.

    python calibrate.py              # simulator: it touches the corners itself
    python calibrate.py --manual     # simulator: you jog it with the keys below
    ROBOT_IP=... python calibrate.py # the real arm: you move it by hand

Corner order, as you stand in front of the robot:

        4 ----------- 3        robot (far side)
        |             |
        1 ----------- 2        you

On the real arm any order round the sheet works: the corners are put in this
order for you, with corner 1 to 2 along the sheet's long edge.

Every touch is made with the marker upright and the tool turned as it will be
while drawing. That is what makes the result independent of the marker:
wherever its real tip sits on the flange, it sits there the same way at every
corner and at every point of the drawing, so the offset cancels (see the top
of drawing_arm.py). So keep the marker upright when you touch a corner; the
script says how far it leant at each one.

On the real arm (the controller's own safety fence is off while this runs,
and back on when it ends; the library's safety box stays on):
  0. Before the arm moves, the gripper opens. Put the marker in, tip down,
     press enter, hold it until the fingers close on it, and let go.
     (Calibrating again with the marker still in? `--keep-marker` skips this.
     Put a marker in afresh, and calibrate afresh: it won't sit exactly as the
     last one did.)
  1. The arm goes to the home pose of the Assignment 3 lab, and every joint
     is let go. For each corner in turn, bring the marker over the corner,
     down to a centimetre or two above it, and press enter. If the marker
     leaves the area the arm can draw on, the arm locks where it is until you
     say to carry on.
  2. For each corner, the arm puts the marker upright a little over it, and
     lets go again. Put the tip on the corner, keeping the marker upright,
     and press enter. The arm then takes it to the next corner, never lower
     than the paper.

Nothing that goes wrong on the real arm ends the run, short of Ctrl-C. A
fault the controller latches (a bump, an e-stop, abnormal movement in manual
mode) is cleared, after asking you to release the e-stop if that is what it
was; an arm taken past the edge of where it may be is moved back inside on
its own. Each step is saved as it is done (calibration-robot-progress.json),
so a run stopped part-way can be picked up from where it stopped, as long as
the marker hasn't moved in the gripper.

In the simulator's --manual mode, step 2 is done with the keyboard instead:

    w / s   away from you / towards you      a / d   left / right
    r / f   raise / lower                    1 2 3   steps of 0.5, 2, 10 mm
    enter   the tip is on the corner         q       give up

The four tip positions are saved (calibration-robot.json, or
calibration-sim.json), and your `plane_frame` turns them into the paper frame.
"""

import argparse
import collections
import contextlib
import json
import math
import select
import sys
import termios
import tty

import numpy as np
from xarm7_lib import HOME_POSE

import drawing_arm as da
import plane

CORNER_NAMES = {
    1: "corner 1: nearest you, on your left",
    2: "corner 2: nearest you, on your right",
    3: "corner 3: nearest the robot, on your right",
    4: "corner 4: nearest the robot, on your left",
}

# Key -> direction in the robot's frame. You stand on the robot's +x side
# facing it, so away from you is -x and your right is +y.
MOVES = {"w": (-1, 0, 0), "s": (1, 0, 0), "a": (0, -1, 0), "d": (0, 1, 0),
         "r": (0, 0, 1), "f": (0, 0, -1)}
STEPS = {"1": 0.0005, "2": 0.002, "3": 0.010}

# The joints free in the home pose, 0-indexed. Joints 1, 4 and 7 are the
# Assignment 3 lab's, turning about vertical axes, so the marker moves in a
# horizontal plane. But every pose with those three vertical and the marker
# down has the flange 17 cm up, too high for a short marker to reach the
# paper, so in the swing every joint is free: take the marker wherever it has
# to go. The library won't free all seven (it wants one to watch), so joint 7
# is "watched" with a drift allowance it can never reach. Its safety guard
# still stops the arm on a self-collision or the safety box. Joints other than
# 1, 4 and 7 hold up weight; the controller's teaching mode carries it, using
# the payload it is set to. Lab staff: set that to the gripper and marker, or
# the arm sags in your hand.
SWING_FREE = (0, 1, 2, 3, 4, 5)
SWING_TOLERANCE = 100.0  # rad, for joint 7: never reached
# The jog starts this far over where the marker was left over corner 1. It was
# held at some angle then, and the tool offset that calibration cancels in the
# drawing pose doesn't cancel across a tilt: three centimetres covers it.
SWING_ABOVE = 0.03  # m

# Where the marker may be swung in the home pose: around the part of the table
# the drawing pose can reach, with several centimetres' slack, so that only a
# real excursion locks the arm. Past it, the arm locks and goes back.
REACH = (0.22, 0.72)  # m from joint 1's axis
SIDEWAYS = math.radians(40)  # either side of straight ahead
# The arm also locks this much clearance short of the walls and of itself,
# where the library's safety check refuses at 1 cm. The arm then moves itself
# back, and the real arm refuses a move that *starts* from a refused pose:
# locking early is what keeps that move legal. The half centimetre between the
# two is more than a hand moves the arm in one 10 ms sample.
SWING_MARGIN = 0.015  # m
_SWING_GUARD = []
# Where a run that was stopped part-way leaves what it got done.
PROGRESS = da.HERE / "calibration-robot-progress.json"
CONTROLS = __doc__[__doc__.index("    w / s"):__doc__.index("The four tip")].rstrip()


def report(corners, label):
    """Build the frame with your plane.py, check it, and say what it found."""
    H = plane.plane_frame(corners)
    da.check_frame(plane, H, corners)
    width, height = da.paper_size(corners)
    heights = np.asarray(plane.robot_to_paper(H, corners), dtype=float)[:, 2]
    print(f"[calibrate] {label}: sheet {width * 100:.1f} x {height * 100:.1f} cm, "
          f"normal {np.round(H[:3, 2], 3)}, origin {np.round(H[:3, 3], 3)} m")
    print(f"[calibrate] the corners sit {np.round(heights * 1000, 1)} mm off your plane")
    if np.max(np.abs(heights)) > 0.003:
        print("[calibrate] that is more than 3 mm: is the paper flat, and did every "
              "touch land on the corner? Consider running this again.")
    return H


@contextlib.contextmanager
def keys():
    """Single keypresses from the terminal, without waiting for enter."""
    if not sys.stdin.isatty():
        yield lambda: sys.stdin.read(1) or "q"
        return
    saved = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        yield lambda: sys.stdin.read(1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, saved)


def jog(arm, ik, start, status=None):
    """Jog the marker from `start` until enter; returns the computed tip there."""
    point = np.array(start, dtype=float)
    step = STEPS["2"]
    ik.move_to(arm, point)
    with keys() as read:
        while True:
            extra = status(arm) if status is not None else ""
            print(f"\r[jog] step {step * 1000:4.1f} mm  tip {np.round(point, 4)}{extra}   ",
                  end="", flush=True)
            key = read().lower()
            if key in ("\n", "\r"):
                print()
                return ik.model.tip(arm.joint_values)
            if key == "q":
                print()
                raise SystemExit("[calibrate] stopped; nothing saved.")
            if key in STEPS:
                step = STEPS[key]
            elif key in MOVES:
                target = point + step * np.array(MOVES[key], dtype=float)
                try:
                    ik.move_to(arm, target)
                    point = target
                except da.IKError as err:
                    print(f"\n[jog] can't go there: {err}")


def out_of_reach(model, q):
    """Why the marker shouldn't be where `q` puts it, or None."""
    tip = model.tip(q)
    r, heading = math.hypot(tip[0], tip[1]), math.atan2(tip[1], tip[0])
    if not REACH[0] <= r <= REACH[1] or abs(heading) > SIDEWAYS:
        return (f"the marker is {r * 100:.0f} cm out and {math.degrees(heading):+.0f} deg "
                "round from straight ahead, outside where the arm can draw")
    if not _SWING_GUARD:
        from xarm7_lib import SafetyGuard
        _SWING_GUARD.append(SafetyGuard(margin=SWING_MARGIN))
    violation = _SWING_GUARD[0].check(q)
    return None if violation is None else f"the arm is getting close to a limit ({violation})"


SWING_PROMPT = ("swing the marker over it, push it down to a centimetre or two above "
                "it, and press enter.")
TOUCH_PROMPT = ("put the marker's tip on it, keeping the marker upright as it is now, "
                "and press enter.")
TILT_WARNING = 5.0  # deg the marker may lean at a touch before we say so


def swing_over(arm, model, label, prompt=SWING_PROMPT, height=None):
    """Free drive until enter; returns the marker tip then, and the joints.

    Every joint is free (see SWING_FREE). Nothing that goes wrong here ends
    the calibration, short of Ctrl-C. Taking the marker out of reach, or the
    arm into the library's safety guard, locks the arm, and once hands are off
    it goes back to the last place the marker was in reach, by itself, from
    past the guard's edge if need be. A fault the controller latches (a bump,
    an e-stop, "abnormal movement in manual mode") is cleared, and the arm is
    let go again from wherever it stopped.
    """
    last_good = arm.joint_values
    while True:
        print(f"\n[calibrate] {label}: {prompt}")
        recent = collections.deque(maxlen=20)
        locked = []
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)

        def on_sample(t, q):
            nonlocal last_good
            why = out_of_reach(model, q)
            if why is not None:
                locked.append(why)
                return True  # ends the free drive: the arm holds where it is
            last_good = np.array(q, dtype=float)
            recent.append(last_good)
            return bool(select.select([sys.stdin], [], [], 0)[0])

        try:
            reason = arm.free_drive(SWING_FREE, duration=math.inf,
                                    tolerance=SWING_TOLERANCE, on_sample=on_sample).reason
        except da.ARM_ERRORS as err:  # couldn't start, or couldn't stop cleanly
            reason = str(err)
        if reason == "interrupted":
            raise SystemExit("[calibrate] stopped (Ctrl-C). Run it again to pick up "
                             "where you left off.")
        if not locked and reason == "stopped by the caller" and recent:
            sys.stdin.readline()
            return np.median([model.tip(q) for q in recent], axis=0), recent[-1]

        print(f"[calibrate] free drive stopped: {locked[0] if locked else reason}")
        da.clear_faults(arm)
        if locked or out_of_reach(model, arm.joint_values) is not None:
            da.return_to(arm, model, last_good,
                         "putting the arm back where the marker was last in reach", height)
        else:
            print("[calibrate] the arm is still in reach; carrying on from where it is")


def touch_by_hand(arm, ik, model, rough, corners=(), save=None):
    """At each corner: the arm puts the marker upright over it, then you put
    the tip on it in free drive. Returns the four touched tips, and the paper's
    height.

    `corners` are the ones already touched, by a run that was stopped; `save`
    is called with the list after every touch. Moves between corners never
    take the marker lower than it has already been put by hand: the swing's
    lowest point at first, then the paper.
    """
    corners = [np.asarray(c, dtype=float) for c in corners]
    if corners:
        floor, clearance, height = corners[-1][2], 0.01, corners[-1][2] + 0.02
    else:
        floor, clearance, height = min(p[2] for p in rough), 0.0, jog_height(rough)
    for i, near in enumerate(rough, start=1):
        if i <= len(corners):
            continue
        q = ik.solve([near[0], near[1], height], da.PEN_DIR, da.READY_Q)
        low = floor
        da.move_safely(arm, model, q, f"moving the marker upright over {CORNER_NAMES[i]}",
                       height=lambda tip: tip[2] - low, clearance=clearance)
        tip, q_touch = swing_over(arm, model, CORNER_NAMES[i], TOUCH_PROMPT,
                                  height=lambda tip: tip[2] - low)
        axis = model.pen_axis(q_touch)
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(axis @ da.PEN_DIR)))))
        print(f"[calibrate] corner {i} at {np.round(tip, 4)} m, marker {tilt:.1f} deg "
              "from upright")
        if tilt > TILT_WARNING:
            print(f"[calibrate] that is more than {TILT_WARNING:.0f} deg: the touch is off "
                  "by up to a few mm. Consider running this again, marker upright.")
        corners.append(tip)
        if save is not None:
            save(corners)
        # Now the paper's height is known: the next corner starts 2 cm over
        # it, and nothing on the way goes within 1 cm of it.
        floor, clearance = tip[2], 0.01
        height = tip[2] + 0.02
    return np.array(corners), floor


def touch_corners(arm, ik, model, rough, height, status=None):
    """Jog onto each corner, starting over its rough position at `height`."""
    print("\n" + CONTROLS)
    corners = []
    for i, near in enumerate(rough, start=1):
        above = np.array([near[0], near[1], height])
        if corners:  # across the sheet with the marker up, then down to `height`
            ik.move_to(arm, above + [0.0, 0.0, da.PEN_UP])
        print(f"\n[calibrate] {CORNER_NAMES[i]}: jog the tip onto it, and press enter.")
        tip = jog(arm, ik, above, status)
        corners.append(tip)
        print(f"[calibrate] corner {i} at {np.round(tip, 4)} m")
        ik.move_to(arm, tip + [0.0, 0.0, da.PEN_UP])
        height = tip[2] + 0.01  # the next corner starts a centimetre over this one
    return np.array(corners)


def to_drawing_pose(arm, ik, model, point, rough):
    """From wherever the arm is to the drawing pose with the tip at `point`.

    The marker's real length is unknown until the corners are touched, so the
    table's height in these coordinates is too. What is known is where the
    marker was put by hand, over the paper: the move never takes it any lower
    than the lowest of those.
    """
    floor = min(p[2] for p in rough)
    q = ik.solve(point, da.PEN_DIR, da.READY_Q)
    da.move_safely(arm, model, q, "moving to the drawing pose, marker down",
                   height=lambda tip: tip[2] - floor, clearance=0.0)


def jog_height(rough):
    """Where the jog starts over corner 1: a little over where the marker was
    left there by hand."""
    return rough[0][2] + SWING_ABOVE


def anticlockwise(corners):
    """The corners in plane.py's order: anticlockwise seen from above, so the
    diagonals' cross product points up. Touched the other way round (standing
    on another side of the table, say), they are turned round, 1 kept first."""
    corners = np.asarray(corners, dtype=float)
    # Round the sheet anticlockwise seen from above, by each corner's bearing
    # from the middle of the sheet, starting from the first one touched. That
    # puts any of the 24 touch orders right, zigzags included.
    offsets = corners[:, :2] - corners[:, :2].mean(axis=0)
    bearing = np.arctan2(offsets[:, 1], offsets[:, 0])
    order = [int(i) for i in np.argsort((bearing - bearing[0]) % (2 * np.pi))]
    if order != [0, 1, 2, 3]:
        print("[calibrate] the corners weren't touched in order round the sheet "
              f"anticlockwise; taking them as touches {[i + 1 for i in order]}")
        corners = corners[order]
    # Landscape, as the strokes are drawn: x (corner 1 to 2) along the long edge.
    if np.linalg.norm(corners[1] - corners[0]) < np.linalg.norm(corners[3] - corners[0]):
        print("[calibrate] starting from the next corner round, so the sheet's x axis "
              "runs along its long edge")
        corners = corners[[3, 0, 1, 2]]
    return corners


def load_progress():
    """What a stopped run got done: the swung positions and the touched corners."""
    try:
        data = json.loads(PROGRESS.read_text())
        return [np.array(p) for p in data["rough"]], [np.array(c) for c in data["corners"]]
    except (OSError, ValueError, KeyError):
        return [], []


def save_progress(rough, corners):
    PROGRESS.write_text(json.dumps({
        "rough": [np.asarray(p).tolist() for p in rough],
        "corners": [np.asarray(c).tolist() for c in corners]}, indent=1))


def resume(keep_marker):
    """The progress to pick up from, if there is any and the person says so."""
    rough, corners = load_progress()
    if not rough:
        return [], []
    done = (f"{len(corners)} of 4 corners touched" if corners
            else f"the marker swung over {len(rough)} of 4 corners")
    print(f"[calibrate] the last run stopped with {done}.")
    if keep_marker or not sys.stdin.isatty():
        answer = "y" if keep_marker else "n"
    else:
        answer = input("[calibrate] pick up from there? Only if the marker hasn't been "
                       "taken out or moved in the gripper since. [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        return rough, corners
    PROGRESS.unlink(missing_ok=True)
    return [], []


def calibrate_real(model, ik, keep_marker):
    # The fence goes off before anything else: an arm left outside it keeps
    # latching an error until it is. A latched error, or an e-stop still in,
    # is dealt with here too, before anything moves.
    arm = da.open_arm(fence=False)
    try:
        rough, touched = resume(keep_marker)
        # The marker goes in first, wherever the arm is (the home pose, as the
        # last lab left it), before anything moves. Picking up a stopped run
        # needs the same marker, just where it was.
        if keep_marker or rough:
            da.check_marker(arm)
        else:
            da.load_marker(arm)
        if len(rough) < 4:
            da.move_safely(arm, model, np.array(HOME_POSE),
                           "moving to the home pose, as in the Assignment 3 lab")
        for i in range(len(rough) + 1, 5):
            rough.append(swing_over(arm, model, CORNER_NAMES[i])[0])
            save_progress(rough, [])
        corners, paper = touch_by_hand(arm, ik, model, rough, touched,
                                       save=lambda done: save_progress(rough, done))
        last = corners[-1]
        corners = anticlockwise(corners)

        # Saved before the checks, so a frame plane.py gets wrong doesn't cost
        # the touches; draw.py checks the frame again before it moves.
        da.save_calibration(da.calibration_path(True), corners, "robot")
        PROGRESS.unlink(missing_ok=True)
        print(f"[calibrate] saved {da.calibration_path(True).name}")
        report(corners, "your sheet")
        # Up off the last corner, upright, never lower than the paper.
        q = ik.solve(last + [0.0, 0.0, 0.05], da.PEN_DIR, da.READY_Q)
        da.move_safely(arm, model, q, "lifting the marker off the paper",
                       height=lambda tip: tip[2] - paper, clearance=0.0)
    finally:
        da.close_arm(arm, fence=True)


def calibrate_sim(model, ik, manual):
    truth = da.sim_paper_frame()
    offset = da.sim_marker_offset(model)
    arm = da.open_arm(da.READY_Q)
    try:
        scene = da.SimScene(arm, model)
        if manual:
            corners = sim_by_hand(arm, ik, model, scene, truth)
        else:
            corners = sim_automatic(arm, ik, model, scene, truth, offset)
        H = report(corners, "simulated sheet")
        scene.outline("calibrated", H, 0xE0A030)

        # The frame you found is the sheet's, moved by the marker's offset.
        normal_error = math.degrees(math.acos(min(1.0, float(H[:3, 2] @ truth[:3, 2]))))
        origin_error = np.linalg.norm(H[:3, 3] + offset - truth[:3, 3])
        print(f"[calibrate] the simulator's marker tip is {np.round(offset * 1000, 1)} mm "
              f"from where MARKER_LENGTH puts it; allowing for that, your frame's origin "
              f"is {origin_error * 1000:.1f} mm and its normal {normal_error:.2f} deg from "
              "the sheet's")
        if origin_error > 0.005 or normal_error > 1.0:
            print("[calibrate] that is a lot more than the touches' own error -- check plane.py.")
        da.save_calibration(da.calibration_path(False), corners, "simulator",
                            sim_marker_offset=offset)
        print(f"[calibrate] saved {da.calibration_path(False).name}")
    finally:
        arm.close()


def sim_automatic(arm, ik, model, scene, truth, offset):
    """Touch each corner, give or take a careful person's millimetre or two."""
    rng = np.random.default_rng()
    targets = []
    for corner in da.sim_paper_corners():
        slip = rng.normal(0.0, da.SIM_TOUCH_NOISE, 2)
        # Where the *real* tip should land; the computed tip is `offset` away.
        targets.append(corner + truth[:3, :2] @ slip - offset)
    plan, touches = da.plan_touches(np.array(targets), truth[:3, 2], ik, da.READY_Q)
    measured = da.run(arm, plan, model, scene=scene, label="touching the four corners")
    return np.array([model.tip(measured[k]) for k in touches])


def sim_by_hand(arm, ik, model, scene, truth):
    """Practice for the real arm, with the simulator saying how close you are.

    There is nothing to push in the simulator, so the home-pose step is
    stood in for: the arm goes to the home pose, and the "swung" positions
    are the corners give or take a couple of centimetres, with the real tip a
    centimetre or two over the paper. From there it is the real arm's
    procedure: down to the drawing pose, then your jogging.
    """
    da.move_safely(arm, model, np.array(HOME_POSE), "moving to the home pose")
    offset = da.sim_marker_offset(model)
    rng = np.random.default_rng()
    rough = [c + np.append(rng.uniform(-0.02, 0.02, 2), rng.uniform(0.01, 0.02)) - offset
             for c in da.sim_paper_corners()]
    print("[calibrate] (on the real arm you would now swing the marker over each "
          "corner by hand)")

    def status(arm):
        q = arm.joint_values
        scene.update(q, force=True)
        height = (da.sim_marker_tip(model, q) - truth[:3, 3]) @ truth[:3, 2]
        return (f"  real tip {height * 1000:+5.1f} mm off the paper"
                + ("  (touching)" if height <= 0.0005 else ""))

    start = jog_height(rough)
    to_drawing_pose(arm, ik, model, [rough[0][0], rough[0][1], start], rough)
    return touch_corners(arm, ik, model, rough, start, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manual", action="store_true",
                        help="in the simulator, jog to the corners yourself "
                             "(practice for the real arm)")
    parser.add_argument("--keep-marker", action="store_true",
                        help="on the real arm, calibrate with the marker already in the "
                             "gripper instead of opening it to put one in")
    args = parser.parse_args()
    model = da.ArmModel()
    ik = da.PenIK(model, da.load_robot_class())
    if da.is_real():
        calibrate_real(model, ik, args.keep_marker)
    else:
        calibrate_sim(model, ik, args.manual)


if __name__ == "__main__":
    main()
