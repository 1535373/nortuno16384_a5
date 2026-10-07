"""Find the paper: touch the marker to its four corners.  Provided.

    python calibrate.py

With ROBOT_IP set, this is the real arm, and you do the touching. The arm
moves to its ready pose (marker pointing down, over the table in front of the
base), then goes limp in turn for each corner: push the marker tip onto the
corner it asks for, hold it there gently, and press enter. Corner order, as
you stand in front of the robot:

        4 ----------- 3        robot (far side)
        |             |
        1 ----------- 2        you

Without ROBOT_IP, this is the simulator, which has a sheet of paper of its own
(drawn in the meshcat view) and does the touching itself, with your IK.

Either way the four tip positions are saved (calibration-robot.json or
calibration-sim.json), and your `plane_frame` turns them into the paper frame.
The sim run also tells you how far your frame is from the sheet's true one.
"""

import collections
import math
import select
import sys
import termios

import numpy as np

import drawing_arm as da
import plane

CORNER_NAMES = {
    1: "corner 1: nearest you, on your left",
    2: "corner 2: nearest you, on your right",
    3: "corner 3: nearest the robot, on your right",
    4: "corner 4: nearest the robot, on your left",
}


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


def calibrate_sim(model, ik):
    truth = da.sim_paper_frame()
    rng = np.random.default_rng()
    # Where a careful person's touches land: within a millimetre or two of the
    # corners, along the sheet.
    targets = []
    for corner in da.sim_paper_corners():
        slip = rng.normal(0.0, da.SIM_TOUCH_NOISE, 2)
        targets.append(corner + truth[:3, :2] @ slip)
    plan, touches = da.plan_touches(np.array(targets), truth[:3, 2], ik, da.READY_Q)

    arm = da.open_arm(da.READY_Q)
    try:
        scene = da.SimScene(arm, model)
        measured = da.run(arm, plan, model, scene=scene, label="touching the four corners")
        corners = np.array([model.tip(measured[k]) for k in touches])
        H = report(corners, "simulated sheet")
        scene.outline("calibrated", H, 0xE0A030)

        normal_error = math.degrees(math.acos(min(1.0, float(H[:3, 2] @ truth[:3, 2]))))
        origin_error = np.linalg.norm(H[:3, 3] - truth[:3, 3])
        print(f"[calibrate] against the simulated sheet's true frame: origin off by "
              f"{origin_error * 1000:.1f} mm, normal off by {normal_error:.2f} deg")
        if origin_error > 0.005 or normal_error > 1.0:
            print("[calibrate] that is a lot more than the touches' own error -- check plane.py.")
        da.save_calibration(da.calibration_path(False), corners, "simulator")
        print(f"[calibrate] saved {da.calibration_path(False).name}")
    finally:
        arm.close()


def calibrate_real(model, ik):
    arm = da.open_arm()
    try:
        da.hands_clear("moving to the ready pose, marker down over the table")
        if not arm.set_joint_targets(da.READY_Q, speed=da.MOVE_SPEED):
            raise SystemExit("[calibrate] the arm never reached the ready pose")

        # Joints 3 and 5 are watched rather than freed, so the arm stays in
        # the plane its IK works in while you push it around.
        free = [0, 1, 3, 5, 6]
        corners = []
        for i in (1, 2, 3, 4):
            print(f"\n[calibrate] {CORNER_NAMES[i]}.\n"
                  "            Push the marker tip onto it, hold it there gently, "
                  "and press enter.")
            recent = collections.deque(maxlen=20)
            if sys.stdin.isatty():
                termios.tcflush(sys.stdin, termios.TCIFLUSH)

            def on_sample(t, q):
                recent.append(np.array(q, dtype=float))
                return bool(select.select([sys.stdin], [], [], 0)[0])

            run = arm.free_drive(free, duration=math.inf, on_sample=on_sample)
            if run.reason != "stopped by the caller" or not recent:
                raise SystemExit(f"[calibrate] free drive ended ({run.reason}); "
                                 "nothing saved. Run calibrate.py again.")
            sys.stdin.readline()
            tip = np.median([model.tip(q) for q in recent], axis=0)
            corners.append(tip)
            print(f"[calibrate] corner {i} at {np.round(tip, 4)} m")

        corners = np.array(corners)
        H = report(corners, "your sheet")
        da.save_calibration(da.calibration_path(True), corners, "robot")
        print(f"[calibrate] saved {da.calibration_path(True).name}")

        # Lift the marker straight off the last corner.
        up = ik.solve(corners[-1] + 0.05 * H[:3, 2], -H[:3, 2], arm.joint_values)
        da.hands_clear("lifting the marker off the paper")
        arm.set_joint_targets(up, speed=0.2)
    finally:
        arm.stop()
        arm.close()


def main():
    model = da.ArmModel()
    ik = da.PenIK(model, da.load_robot_class())
    if da.is_real():
        calibrate_real(model, ik)
    else:
        calibrate_sim(model, ik)


if __name__ == "__main__":
    main()
