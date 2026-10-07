"""Draw your strokes on the paper with the xArm7.  Provided.

    python draw.py strokes/kdc.json              # simulator, or the real arm with ROBOT_IP set
    python draw.py strokes/kdc.json --plan-only  # plan and plot it; nothing moves

1. Your plane.py builds the paper frame from the latest calibration
   (calibration-robot.json or calibration-sim.json; run calibrate.py first).
2. The strokes are laid out on that sheet, with the pen lifted PEN_UP between
   them, and your robot.py's inverse kinematics is solved along the whole path
   before anything moves. A point it can't reach stops the run here.
3. The arm moves to the start and traces the path, streaming joint setpoints at
   100 Hz.
4. The result is saved to results/: a plot of the strokes you planned next to
   the path the marker tip really took (from the arm's measured joint angles),
   and the raw log.

In the simulator the sheet is drawn in the meshcat view, and ink appears
wherever the marker meets it.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import drawing_arm as da
import plane

RESULTS = da.HERE / "results"


def plot_result(name, strokes, size, plan, trace, inked, out):
    """The planned strokes and the marker's measured path, on the sheet."""
    width, height = size
    fig, ax = plt.subplots(figsize=(9, 9 * height / width + 0.6))
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((0, 0), width, height, fc="#fbfaf5", ec="black", lw=1.5))
    for i, stroke in enumerate(strokes):
        ax.plot(stroke[:, 0], stroke[:, 1], "--", color="#8a94a6", lw=1.2,
                label="planned" if i == 0 else None)
    # The measured path, solid where the pen was down.
    pieces, current = [], []
    for point, down in zip(trace, inked):
        if down:
            current.append(point)
        elif current:
            pieces.append(np.array(current))
            current = []
    if current:
        pieces.append(np.array(current))
    for i, piece in enumerate(pieces):
        ax.plot(piece[:, 0], piece[:, 1], "-", color="#111", lw=2,
                label="marker (measured)" if i == 0 else None)
    up = np.where(inked, np.nan, 1.0)
    ax.plot(trace[:, 0] * up, trace[:, 1] * up, ":", color="#c33", lw=0.8,
            label="marker, lifted")
    ax.set_xlim(-0.01, width + 0.01)
    ax.set_ylim(-0.01, height + 0.01)
    ax.set_xlabel("u [m]")
    ax.set_ylabel("v [m]")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title(name, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("strokes", help="the strokes file sketch.py saved")
    parser.add_argument("--plan-only", action="store_true",
                        help="plan the drawing and plot it, without moving the arm")
    parser.add_argument("--no-show", action="store_true", help="save the plot, don't open it")
    args = parser.parse_args()

    real = da.is_real()
    where = "robot" if real else "sim"
    corners = da.load_calibration()
    H = plane.plane_frame(corners)
    da.check_frame(plane, H, corners)
    size = da.paper_size(corners)
    strokes = da.load_strokes(args.strokes)
    if not strokes:
        raise SystemExit(f"{args.strokes} has no strokes in it")
    da.check_strokes(strokes, size)

    model = da.ArmModel()
    ik = da.PenIK(model, da.load_robot_class())
    started = time.perf_counter()
    plan = da.plan_drawing(strokes, plane, H, ik, da.READY_Q)
    print(f"[plan] {len(strokes)} strokes, {plan.duration:.0f} s on the arm "
          f"(your IK was called {ik.calls} times in {time.perf_counter() - started:.0f} s)")
    da.check_plan(plan)

    RESULTS.mkdir(exist_ok=True)
    stem = Path(args.strokes).stem
    if args.plan_only:
        trace = np.asarray(plane.robot_to_paper(H, plan.tip), dtype=float)
        out = RESULTS / f"{stem}-plan.png"
        plot_result(f"{stem}: the plan, on the calibrated sheet", strokes, size, plan,
                    trace[:, :2], plan.pen_down, out)
        print(f"[draw] saved {out}")
        if not args.no_show:
            plt.show()
        return

    arm = da.open_arm(da.READY_Q)
    try:
        scene = None
        if not real:
            scene = da.SimScene(arm, model, H_cal=H)
            scene.planned(plan)
        measured = da.run(arm, plan, model, plane, H, scene=scene, label=f"drawing {stem}")
    finally:
        arm.close()

    tips = model.tips(measured)
    if real:
        # Where the tip went on the calibrated sheet; "inked" where the plan had
        # the pen down, since nothing here can see the paper.
        trace = np.asarray(plane.robot_to_paper(H, tips), dtype=float)
        inked = plan.pen_down
    else:
        # On the simulated sheet's true frame, and inked only where the tip
        # really reached it: a calibration that is off shows here.
        truth = da.sim_paper_frame()
        trace = (tips - truth[:3, 3]) @ truth[:3, :3]
        inked = trace[:, 2] <= da.SimScene.INK_HEIGHT

    off = np.linalg.norm(tips - plan.tip, axis=1)
    print(f"[draw] the marker tip followed the plan to within {np.median(off) * 1000:.1f} mm "
          f"(median), {off.max() * 1000:.1f} mm (worst)")
    out = RESULTS / f"{stem}-{where}.png"
    plot_result(f"{stem} on the {'real arm' if real else 'simulator'}", strokes, size, plan,
                trace[:, :2], inked, out)
    np.savez(RESULTS / f"{stem}-{where}.npz", t=plan.t, q_planned=plan.q, q_measured=measured,
             tip_planned=plan.tip, tip_measured=tips, pen_down=plan.pen_down, corners=corners)
    print(f"[draw] saved {out} and {out.with_suffix('.npz').name}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
