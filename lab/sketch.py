"""Draw the strokes you want the arm to trace, and save them.  Provided.

    python sketch.py strokes/rkd.json          # draw a new file, or edit that one
    python sketch.py strokes/rkd.json --show   # just look at it

A window shows the sheet as you see it standing in front of the robot: the
robot is beyond the top edge, and corner 1 (the first one you touch when
calibrating) is the bottom-left. Each stroke is a polyline the marker traces
without lifting.

    left click        add a point to the current stroke
    right click/enter finish the stroke (the next click starts a new one)
    backspace / u     take back the last point
    g                 snap to a 5 mm grid, on or off (on to start)
    c                 clear everything
    s                 save (closing the window saves too)

The file holds the strokes in metres on the sheet, measured from corner 1, so
the same file draws in the simulator and on the real arm: `draw.py` puts it on
whichever sheet that arm was calibrated on.

The sheet's size comes from your latest calibration (the real arm's if there
is one, else the simulator's), so what you draw is what fits. Keep inside the
dashed margin; draw.py refuses strokes that leave it.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import drawing_arm as da

GRID = 0.005  # m
CONTROLS = __doc__[__doc__.index("    left click"):__doc__.index("The file holds")].rstrip()


def sheet_size():
    """(width, height, where it came from)."""
    for real in (True, False):
        path = da.calibration_path(real)
        if path.exists():
            corners = np.array(json.loads(path.read_text())["corners"], dtype=float)
            width, height = da.paper_size(corners)
            return width, height, path.name
    width, height = da.SIM_PAPER_SIZE
    return width, height, "the simulator's sheet (no calibration yet)"


def load(path):
    if not path.exists():
        return []
    return [s.tolist() for s in da.load_strokes(path)]


def save(path, strokes, size):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "units": "m",
        "frame": "paper: u from corner 1 towards corner 2, v from corner 1 towards corner 4",
        "paper": [round(size[0], 4), round(size[1], 4)],
        "strokes": [[[round(u, 4), round(v, 4)] for u, v in s] for s in strokes if len(s) > 1],
    }
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"[sketch] saved {len(data['strokes'])} strokes to {path}")


def drawing_time(strokes):
    """Rough seconds the arm will take, from draw.py's speeds."""
    if not strokes:
        return 0.0, 0.0
    pen = sum(np.sum(np.linalg.norm(np.diff(np.array(s), axis=0), axis=1)) for s in strokes)
    gaps = sum(np.linalg.norm(np.subtract(b[0], a[-1])) for a, b in zip(strokes, strokes[1:]))
    seconds = (pen / da.DRAW_SPEED + gaps / da.TRAVEL_SPEED
               + len(strokes) * 2 * da.PEN_UP / da.PLUNGE_SPEED)
    return pen, seconds


class Sketch:
    def __init__(self, path, size, source, editable=True):
        for key in ("keymap.save", "keymap.grid", "keymap.back", "keymap.quit"):
            plt.rcParams[key] = []
        self.path, self.size, self.editable = path, size, editable
        self.strokes = [s for s in load(path) if s]
        self.current = []
        self.snap = True
        self.dirty = False

        width, height = size
        self.fig, self.ax = plt.subplots(figsize=(9, 9 * height / width + 0.6))
        ax = self.ax
        ax.set_aspect("equal")
        pad = 0.01
        ax.set_xlim(-pad, width + pad)
        ax.set_ylim(-pad, height + pad)
        ax.add_patch(plt.Rectangle((0, 0), width, height, fill=True, fc="#fbfaf5",
                                   ec="black", lw=1.5, zorder=0))
        m = da.PAPER_MARGIN
        ax.add_patch(plt.Rectangle((m, m), width - 2 * m, height - 2 * m, fill=False,
                                   ec="#999", ls="--", lw=0.8, zorder=1))
        ax.set_xticks(np.arange(0, width + 1e-9, 0.02))
        ax.set_yticks(np.arange(0, height + 1e-9, 0.02))
        ax.grid(True, color="#e3e3e3", lw=0.6, zorder=0)
        ax.set_xlabel("u [m]  (towards corner 2)")
        ax.set_ylabel("v [m]  (towards corner 4)")
        for (u, v), name in zip([(0, 0), (width, 0), (width, height), (0, height)], "1234"):
            ax.annotate(name, (u, v), textcoords="offset points",
                        xytext=(-12 if u == 0 else 6, -12 if v == 0 else 6), color="#555")
        ax.text(width / 2, height + pad * 0.6, "robot this side", ha="center",
                va="center", fontsize=8, color="#555")
        self.source = source
        self.artists = []
        if editable:
            self.fig.canvas.mpl_connect("button_press_event", self.on_click)
            self.fig.canvas.mpl_connect("key_press_event", self.on_key)
            self.fig.canvas.mpl_connect("close_event", self.on_close)
        self.redraw()

    # -- editing -------------------------------------------------------

    def _point(self, event):
        u, v = event.xdata, event.ydata
        if self.snap:
            u, v = round(u / GRID) * GRID, round(v / GRID) * GRID
        return [float(np.clip(u, 0, self.size[0])), float(np.clip(v, 0, self.size[1]))]

    def finish(self):
        if len(self.current) > 1:
            self.strokes.append(self.current)
        self.current = []

    def on_click(self, event):
        toolbar = getattr(self.fig.canvas, "toolbar", None)
        if event.inaxes is not self.ax or (toolbar is not None and toolbar.mode):
            return
        if event.button == 1:
            self.current.append(self._point(event))
            self.dirty = True
        elif event.button == 3:
            self.finish()
        self.redraw()

    def on_key(self, event):
        if event.key in ("enter", "n"):
            self.finish()
        elif event.key in ("backspace", "u"):
            if not self.current and self.strokes:
                self.current = self.strokes.pop()
            if self.current:
                self.current.pop()
            self.dirty = True
        elif event.key == "g":
            self.snap = not self.snap
        elif event.key == "c":
            self.strokes, self.current, self.dirty = [], [], True
        elif event.key == "s":
            self.finish()
            save(self.path, self.strokes, self.size)
            self.dirty = False
        else:
            return
        self.redraw()

    def on_close(self, _event):
        self.finish()
        if self.dirty:
            save(self.path, self.strokes, self.size)

    # -- drawing -------------------------------------------------------

    def redraw(self):
        for artist in self.artists:
            artist.remove()
        self.artists = []
        ax = self.ax
        for i, stroke in enumerate(self.strokes):
            s = np.array(stroke)
            self.artists += ax.plot(s[:, 0], s[:, 1], "-", color="black", lw=2.5,
                                    solid_capstyle="round", zorder=3)
            self.artists.append(ax.annotate(str(i + 1), s[0], textcoords="offset points",
                                            xytext=(-8, 4), fontsize=7, color="#c33"))
            self.artists += ax.plot(*s[0], "o", color="#c33", ms=3, zorder=4)
        if self.current:
            s = np.array(self.current)
            self.artists += ax.plot(s[:, 0], s[:, 1], "o-", color="#2b59c3", lw=2,
                                    ms=4, zorder=5)
        length, seconds = drawing_time(self.strokes + ([self.current] if len(self.current) > 1 else []))
        title = (f"{self.path.name}: {len(self.strokes)} strokes, {length * 100:.0f} cm of ink, "
                 f"about {seconds:.0f} s on the arm\n"
                 f"sheet {self.size[0] * 100:.1f} x {self.size[1] * 100:.1f} cm from {self.source}")
        if self.editable:
            title += f"   |   snap {'on' if self.snap else 'off'} (g)"
        ax.set_title(title, fontsize=9)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", help="strokes file to create or edit, e.g. strokes/rkd.json")
    parser.add_argument("--show", action="store_true", help="view only, no editing")
    args = parser.parse_args()
    path = Path(args.file)
    if path.suffix != ".json":
        raise SystemExit("give the strokes file a .json name, e.g. strokes/rkd.json")
    width, height, source = sheet_size()
    if args.show and not path.exists():
        raise SystemExit(f"no such file: {path}")
    if not args.show:
        print(CONTROLS)
    sketch = Sketch(path, (width, height), source, editable=not args.show)  
    plt.show()


if __name__ == "__main__":
    main()
