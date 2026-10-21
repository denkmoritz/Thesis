"""
Step 3 of the zone rebuild: rectified pairs of the real scenes.

Both frames are rectified with the step-2 calibration (no fitting to labels).
After rectification the same insect must lie on the same image row in DVS and
RGB, whatever its depth. For each DVS point a horizontal line is drawn across
both images; the RGB label should sit on it.

Printed per scene, on frames with exactly one DVS insect and one RGB label
(no matching ambiguity):
  - row offset  = row(RGB label) - row(DVS point), in RGB px (rectified px x 1.58)
  - disparity   = x(DVS) - x(RGB) in rectified px; must be > 0 for a point in
                  front of the cameras, depth Z = f * B / d with B = 60 mm
  - timing: row offset for DVS frame f+k vs RGB frame f, k = -2..+1, on all
    pairs and on hand-placed keyframes where the insect moves > 3 DVS px/frame
  - row offset inside / outside the area the calibration board covered in RGB

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/rectify.py
"""

import csv
import os
from collections import defaultdict

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, load_rgb, pick_frames, read_frame, center
from dti.geometry import calibration as s2

OUT_DIR = "outputs/step3"
BOARD_CACHE = "outputs/step2/board_points_rgb.npy"
SHIFTS = [-2, -1, 0, 1]
BASELINE_MM = 60.0
WHITE, GREEN = (255, 255, 255), (0, 255, 0)


def load_dvs(scene):
    """frame -> list of dicts (box, centre, keyframe), 'certain' only."""
    out = defaultdict(list)
    with open(f"{DATA_DIR}/{scene}_annotation.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["confidence"] != "certain":
                continue
            box = tuple(float(r[k]) for k in ("left", "top", "width", "height"))
            out[int(r["frame_index"])].append(dict(box=box, c=center(box), key=r["is_keyframe"] == "True"))
    return out


def board_hull():
    """Convex hull of all RGB board points of the calibration video (cached)."""
    if not os.path.exists(BOARD_CACHE):
        dl, dr = s2.detect(s2.DVS_MP4), s2.detect(s2.RGB_MP4)
        pts = np.concatenate([dr[f] for f in sorted(set(dl) & set(dr))])
        np.save(BOARD_CACHE, pts)
    return cv2.convexHull(np.float32(np.load(BOARD_CACHE)))


def setup():
    cal = dict(np.load(s2.CALIB))
    rect = s2.build_rectification(cal)
    rect.update(dL=cal["dL"], KR=cal["KR"], dR=cal["dR"])
    rect["to_rgb"] = cal["KR"][1, 1] / rect["P1"][1, 1]
    # P2[0,3] = f * Tx (in calibration grid units); convert with the known 60 mm baseline
    rect["fB_mm"] = abs(rect["P2"][0, 3]) / np.linalg.norm(cal["T"]) * BASELINE_MM
    return rect


def rect_dvs(rect, pts):
    return s2.rect_points_dvs(rect, np.float32(pts).reshape(-1, 2))


def rect_rgb(rect, pts):
    return s2.rect_points_rgb(rect, np.float32(pts).reshape(-1, 2))


def single_pairs(dvs, rgb, k):
    """(frame, dvs record at f+k, rgb box at f) for frames with one DVS insect and one RGB label."""
    out = []
    for f, labs in rgb.items():
        d = dvs.get(f + k, [])
        if len(labs) == 1 and len(d) == 1:
            out.append((f, d[0], labs[0]))
    return out


def motion(dvs, f):
    if len(dvs.get(f, [])) == 1 and len(dvs.get(f - 1, [])) == 1:
        return float(np.hypot(*np.subtract(dvs[f][0]["c"], dvs[f - 1][0]["c"])))
    return 0.0


def measure(rect, pairs):
    """Row offset (RGB px) and disparity (rect px) for a list of pairs."""
    if not pairs:
        return np.zeros(0), np.zeros(0)
    qd = rect_dvs(rect, [p[1]["c"] for p in pairs])
    qr = rect_rgb(rect, [center(p[2]) for p in pairs])
    return (qr[:, 1] - qd[:, 1]) * rect["to_rgb"], qd[:, 0] - qr[:, 0]


def fmt(v):
    if len(v) == 0:
        return "   n=0"
    return f"n={len(v):4d} med {np.median(v):+6.1f} IQR {np.percentile(v, 25):+6.1f}..{np.percentile(v, 75):+6.1f}"


def render(rect, scene, f, dvs, rgb):
    size = rect["size"]
    d_img = cv2.remap(cv2.resize(read_frame(scene, "dvs", f), size), *rect["mapL"], cv2.INTER_LINEAR)
    r_img = cv2.remap(read_frame(scene, "rgb", f), *rect["mapR"], cv2.INTER_LINEAR)
    canvas = np.hstack([d_img, np.full((size[1], 16, 3), 40, np.uint8), r_img])
    off = size[0] + 16
    cols = s2.colors(max(1, len(dvs.get(f, []))))
    rows = []
    for rec, c in zip(dvs.get(f, []), cols):
        q = rect_dvs(rect, [rec["c"]])[0]
        y = int(round(q[1]))
        cv2.line(canvas, (0, y), (canvas.shape[1], y), c, 2)
        cv2.circle(canvas, (int(q[0]), y), 10, c, 2)
        rows.append(q[1])
    lab_txt = []
    for b in rgb.get(f, []):
        corners = rect_rgb(rect, [(b[0], b[1]), (b[0] + b[2], b[1] + b[3])])
        (x0, y0), (x1, y1) = corners
        pad = max(0, (14 - min(x1 - x0, y1 - y0)) / 2)
        cv2.rectangle(canvas, (int(x0 - pad) + off, int(y0 - pad)), (int(x1 + pad) + off, int(y1 + pad)), GREEN, 3)
        qc = rect_rgb(rect, [center(b)])[0]
        if rows:
            dy = min((qc[1] - r for r in rows), key=abs) * rect["to_rgb"]
            lab_txt.append(f"{dy:+.1f}")
    canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lines = [
        f"{scene}  frame {f}, rectified   left: DVS frame {f}   right: RGB frame {f}   (calibration from step 2, "
        "nothing fitted to labels)",
        f"coloured line + circle = DVS annotation of frame {f} ({len(dvs.get(f, []))}), line = its row in both images;  "
        f"green box = RGB label ({len(rgb.get(f, []))}), tiny boxes padded",
        "row offset of each RGB label to the nearest DVS line (RGB px, + = label below line): "
        + (", ".join(lab_txt) if lab_txt else "n/a"),
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    hull = board_hull()
    print(f"1 rectified px = {rect['to_rgb']:.2f} RGB px;  f*B = {rect['fB_mm']:.0f} rect px * mm  "
          f"(depth Z = f*B / disparity)")

    print("\n[a] Row offset RGB - DVS (RGB px) and depth, same frame index, single-insect frames")
    print(f"{'scene':10s} {'row offset':40s} {'|off|<5px':>9s} {'disp<=0':>8s} {'depth median, IQR (m)':>24s}")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        dy, disp = measure(rect, single_pairs(dvs, rgb, 0))
        z = rect["fB_mm"] / disp[disp > 0] / 1000
        print(f"{scene:10s} {fmt(dy):40s} {np.mean(np.abs(dy) < 5):9.0%} {np.mean(disp <= 0):8.0%} "
              f"{np.median(z):8.2f}  {np.percentile(z, 25):5.2f}..{np.percentile(z, 75):5.2f}")

    print("\n[b] Timing: median |row offset| (RGB px) for DVS frame f+k vs RGB frame f")
    print("    all = all single-insect pairs; key+moving = DVS box hand-placed and insect moves > 3 DVS px/frame")
    print(f"{'scene':10s} " + " ".join(f"{'k=%+d all' % k:>12s} {'key+mov':>8s}" for k in SHIFTS))
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        cells = []
        for k in SHIFTS:
            pairs = single_pairs(dvs, rgb, k)
            sel = [p for p in pairs if p[1]["key"] and motion(dvs, p[0] + k) > 3]
            a, s = np.abs(measure(rect, pairs)[0]), np.abs(measure(rect, sel)[0])
            cells.append(f"{np.median(a):12.1f} {np.median(s) if len(s) else float('nan'):5.1f}({len(s):2d})")
        print(f"{scene:10s} " + " ".join(cells))

    print("\n[c] Row offset (RGB px) inside / outside the RGB area covered by the calibration board")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        pairs = single_pairs(dvs, rgb, 0)
        dy, _ = measure(rect, pairs)
        inside = np.array([cv2.pointPolygonTest(hull, center(p[2]), False) >= 0 for p in pairs])
        print(f"{scene:10s} inside {fmt(dy[inside])}   outside {fmt(dy[~inside])}")

    print("\n[d] Images")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        dvs_plain = {f: [r["box"] for r in v] for f, v in dvs.items()}
        picks = pick_frames(rgb, dvs_plain)
        for f in picks:
            render(rect, scene, f, dvs, rgb)
        print(f"{scene:10s} frames {picks}")


if __name__ == "__main__":
    main()
