"""
Step 6 of the zone rebuild: which part of the (corrected) line is physically possible.

Along the rectified line, the RGB position of an insect is x_rgb = x_dvs - d with
disparity d = f*B / Z (f*B from calibration + 60 mm baseline). A depth range
[Z_near, Z_far] therefore gives a disparity range = a segment of the line.

Two scenarios, compared on the step-4 TEST frames (nothing fitted here):
  A  fixed range 0.3 .. 3.5 m
  B  physical limits: far = infinitely far (d >= 0); near = where the insect
     would leave the RGB image (end of the shared field of view) -- computed
     per DVS point, not chosen

Printed per scene: share of RGB labels whose disparity lies in the segment
(strict and with 10 RGB px tolerance for label noise), segment length in raw
RGB px vs the full line, and the depth of the labels (Z = f*B / d).

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/depth_segment.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import SCENES, load_rgb, read_frame, center
from dti.geometry.rectify import load_dvs, setup, rect_dvs, rect_rgb
from dti.geometry.scene_correction import K
from dti.geometry.epiline import rect_to_raw, raw_curve, _pairs_with_labels, CORR, RGB_W, RGB_H
from dti.geometry import calibration as s2

OUT_DIR = "outputs/step6"
Z_A = (300.0, 3500.0)  # mm
TOL_RGB = 10.0
WHITE, GREEN = (255, 255, 255), (0, 255, 0)


def line_y(y_dvs, x, coef):
    a, b, c = coef
    return (y_dvs + a + b * x) / (1 - c)


def raw_points(rect, x_dvs, y_dvs, coef, d):
    """Raw RGB pixels of the corrected line at disparities d; NaN where not valid (outside image / fold-over)."""
    x = x_dvs - np.asarray(d, float)
    pts = np.c_[x, line_y(y_dvs, x, coef)]
    P = rect["P2"]
    rays = (rect["R2"].T @ np.c_[(pts[:, 0] - P[0, 2]) / P[0, 0], (pts[:, 1] - P[1, 2]) / P[1, 1],
                                 np.ones(len(pts))].T).T
    raw = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), rect["KR"], rect["dR"])[0].reshape(-1, 2)
    back = cv2.undistortPoints(np.float32(raw)[:, None], rect["KR"], rect["dR"], R=rect["R2"],
                               P=rect["P2"]).reshape(-1, 2)
    ok = (np.linalg.norm(back - pts, axis=1) < 1) & (rays[:, 2] > 0)
    ok &= (raw[:, 0] >= 0) & (raw[:, 0] < RGB_W) & (raw[:, 1] >= 0) & (raw[:, 1] < RGB_H)
    raw[~ok] = np.nan
    return raw


def segment_B(rect, x_dvs, y_dvs, coef):
    """Disparity range [0, d_max]: from 'infinitely far' until the point leaves the RGB image."""
    d = np.arange(0, 3000, 1.0)
    raw = raw_points(rect, x_dvs, y_dvs, coef, d)
    valid = ~np.isnan(raw[:, 0])
    if not valid[0]:
        return None  # even an infinitely far insect is outside the RGB image
    d_max = d[np.argmin(valid)] - 1 if not valid.all() else d[-1]
    return 0.0, float(d_max)


def seg_length(rect, x_dvs, y_dvs, coef, d0, d1):
    raw = raw_points(rect, x_dvs, y_dvs, coef, np.arange(d0, d1 + 1, 1.0))
    raw = raw[~np.isnan(raw[:, 0])]
    return float(np.linalg.norm(np.diff(raw, axis=0), axis=1).sum()) if len(raw) > 1 else 0.0


def render(rect, scene, f, dvs, rgb, coef):
    d_img = read_frame(scene, "dvs", f + K)
    r_img = read_frame(scene, "rgb", f)
    recs = dvs.get(f + K, [])
    cols = s2.colors(max(1, len(recs)))
    dA = (rect["fB_mm"] / Z_A[1], rect["fB_mm"] / Z_A[0])
    for rec, c in zip(recs, cols):
        x, y, w, h = rec["box"]
        pad = max(0, (10 - min(w, h)) / 2)
        cv2.rectangle(d_img, (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad)), c, 2)
        xd, yd = rect_dvs(rect, [rec["c"]])[0]
        cv2.polylines(r_img, [np.int32(raw_curve(rect, yd, coef))], False, (120, 120, 120), 1)  # full line
        sB = segment_B(rect, xd, yd, coef)
        if sB:
            pB = raw_points(rect, xd, yd, coef, np.arange(sB[0], sB[1] + 1))
            cv2.polylines(r_img, [np.int32(pB[~np.isnan(pB[:, 0])])], False, c, 9)
        pA = raw_points(rect, xd, yd, coef, np.arange(dA[0], dA[1] + 1))
        pA = pA[~np.isnan(pA[:, 0])]
        if len(pA) > 1:
            cv2.polylines(r_img, [np.int32(pA)], False, WHITE, 2)
    for b in rgb[f]:
        pad = max(0, (14 - min(b[2], b[3])) / 2)
        cv2.rectangle(r_img, (int(b[0] - pad), int(b[1] - pad)), (int(b[0] + b[2] + pad), int(b[1] + b[3] + pad)),
                      GREEN, 3)
    s = RGB_H / 720
    d_img = cv2.resize(d_img, None, fx=s, fy=s)
    canvas = cv2.resize(np.hstack([d_img, np.full((RGB_H, 16, 3), 40, np.uint8), r_img]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    lines = [
        f"{scene}  RGB frame {f} (test part), NOT rectified   left: DVS frame {f + K} (annotation f-1, enlarged)   "
        "right: RGB frame (1920x1200)",
        "thin grey = full corrected line;  thick coloured = segment B (infinitely far .. leaves RGB image);  "
        "thin white = segment A (0.3 .. 3.5 m)",
        f"colour links DVS box and its segment;  green box = RGB label ({len(rgb[f])})",
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    corr = json.load(open(CORR))
    fB = rect["fB_mm"]
    tol = TOL_RGB / rect["to_rgb"]
    dA = (fB / Z_A[1], fB / Z_A[0])
    print(f"f*B = {fB:.0f} rect px*mm; 1 rect px = {rect['to_rgb']:.2f} RGB px")
    print(f"A: {Z_A[0] / 1000:.1f}..{Z_A[1] / 1000:.1f} m  ->  disparity {dA[0]:.1f}..{dA[1]:.1f} rect px "
          f"({dA[0] * rect['to_rgb']:.0f}..{dA[1] * rect['to_rgb']:.0f} RGB px)")
    print("B: 0 .. d_max per DVS point (d_max = RGB image border), near limit Z = f*B/d_max\n")
    print(f"{'scene':10s} {'n':>4s} {'label depth m: med (p5..p95)':>30s} {'neg d':>6s} | "
          f"{'A in / +-10px':>13s} {'B in / +-10px':>13s} | {'len A':>6s} {'len B':>6s} {'full':>6s} | "
          f"{'B near Z m':>10s}")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        coef = np.array(c["coef_rect_px"])
        coef = coef if len(coef) == 3 else np.array([coef[0], 0.0, 0.0])
        test = sorted(f for f in rgb if rgb[f] and c["test_frames"][0] <= f <= c["test_frames"][1])
        rows = []
        for f in test:
            recs = dvs.get(f + K, [])
            if not recs:
                continue
            for _, lab in _pairs_with_labels(rect, recs, rgb[f], coef):
                xd, yd = rect_dvs(rect, [recs[lab["dvs_i"]]["c"]])[0]
                xr = rect_rgb(rect, [center(lab["box"])])[0, 0]
                d = xd - xr
                sB = segment_B(rect, xd, yd, coef)
                inA = dA[0] <= d <= dA[1]
                inA_t = dA[0] - tol <= d <= dA[1] + tol
                inB = sB is not None and sB[0] <= d <= sB[1]
                inB_t = sB is not None and sB[0] - tol <= d <= sB[1] + tol
                lenA = seg_length(rect, xd, yd, coef, *dA)
                lenB = seg_length(rect, xd, yd, coef, *sB) if sB else 0.0
                full = len(raw_curve(rect, yd, coef))
                full = float(np.linalg.norm(np.diff(raw_curve(rect, yd, coef), axis=0), axis=1).sum()) if full > 1 else 0
                zB = fB / sB[1] / 1000 if sB and sB[1] > 0 else np.nan
                rows.append((d, inA, inA_t, inB, inB_t, lenA, lenB, full, zB))
        r = np.array(rows, float)
        d = r[:, 0]
        z = fB / d[d > 0] / 1000
        print(f"{scene:10s} {len(r):4d} {np.median(z):12.2f} ({np.percentile(z, 5):.2f}..{np.percentile(z, 95):5.2f}) "
              f"{np.mean(d <= 0):6.0%} | {r[:, 1].mean():5.0%} / {r[:, 2].mean():4.0%} "
              f"{r[:, 3].mean():6.0%} / {r[:, 4].mean():4.0%} | {np.median(r[:, 5]):6.0f} {np.median(r[:, 6]):6.0f} "
              f"{np.median(r[:, 7]):6.0f} | {np.nanmedian(r[:, 8]):10.2f}")
        busiest = max(test, key=lambda f: (len(rgb[f]), -f))
        picks = [busiest] + [f for f in (test[len(test) // 3], test[2 * len(test) // 3]) if f != busiest]
        for f in picks[:3]:
            render(rect, scene, f, dvs, rgb, coef)
    print("\nlen = median segment length in raw RGB px; full = whole corrected line inside the RGB image")


if __name__ == "__main__":
    main()
