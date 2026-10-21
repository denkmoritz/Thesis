"""
Step 5 of the zone rebuild: the (corrected) epipolar line in the raw RGB image.

For a DVS point, the corrected line in the rectified RGB image is
    y(x) = y_dvs + a + b*x + c*y(x)   ->   y(x) = (y_dvs + a + b*x) / (1 - c)
(step-4 correction, coefficients in rectified px). Each point on it is mapped
back into raw RGB pixels:
    rectified pixel -> ray of the rectified camera (P2^-1) -> undo the
    rectification rotation (R2^T) -> project with RGB intrinsics + distortion
so the straight line becomes a slightly curved line on the original frame.

Checks (printed):
  - correctness: the uncorrected curve must coincide with the epipolar line
    from the fundamental matrix F (distortion applied), max gap in px
  - per scene, on the step-4 TEST frames: distance of each matched RGB label
    centre to its curve in raw RGB px, without and with correction

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/epiline.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import SCENES, load_rgb, read_frame, center
from dti.geometry.rectify import load_dvs, setup, rect_dvs, rect_rgb
from dti.geometry.scene_correction import match, K
from dti.geometry import calibration as s2

OUT_DIR = "outputs/step5"
CORR = "outputs/step4/scene_corrections.json"
WHITE, GREEN, GREY = (255, 255, 255), (0, 255, 0), (170, 170, 170)
RGB_W, RGB_H = 1920, 1200


def rect_to_raw(rect, pts):
    """Rectified RGB pixels (n, 2) -> raw (distorted) RGB pixels (n, 2)."""
    P = rect["P2"]
    xn = (pts[:, 0] - P[0, 2]) / P[0, 0]
    yn = (pts[:, 1] - P[1, 2]) / P[1, 1]
    rays = (rect["R2"].T @ np.c_[xn, yn, np.ones(len(xn))].T).T
    rays = rays[rays[:, 2] > 0]
    return cv2.projectPoints(rays, np.zeros(3), np.zeros(3), rect["KR"], rect["dR"])[0].reshape(-1, 2)


def raw_curve(rect, y_dvs, coef):
    """Raw-RGB polyline of the (corrected) line through a rectified DVS row, clipped to the image."""
    x = np.arange(-400, RGB_W + 400, 2.0)  # rectified x; wide enough to cover the raw image
    if coef is None:
        y = np.full_like(x, y_dvs)
    elif len(coef) == 1:
        y = np.full_like(x, y_dvs + coef[0])
    else:
        a, b, c = coef
        y = (y_dvs + a + b * x) / (1 - c)
    pts = np.c_[x, y]
    raw = rect_to_raw(rect, pts)
    # far outside the calibrated area the distortion polynomial folds back and maps points into the
    # image a second time; keep only points that map back to where they came from (round trip < 1 px)
    back = cv2.undistortPoints(np.float32(raw)[:, None], rect["KR"], rect["dR"], R=rect["R2"],
                               P=rect["P2"]).reshape(-1, 2)
    ok = np.linalg.norm(back - pts, axis=1) < 1.0
    inside = (raw[:, 0] >= 0) & (raw[:, 0] < RGB_W) & (raw[:, 1] >= 0) & (raw[:, 1] < RGB_H)
    return raw[ok & inside]


def f_matrix_curve(cal, p_dvs):
    """Epipolar line from F (undistorted RGB px), with RGB distortion applied -- as in step 2."""
    ul = cv2.undistortPoints(np.float32([[p_dvs]]), cal["KL"], cal["dL"], P=cal["KL"])
    a, b, c = cv2.computeCorrespondEpilines(ul, 1, cal["F"]).reshape(3)
    xs = np.linspace(0, RGB_W, 400)
    ys = -(a * xs + c) / b
    norm = cv2.undistortPoints(np.float32(np.c_[xs, ys])[:, None], cal["KR"], None).reshape(-1, 2)
    raw = cv2.projectPoints(np.c_[norm, np.ones(len(norm))], np.zeros(3), np.zeros(3), cal["KR"], cal["dR"])[0]
    return raw.reshape(-1, 2)


def curve_gap(c1, c2):
    """Max distance from points of c1 to the polyline c2 (dense enough to use nearest vertex)."""
    d = np.linalg.norm(c1[:, None] - c2[None], axis=2).min(axis=1)
    return d.max()


def dist_to_curve(p, curve):
    return float(np.linalg.norm(curve - np.asarray(p)[None], axis=1).min()) if len(curve) else np.inf


def dense(curve, step=0.5):
    """Resample a polyline so nearest-vertex distance is accurate to ~step px."""
    out = [curve[0]]
    for a, b in zip(curve[:-1], curve[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        out.extend(a + (b - a) * t for t in np.linspace(1 / n, 1, n))
    return np.array(out)


def render(rect, scene, f, dvs, rgb, coef, split):
    d_img = read_frame(scene, "dvs", f + K)
    r_img = read_frame(scene, "rgb", f)
    recs = dvs.get(f + K, [])
    cols = s2.colors(max(1, len(recs)))
    for rec, c in zip(recs, cols):
        x, y, w, h = rec["box"]
        pad = max(0, (10 - min(w, h)) / 2)
        cv2.rectangle(d_img, (int(x - pad), int(y - pad)), (int(x + w + pad), int(y + h + pad)), c, 2)
        yd = rect_dvs(rect, [rec["c"]])[0, 1]
        cv2.polylines(r_img, [np.int32(raw_curve(rect, yd, None))], False, GREY, 1)
        cv2.polylines(r_img, [np.int32(raw_curve(rect, yd, coef))], False, c, 3)
    for b in rgb[f]:
        pad = max(0, (14 - min(b[2], b[3])) / 2)
        cv2.rectangle(r_img, (int(b[0] - pad), int(b[1] - pad)), (int(b[0] + b[2] + pad), int(b[1] + b[3] + pad)),
                      GREEN, 3)
    s = RGB_H / 720
    d_img = cv2.resize(d_img, None, fx=s, fy=s)
    canvas = cv2.resize(np.hstack([d_img, np.full((RGB_H, 16, 3), 40, np.uint8), r_img]), None, fx=0.5, fy=0.5,
                        interpolation=cv2.INTER_AREA)
    lines = [
        f"{scene}  RGB frame {f} ({split} part), NOT rectified   left: DVS frame {f + K} (annotation f-1, shown "
        f"enlarged)   right: RGB frame {f} (1920x1200)",
        "coloured box (DVS) and coloured curve (RGB) of the same colour belong together: curve = corrected epipolar "
        "line; thin grey curve = same line without correction",
        f"green box = RGB label ({len(rgb[f])});  DVS annotations: {len(recs)}",
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    cal = dict(np.load(s2.CALIB))
    corr = json.load(open(CORR))

    # correctness check on a grid of DVS points
    gaps = []
    for u in np.linspace(100, 1180, 6):
        for v in np.linspace(80, 640, 5):
            yd = rect_dvs(rect, [(u, v)])[0, 1]
            mine = raw_curve(rect, yd, None)
            ref = f_matrix_curve(cal, (u, v))
            ref = ref[(ref[:, 0] >= 0) & (ref[:, 0] < RGB_W) & (ref[:, 1] >= 0) & (ref[:, 1] < RGB_H)]
            if len(mine) and len(ref):
                gaps.append(curve_gap(ref, dense(mine)))
    print(f"Correctness: uncorrected raw curve vs F-matrix epipolar line, 30 DVS points: "
          f"median gap {np.median(gaps):.2f} px, max {np.max(gaps):.2f} px")

    print("\nDistance of RGB label centre to its curve in raw RGB px, step-4 TEST frames")
    print(f"{'scene':10s} {'n':>4s}  {'without: median  <10px':>24s}  {'with correction: median  <10px':>32s}")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        c = corr[scene]
        coef = np.array(c["coef_rect_px"])
        test = sorted(f for f in rgb if rgb[f] and c["test_frames"][0] <= f <= c["test_frames"][1])
        d0, d1 = [], []
        for f in test:
            recs = dvs.get(f + K, [])
            if not recs:
                continue
            curves0 = [dense(raw_curve(rect, rect_dvs(rect, [r["c"]])[0, 1], None)) for r in recs]
            curves1 = [dense(raw_curve(rect, rect_dvs(rect, [r["c"]])[0, 1], coef)) for r in recs]
            # same pairing as step 4: matched with the fitted model
            for _, lab in _pairs_with_labels(rect, recs, rgb[f], coef):
                i = lab["dvs_i"]
                d0.append(dist_to_curve(center(lab["box"]), curves0[i]))
                d1.append(dist_to_curve(center(lab["box"]), curves1[i]))
        d0, d1 = np.array(d0), np.array(d1)
        print(f"{scene:10s} {len(d0):4d}  {np.median(d0):14.1f} {np.mean(d0 < 10):7.0%}  "
              f"{np.median(d1):22.1f} {np.mean(d1 < 10):7.0%}")
        busiest = max(test, key=lambda f: (len(rgb[f]), -f))
        picks = [busiest] + [f for f in (test[len(test) // 3], test[2 * len(test) // 3]) if f != busiest]
        for f in picks[:3]:
            render(rect, scene, f, dvs, rgb, coef, "test")
        print(f"{'':10s} images: frames {picks[:3]}")


def _pairs_with_labels(rect, recs, labs, coef):
    """step-4 matching, returning which DVS record and which label box form each pair."""
    out = []
    qr = rect_rgb(rect, [center(b) for b in labs])
    qd = rect_dvs(rect, [r["c"] for r in recs])
    for p in match(rect, recs, labs, coef):
        i_lab = int(np.argmin(np.linalg.norm(qr - p[0], axis=1)))
        i_dvs = int(np.argmin(np.abs(qd[:, 1] - p[1])))
        out.append((p, dict(box=labs[i_lab], dvs_i=i_dvs)))
    return out


if __name__ == "__main__":
    main()
