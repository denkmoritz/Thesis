"""
Step 4 of the zone rebuild: per-scene correction of the rectified row offset.

Step 3 showed a systematic row offset between DVS line and RGB label per scene
(e.g. ~+17 RGB px in 3_m-h-h), most likely because the rig pose changed between
the calibration recording and the scene (rig moved / touched per scene, no new
extrinsic calibration). Here we correct it, fitted on one part of the data and
judged only on another part:

  - split per scene in time: first 60% of labelled frames = fit, last 40% = test
  - models for the row offset (rectified RGB row - rectified DVS row):
      none    : 0
      const   : a                     (one number per scene)
      linear  : a + b*x + c*y         (x, y = rectified RGB position; a small
                                       camera rotation looks like this)
  - matching: each RGB label is paired with a different DVS insect so that the
    total |offset - model| is smallest. On the fit part the matching is repeated
    with the fitted model; on the test part only the fitted model is used.

DVS annotation of frame f-1 is used for RGB frame f (step 3 timing result).

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/scene_correction.py
"""

import itertools
import json
import os

import cv2
import numpy as np

from dti.io.data_sync import SCENES, load_rgb, read_frame, center
from dti.geometry.rectify import load_dvs, setup, rect_dvs, rect_rgb
from dti.geometry import calibration as s2

OUT_DIR = "outputs/step4"
K = -1
FIT_SHARE = 0.6
MAX_INSECTS = 7
WHITE, GREEN, GREY = (255, 255, 255), (0, 255, 0), (170, 170, 170)
MODELS = ["none", "const", "linear"]


def predict(model, xy):
    """Model offset (rect px) at rectified RGB positions xy (n, 2)."""
    if model is None:
        return np.zeros(len(xy))
    return np.c_[np.ones(len(xy)), xy] @ model if len(model) == 3 else np.full(len(xy), model[0])


def match(rect, dvs_recs, labs, model):
    """Pairs (rgb_xy, dvs_row, rgb_row) with the smallest total |offset - model|."""
    if not dvs_recs or not labs or len(dvs_recs) > MAX_INSECTS or len(labs) > MAX_INSECTS:
        return []
    qd = rect_dvs(rect, [r["c"] for r in dvs_recs])
    qr = rect_rgb(rect, [center(b) for b in labs])
    pred = predict(model, qr)
    n = min(len(qd), len(qr))
    best, best_cost = None, np.inf
    for li in itertools.permutations(range(len(qr)), n):
        for di in itertools.permutations(range(len(qd)), n):
            cost = sum(abs(qr[i, 1] - qd[j, 1] - pred[i]) for i, j in zip(li, di))
            if cost < best_cost:
                best, best_cost = list(zip(li, di)), cost
        if len(qr) <= len(qd):
            break  # all labels used; only the DVS choice matters
    return [(qr[i], qd[j, 1], qr[i, 1]) for i, j in best]


def collect(rect, dvs, rgb, frames, model):
    xy, off = [], []
    for f in frames:
        for p in match(rect, dvs.get(f + K, []), rgb[f], model):
            xy.append(p[0])
            off.append(p[2] - p[1])
    return np.array(xy).reshape(-1, 2), np.array(off)


def fit(kind, xy, off):
    """Robust fit: least squares, drop residuals > 3 MAD, refit."""
    if kind == "none":
        return None
    A = np.ones((len(off), 1)) if kind == "const" else np.c_[np.ones(len(off)), xy]
    keep = np.ones(len(off), bool)
    for _ in range(2):
        coef = np.linalg.lstsq(A[keep], off[keep], rcond=None)[0]
        res = off - A @ coef
        mad = np.median(np.abs(res - np.median(res))) + 1e-6
        keep = np.abs(res) < 3 * 1.4826 * mad + 1.0
    return coef


def summary(res_rgb):
    a = np.abs(res_rgb)
    return (f"n={len(a):4d} med|.| {np.median(a):5.1f}  <5px {np.mean(a < 5):4.0%}  <10px {np.mean(a < 10):4.0%}  "
            f"signed med {np.median(res_rgb):+6.1f}")


def draw_plot(results, to_rgb, path):
    """Per scene: test row offset (RGB px) vs RGB image row, before (grey) and after (colour)."""
    W, H, M = 360, 300, 40
    panels = []
    for scene, r in results.items():
        p = np.full((H + 2 * M, W + 2 * M, 3), 255, np.uint8)
        cv2.rectangle(p, (M, M), (M + W, M + H), (0, 0, 0), 1)

        def to_px(yimg, off):
            return int(M + yimg / 1200 * W), int(M + H / 2 - np.clip(off, -40, 40) / 40 * H / 2)

        for v in (-20, 0, 20):
            yy = to_px(0, v)[1]
            cv2.line(p, (M, yy), (M + W, yy), (200, 200, 200) if v else (0, 0, 0), 1)
            cv2.putText(p, f"{v:+d}", (4, yy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1, cv2.LINE_AA)
        yimg = r["test_xy"][:, 1]
        for y0, o in zip(yimg, r["test_before"]):
            cv2.circle(p, to_px(y0, o), 3, GREY, -1)
        for y0, o in zip(yimg, r["test_after"]):
            cv2.circle(p, to_px(y0, o), 3, (200, 80, 0), -1)
        cv2.putText(p, f"{scene}  (model: {r['model']})", (M, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                    cv2.LINE_AA)
        cv2.putText(p, "rectified image row: 0 (top) .. 1200 (bottom)", (M, H + M + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 0), 1, cv2.LINE_AA)
        panels.append(p)
    blank = np.full_like(panels[0], 255)
    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:] + [blank])])
    cap = np.zeros((80, grid.shape[1], 3), np.uint8)
    for i, t in enumerate([
        "row offset RGB label - DVS line (RGB px, clipped to +-40) on the TEST frames (last 40% of each scene)",
        "vs rectified image row (top of image = left).  grey = no correction,  blue = with the correction fitted "
        "on the first 60%",
        "lines at -20, 0, +20 px",
    ]):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(path, np.vstack([grid, cap]))


def render(rect, scene, f, dvs, rgb, model, split_label):
    size = rect["size"]
    d_img = cv2.remap(cv2.resize(read_frame(scene, "dvs", f + K), size), *rect["mapL"], cv2.INTER_LINEAR)
    r_img = cv2.remap(read_frame(scene, "rgb", f), *rect["mapR"], cv2.INTER_LINEAR)
    canvas = np.hstack([d_img, np.full((size[1], 16, 3), 40, np.uint8), r_img])
    off = size[0] + 16
    recs = dvs.get(f + K, [])
    cols = s2.colors(max(1, len(recs)))
    for rec, c in zip(recs, cols):
        q = rect_dvs(rect, [rec["c"]])[0]
        cv2.circle(canvas, (int(q[0]), int(q[1])), 10, c, 2)
        cv2.line(canvas, (0, int(q[1])), (canvas.shape[1], int(q[1])), GREY, 1)  # before
        # after: the corrected row depends on the RGB position (linear model), draw it across the RGB pane
        xs = np.arange(0, size[0], 8, dtype=np.float64)
        ys = q[1] + predict(model, np.c_[xs, np.full_like(xs, q[1])])
        cv2.polylines(canvas, [np.int32(np.c_[xs + off, ys])], False, c, 2)
    txt = []
    for b in rgb[f]:
        (x0, y0), (x1, y1) = rect_rgb(rect, [(b[0], b[1]), (b[0] + b[2], b[1] + b[3])])
        pad = max(0, (14 - min(x1 - x0, y1 - y0)) / 2)
        cv2.rectangle(canvas, (int(x0 - pad) + off, int(y0 - pad)), (int(x1 + pad) + off, int(y1 + pad)), GREEN, 3)
    for p in match(rect, recs, rgb[f], model):
        before = (p[2] - p[1]) * rect["to_rgb"]
        after = (p[2] - p[1] - predict(model, p[0][None])[0]) * rect["to_rgb"]
        txt.append(f"{before:+.1f} -> {after:+.1f}")
    canvas = cv2.resize(canvas, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    lines = [
        f"{scene}  RGB frame {f} ({split_label} part), rectified   left: DVS frame {f + K} (annotation f-1)   "
        f"right: RGB frame {f}",
        "circle = DVS annotation; grey line = row without correction; coloured line = corrected row in the RGB "
        "image (same colour as its DVS circle); green box = RGB label",
        "row offset per matched label, RGB px, before -> after correction: " + (", ".join(txt) if txt else "n/a"),
    ]
    cap = np.zeros((22 * len(lines) + 10, canvas.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{OUT_DIR}/{scene}_frame_{f:06d}.jpg", np.vstack([canvas, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rect = setup()
    to_rgb = rect["to_rgb"]
    results, saved = {}, {}
    print(f"DVS frame f{K:+d} vs RGB frame f; fit on first {FIT_SHARE:.0%} of labelled frames per scene, "
          "test on the rest")
    for scene in SCENES:
        dvs, rgb = load_dvs(scene), load_rgb(scene)
        frames = sorted(f for f in rgb if rgb[f])
        cut = int(FIT_SHARE * len(frames))
        fit_frames, test_frames = frames[:cut], frames[cut:]
        print(f"\n{scene}: fit frames {fit_frames[0]}..{fit_frames[-1]} ({len(fit_frames)}), "
              f"test frames {test_frames[0]}..{test_frames[-1]} ({len(test_frames)})")
        models, test = {}, {}
        for kind in MODELS:
            m = None
            for _ in range(3):  # match -> fit -> re-match with the fitted model
                xy, off = collect(rect, dvs, rgb, fit_frames, m)
                m = fit(kind, xy, off)
            models[kind] = m
            txy, toff = collect(rect, dvs, rgb, test_frames, m)
            test[kind] = (txy, (toff - predict(m, txy)) * to_rgb)
            coef = "" if m is None else f"  shift {m[0] * to_rgb:+.1f} px" if len(m) == 1 else (
                f"  shift at image centre {predict(m, np.array([[960.0, 600.0]]))[0] * to_rgb:+.1f} px, "
                f"change left->right {m[1] * 1920 * to_rgb:+.1f} px, top->bottom {m[2] * 1200 * to_rgb:+.1f} px")
            print(f"  {kind:7s} test: {summary(test[kind][1])}{coef}")
        # keep linear only if it clearly beats const on the test part
        med = {k: np.median(np.abs(test[k][1])) for k in MODELS}
        best = "linear" if med["linear"] < med["const"] - 0.5 else "const"
        print(f"  -> chosen: {best}")
        results[scene] = dict(model=best, test_xy=test[best][0], test_before=test["none"][1],
                              test_after=test[best][1])
        # before/after are matched separately; for the plot use the same pairs: re-match with the chosen model
        txy, toff = collect(rect, dvs, rgb, test_frames, models[best])
        results[scene].update(test_xy=txy, test_before=toff * to_rgb,
                              test_after=(toff - predict(models[best], txy)) * to_rgb)
        saved[scene] = dict(model=best, coef_rect_px=[float(v) for v in models[best]],
                            fit_frames=[fit_frames[0], fit_frames[-1]], test_frames=[test_frames[0], test_frames[-1]],
                            dvs_shift=K)
        # example images from the TEST part: most labels, and two spread over the test part
        busiest = max(test_frames, key=lambda f: (len(rgb[f]), -f))
        picks = [busiest] + [f for f in (test_frames[len(test_frames) // 3], test_frames[2 * len(test_frames) // 3])
                             if f != busiest]
        for f in picks[:3]:
            render(rect, scene, f, dvs, rgb, models[best], "test")
        print(f"  images (test part): frames {picks[:3]}")

    draw_plot(results, to_rgb, f"{OUT_DIR}/offset_vs_row_test.png")
    with open(f"{OUT_DIR}/scene_corrections.json", "w") as fh:
        json.dump(saved, fh, indent=2)
    print(f"\nSaved {OUT_DIR}/scene_corrections.json and offset_vs_row_test.png")


if __name__ == "__main__":
    main()
