"""
Step 13 of the zone rebuild: per-scene row correction without RGB labels?

Every scene has its own row offset between the DVS line and the RGB insect
(the rig was moved between scenes). Zone v2 fits it on 60 % of the scene's RGB
labels, which is not allowed for a new scene. Here four ways are compared, for
all six scenes, on the same held-out labels (last 40 % of the labelled frames):

  none        no correction
  label-free  fitted on automatic DVS-RGB matches, no RGB label used:
              150 frames of the scene's video with exactly one DVS insect; in a
              wide band around the uncorrected line (+-40 RGB px, any depth from
              0.2 m outwards) the strongest RGB motion peak (step 11d, background
              model) is taken as the insect. Wrong peaks (plants) scatter, right
              ones agree: the most frequent offset is found first, then the
              linear model a + b*x + c*y is fitted to the matches near it.
  first N     fitted on the first N labelled frames (N = 5, 10: constant shift;
              N = 20, 40: linear model)
  label 60 %  the step-4 way: linear model on the first 60 % of labelled frames

Nothing here changes the zone v2 files (scene_corrections.json is not written).
The label-free result is saved to outputs/step13/labelfree_corrections.json.

Printed per scene and method: remaining row offset on the held-out labels
(RGB px): median |offset|, share below 5 px and 10 px. For label-free also the
number of matches, and the disparities of the matches (sign check of 3.4).

Run: PYTHONPATH=src ./venv/bin/python src/dti/geometry/correction_compare.py
"""

import json
import os

import cv2
import numpy as np

from dti.io.data_sync import DATA_DIR, SCENES, RGB_W, RGB_H, load_rgb
from dti.geometry.rectify import load_dvs, setup, rect_dvs
from dti.geometry import scene_correction as s4
from dti.geometry.zone import raw_to_rect
from dti.io import scene6 as s6
from dti.motion.cue import read_frames, motion_maps

OUT_DIR = "outputs/step13"
ALL_SCENES = SCENES + [s6.SCENE]
K = s4.K
N_FRAMES = 150
BAND_PX = 40.0  # RGB px, half width of the search band across the line
Z_NEAR = 200.0  # mm
D_MIN = -100.0  # rect px; allows the negative disparities of a sideways-shifted rig (scene 3)
MIN_STRENGTH = 0.08
INLIER_PX = 12.0  # RGB px around the most frequent offset
FEW = [(5, "const"), (10, "const"), (20, "linear"), (40, "linear")]


def labelfree_matches(scene, rect, gx, gy, dvs):
    """(x, y, row offset, disparity, strength) in rectified px for automatic DVS-RGB matches."""
    cap = cv2.VideoCapture(f"{DATA_DIR}/{scene}_rgb.mp4")
    n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cand = [f for f in range(10, n_video - 10) if len(dvs.get(f + K, [])) == 1]
    picks = [cand[i] for i in np.unique(np.linspace(0, len(cand) - 1, N_FRAMES).round().astype(int))]
    masks = s6.load_masks("rgb") if scene == s6.SCENE else {}
    band, d_hi = BAND_PX / rect["to_rgb"], rect["fB_mm"] / Z_NEAR
    out = []
    for f in picks:
        m = motion_maps(read_frames(cap, f, n_video)).get("median")
        if m is None:
            continue
        for mk in masks.values():  # privacy masks switch on and off: no motion there
            if any(g in mk["frames"] for g in range(f - 10, f + 11)):
                m[max(0, mk["top"] - 10):mk["bottom"] + 10, max(0, mk["left"] - 10):mk["right"] + 10] = 0
        xd, yd = rect_dvs(rect, [dvs[f + K][0]["c"]])[0]
        disp = xd - gx
        v = np.where((np.abs(gy - yd) <= band) & (disp >= D_MIN) & (disp <= d_hi), m, 0)
        y, x = np.unravel_index(int(v.argmax()), v.shape)
        if v[y, x] / 255 >= MIN_STRENGTH:
            out.append((gx[y, x], gy[y, x], gy[y, x] - yd, disp[y, x], v[y, x] / 255))
    cap.release()
    return np.array(out).reshape(-1, 5), len(picks)


def robust_linear(matches, to_rgb):
    """Most frequent offset first, then the linear model on the matches near it."""
    off = matches[:, 2]
    edges = np.arange(off.min() - 1, off.max() + 3, 2.0)
    hist = np.convolve(np.histogram(off, edges)[0], [1, 2, 1], mode="same")
    mode = edges[hist.argmax()] + 1.0
    inl = np.abs(off - mode) <= INLIER_PX / to_rgb
    if inl.sum() < 15:
        return np.array([mode, 0.0, 0.0]), inl
    coef = s4.fit("linear", matches[inl, :2], off[inl])
    return coef, inl


def fit_on_labels(rect, dvs, rgb, frames, kind):
    m = None
    for _ in range(3):  # match -> fit -> match again with the fitted model (as in step 4)
        xy, off = s4.collect(rect, dvs, rgb, frames, m)
        if len(off) < (3 if kind == "linear" else 1):
            return None
        m = s4.fit(kind, xy, off)
    return m


def describe(coef, to_rgb):
    if coef is None:
        return ""
    if len(coef) == 1:
        return f"shift {coef[0] * to_rgb:+.1f} px"
    centre = s4.predict(coef, np.array([[960.0, 600.0]]))[0] * to_rgb
    return (f"shift at image centre {centre:+.1f} px, left->right {coef[1] * 1920 * to_rgb:+.1f} px, "
            f"top->bottom {coef[2] * 1200 * to_rgb:+.1f} px")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    s4.OUT_DIR = OUT_DIR
    rect = setup()
    to_rgb = rect["to_rgb"]
    fy, fx = np.mgrid[0:RGB_H, 0:RGB_W]
    g = raw_to_rect(rect, np.c_[fx.ravel(), fy.ravel()])
    gx, gy = g[:, 0].reshape(RGB_H, RGB_W), g[:, 1].reshape(RGB_H, RGB_W)
    saved = {}
    for scene in ALL_SCENES:
        dvs = load_dvs(scene)
        rgb = s6.load_rgb() if scene == s6.SCENE else load_rgb(scene)
        frames = sorted(f for f in rgb if rgb[f])
        cut = int(s4.FIT_SHARE * len(frames))
        first, test = frames[:cut], frames[cut:]

        matches, n_picked = labelfree_matches(scene, rect, gx, gy, dvs)
        coef_lf, inl = robust_linear(matches, to_rgb)
        disp = matches[inl, 3]
        methods = [("none", None), ("label-free", coef_lf)]
        methods += [(f"first {n} frames", fit_on_labels(rect, dvs, rgb, first[:n], kind)) for n, kind in FEW
                    if n <= len(first)]
        methods.append(("label 60 %", fit_on_labels(rect, dvs, rgb, first, "linear")))

        print(f"\n{scene}: {len(frames)} labelled frames, held-out test = last {len(test)} "
              f"(frames {test[0]}..{test[-1]})")
        print(f"  label-free: {n_picked} video frames with one DVS insect, {len(matches)} motion peaks strong enough, "
              f"{int(inl.sum())} agree with the most frequent offset")
        print(f"  label-free disparities of these matches (rect px): 5 % {np.percentile(disp, 5):+.0f}, median "
              f"{np.median(disp):+.0f}, 95 % {np.percentile(disp, 95):+.0f};  negative: {np.mean(disp < 0):.0%}")
        print(f"  {'method':16s} {'n':>4s} {'median |offset|':>15s} {'< 5 px':>7s} {'< 10 px':>8s} {'signed median':>13s}   model")
        for name, coef in methods:
            xy, off = s4.collect(rect, dvs, rgb, test, coef)
            res = (off - s4.predict(coef, xy)) * to_rgb
            a = np.abs(res)
            print(f"  {name:16s} {len(a):4d} {np.median(a):15.1f} {np.mean(a < 5):7.0%} {np.mean(a < 10):8.0%} "
                  f"{np.median(res):+13.1f}   {describe(coef, to_rgb)}")
        saved[scene] = dict(coef_rect_px=[float(v) for v in coef_lf], n_matches=int(inl.sum()),
                            disparity_p5_p50_p95=[float(np.percentile(disp, p)) for p in (5, 50, 95)],
                            negative_disparity_share=float(np.mean(disp < 0)))
        busiest = max(test, key=lambda f: (len(rgb[f]), -f))
        s4.render(rect, scene, busiest, dvs, rgb, coef_lf, "held-out, label-free correction")

    with open(f"{OUT_DIR}/labelfree_corrections.json", "w") as fh:
        json.dump(saved, fh, indent=2)
    print(f"\noffsets in RGB px; 'model' = the fitted correction.  Saved {OUT_DIR}/labelfree_corrections.json; "
          f"one image per scene in {OUT_DIR}/")


if __name__ == "__main__":
    main()
