"""
Step 11 of the zone rebuild: one scoring script for every detector.

Input: predictions and labels as YOLO text files (one file per image,
"class cx cy w h [conf]", normalised). YOLOv7 writes such predictions with
`test.py --save-txt --save-conf --conf-thres 0.001`, Ultralytics with
`val save_txt=True save_conf=True conf=0.001`. All metrics are computed here,
so every model is scored by the same code.

Matching rules (a label can be matched by one prediction only; predictions are
processed from the highest to the lowest confidence):
  iou     prediction and label overlap with IoU >= 0.5 (standard AP@0.5)
  circle  the prediction centre lies inside the smallest circle around the
          label box (Gebauer et al. 2024, box-based evaluation)

Metrics, per scene and for all images:
  AP iou      area under the precision-recall curve, IoU rule (same formula as
              YOLOv7 utils/metrics.py: 101 recall points)
  AP circle   the same with the circle rule
  F1 fixed    precision, recall, F1 with the circle rule at --conf
  F1 best     highest F1 over all confidence thresholds, circle rule (this is
              the optimistic "threshold chosen per scene" of Gebauer et al.)
  F1 loose    the same, but several predictions may match one label (the
              paper does not say whether this is allowed; shown for comparison)
  frames      share of frames correctly called "insect" / "no insect" at --conf

Ignore regions (--ignore DIR with {stem}.png, any size, non-zero = ignore):
a prediction that matches no label and has its centre on an ignore pixel is
removed, it counts neither as right nor as wrong. Used for insects that the DVS
saw but that have no RGB label, and for the privacy masks of scene 6.

Self-test with hand-made cases:
  PYTHONPATH=src ./venv/bin/python src/dti/evaluation/metrics.py --selftest
Score a run:
  PYTHONPATH=src ./venv/bin/python src/dti/evaluation/metrics.py \
      --pred <dir with prediction txt> --gt <dir with label txt> [--ignore DIR] \
      [--conf 0.25] [--images <dir with jpg> --out outputs/step11/<name>]
"""

import argparse
import glob
import os

import cv2
import numpy as np

RGB_W, RGB_H = 1920, 1200
GREEN, CYAN, RED, WHITE = (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 255, 255)
trapezoid = getattr(np, "trapezoid", None) or np.trapz


def load_dir(path, size=(RGB_W, RGB_H)):
    """stem -> array (n, 5): x1, y1, x2, y2, conf in pixels (conf = 1 for labels)."""
    out = {}
    for p in glob.glob(f"{path}/*.txt"):
        rows = [l.split() for l in open(p) if l.strip()]
        a = np.array([[float(v) for v in r[1:6]] + [1.0] * (6 - len(r)) for r in rows]).reshape(-1, 5)
        cx, cy, w, h = a[:, 0] * size[0], a[:, 1] * size[1], a[:, 2] * size[0], a[:, 3] * size[1]
        out[os.path.basename(p)[:-4]] = np.c_[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, a[:, 4]]
    return out


def centres(b):
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


def score_matrix(pred, gt, rule):
    """(n_pred, n_gt): match quality, higher = better, <= 0 = no match."""
    if rule == "iou":
        x1, y1 = np.maximum(pred[:, None, 0], gt[None, :, 0]), np.maximum(pred[:, None, 1], gt[None, :, 1])
        x2, y2 = np.minimum(pred[:, None, 2], gt[None, :, 2]), np.minimum(pred[:, None, 3], gt[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        area = lambda b: (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / (area(pred)[:, None] + area(gt)[None] - inter + 1e-9)
        return np.where(iou >= 0.5, iou, 0.0)
    radius = np.hypot(gt[:, 2] - gt[:, 0], gt[:, 3] - gt[:, 1]) / 2  # half the box diagonal
    dist = np.linalg.norm(centres(pred)[:, None] - centres(gt)[None], axis=2)
    return np.where(dist <= radius[None], 1.0 + 1e-6 - dist / (radius[None] + 1e-9), 0.0)


def match(pred, gt, rule):
    """One image. pred must be sorted by confidence (high to low).

    Returns tp (one prediction per label), tp_loose (any prediction in a label counts),
    gt_conf_loose (highest confidence that hits each label, -1 if none).
    """
    s = score_matrix(pred, gt, rule)
    tp = np.zeros(len(pred), bool)
    free = np.ones(len(gt), bool)
    for i in range(len(pred)):
        cand = np.where(free, s[i], 0.0)
        if cand.size and cand.max() > 0:
            tp[i] = True
            free[cand.argmax()] = False
    hit = s > 0
    gt_conf = np.where(hit.any(0), np.where(hit, pred[:, 4:5], -1.0).max(0), -1.0) if len(pred) else np.full(len(gt), -1.0)
    return tp, hit.any(1), gt_conf


def in_ignore(mask, pts, size=(RGB_W, RGB_H)):
    if mask is None or not len(pts):
        return np.zeros(len(pts), bool)
    h, w = mask.shape[:2]
    x = np.clip((pts[:, 0] * w / size[0]).astype(int), 0, w - 1)
    y = np.clip((pts[:, 1] * h / size[1]).astype(int), 0, h - 1)
    return mask[y, x] > 0


def average_precision(conf, tp, n_gt, tail="yolov7"):
    """101-point AP. tail: how the curve ends after the last recall value (YOLOv7 or Ultralytics convention)."""
    if n_gt == 0 or len(conf) == 0:
        return float("nan") if n_gt == 0 else 0.0
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    recall, precision = tpc / n_gt, tpc / np.arange(1, len(tpc) + 1)
    mrec = np.concatenate(([0.0], recall, [recall[-1] + 0.01 if tail == "yolov7" else 1.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(np.concatenate(([1.0], precision, [0.0])))))
    x = np.linspace(0, 1, 101)
    return float(trapezoid(np.interp(x, mrec, mpre), x))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def best_f1(conf, tp, n_gt):
    """Highest F1 over all thresholds (one prediction per label). Returns F1, threshold."""
    if not len(conf) or n_gt == 0:
        return 0.0, float("nan")
    order = np.argsort(-conf, kind="stable")
    tpc = np.cumsum(tp[order])
    f1 = 2 * tpc / (np.arange(1, len(tpc) + 1) + n_gt)
    i = int(f1.argmax())
    return float(f1[i]), float(conf[order][i])


def best_f1_loose(conf, tp_loose, gt_conf):
    best = 0.0
    for t in np.unique(conf):
        n_pred = np.sum(conf >= t)
        p = np.sum(tp_loose & (conf >= t)) / n_pred
        r = np.mean(gt_conf >= t) if len(gt_conf) else 0.0
        best = max(best, 2 * p * r / (p + r) if p + r else 0.0)
    return float(best)


def evaluate(preds, gts, ignore_dir=None, conf_fixed=0.25, tail="yolov7"):
    """Score all images in gts. Returns {group: metrics} for every scene and for 'all', plus per-image details."""
    per_image = {}
    for stem, gt in gts.items():
        pred = preds.get(stem, np.zeros((0, 5)))
        pred = pred[np.argsort(-pred[:, 4], kind="stable")]
        mask = None
        if ignore_dir and os.path.exists(f"{ignore_dir}/{stem}.png"):
            mask = cv2.imread(f"{ignore_dir}/{stem}.png", cv2.IMREAD_GRAYSCALE)
        ign = in_ignore(mask, centres(pred))
        rec = dict(gt=gt, n_ignored=0)
        for rule in ("iou", "circle"):
            tp, tp_loose, gt_conf = match(pred, gt, rule)
            keep = tp | ~ign  # unmatched predictions on ignore pixels are dropped
            rec[rule] = dict(pred=pred[keep], tp=tp[keep], tp_loose=tp_loose[keep], gt_conf=gt_conf)
            if rule == "circle":
                rec["n_ignored"] = int(np.sum(~keep))
        per_image[stem] = rec

    groups = {}
    for stem in per_image:
        groups.setdefault(stem.split("_frame_")[0], []).append(stem)
    groups["all"] = list(per_image)
    results = {}
    for name, stems in groups.items():
        n_gt = sum(len(per_image[s]["gt"]) for s in stems)
        cat = lambda rule, key: np.concatenate([per_image[s][rule][key] for s in stems]) if stems else np.zeros(0)
        c_iou, t_iou = cat("iou", "pred")[:, 4], cat("iou", "tp").astype(bool)
        c, t = cat("circle", "pred")[:, 4], cat("circle", "tp").astype(bool)
        sel = c >= conf_fixed
        p, r, f = prf(np.sum(t & sel), np.sum(~t & sel), n_gt - np.sum(t & sel))
        f_best, thr = best_f1(c, t, n_gt)
        has_gt = np.array([len(per_image[s]["gt"]) > 0 for s in stems])
        has_pred = np.array([np.any(per_image[s]["circle"]["pred"][:, 4] >= conf_fixed) for s in stems])
        results[name] = dict(
            images=len(stems), labels=n_gt, ignored=sum(per_image[s]["n_ignored"] for s in stems),
            ap_iou=average_precision(c_iou, t_iou, n_gt, tail), ap_circle=average_precision(c, t, n_gt, tail),
            p=p, r=r, f1=f, f1_best=f_best, thr_best=thr,
            f1_loose=best_f1_loose(c, cat("circle", "tp_loose").astype(bool), cat("circle", "gt_conf")),
            frame_acc=float(np.mean(has_gt == has_pred)), empty_frames=int(np.sum(~has_gt)))
    return results, per_image


def print_table(results, conf_fixed):
    print(f"{'scene':10s} {'images':>6s} {'labels':>6s} {'ignored':>7s} | {'AP iou':>6s} {'AP circ':>7s} | "
          f"{'P':>5s} {'R':>5s} {'F1':>5s} (conf {conf_fixed:.2f}) | {'F1 best':>7s} {'at conf':>7s} {'F1 loose':>8s} | "
          f"{'frames ok':>9s} {'empty':>5s}")
    for name in sorted(results, key=lambda n: (n == "all", n)):
        m = results[name]
        print(f"{name:10s} {m['images']:6d} {m['labels']:6d} {m['ignored']:7d} | {m['ap_iou']:6.3f} {m['ap_circle']:7.3f} | "
              f"{m['p']:5.2f} {m['r']:5.2f} {m['f1']:5.2f} {'':11s} | {m['f1_best']:7.2f} {m['thr_best']:7.2f} "
              f"{m['f1_loose']:8.2f} | {m['frame_acc']:9.0%} {m['empty_frames']:5d}")
    print("ignored = predictions removed by ignore regions; empty = frames without a label "
          "(frame accuracy says little if there are few)")


def render(stem, rec, image_dir, out_dir, conf_fixed):
    img = cv2.imread(f"{image_dir}/{stem}.jpg")
    for b in rec["gt"]:
        c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), GREEN, 2)
        cv2.circle(img, (int(c[0]), int(c[1])), int(round(np.hypot(b[2] - b[0], b[3] - b[1]) / 2)), GREEN, 1)
    d = rec["circle"]
    sel = d["pred"][:, 4] >= conf_fixed
    for b, tp in zip(d["pred"][sel], d["tp"][sel]):
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), CYAN if tp else RED, 2)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    n_tp = int(np.sum(d["tp"][sel]))
    lines = [
        f"{stem}   green box = RGB label ({len(rec['gt'])}), green circle = smallest circle around the label box",
        f"predictions with confidence >= {conf_fixed:.2f}: cyan = right (centre inside a circle, {n_tp}), "
        f"red = wrong ({int(sel.sum()) - n_tp});  labels not found: {len(rec['gt']) - n_tp}",
    ]
    cap = np.zeros((22 * len(lines) + 10, img.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(cap, t, (8, 22 * i + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.imwrite(f"{out_dir}/{stem}.jpg", np.vstack([img, cap]), [cv2.IMWRITE_JPEG_QUALITY, 90])


def selftest():
    box = lambda cx, cy, s, conf=1.0: [cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2, conf]
    ev = lambda p, g, **kw: evaluate({"a_frame_000001": np.array(p).reshape(-1, 5)},
                                     {"a_frame_000001": np.array(g).reshape(-1, 5)}, **kw)[0]["all"]
    checks = []
    m = ev([box(100, 100, 10, 0.9)], [box(100, 100, 10)])
    checks.append(("perfect prediction: AP 1, F1 1", m["ap_iou"] > 0.999 and m["ap_circle"] > 0.999 and m["f1"] == 1))
    m = ev([box(100, 100, 10, 0.9), box(300, 300, 10, 0.8), box(200, 200, 10, 0.7)], [box(100, 100, 10), box(200, 200, 10)])
    checks.append((f"right, wrong, right: AP = 0.5*1 + 0.5*2/3 = 0.833 (got {m['ap_iou']:.3f})", abs(m["ap_iou"] - 0.8333) < 0.01))
    checks.append((f"same case: P 0.67, R 1.00, F1 0.80 (got {m['p']:.2f}, {m['r']:.2f}, {m['f1']:.2f})",
                   abs(m["p"] - 2 / 3) < 1e-6 and m["r"] == 1 and abs(m["f1"] - 0.8) < 1e-6))
    m = ev([box(106.9, 100, 10, 0.9)], [box(100, 100, 10)])  # radius of a 10x10 box = 7.07
    checks.append(("centre 6.9 px off a 10 px box: circle rule right, IoU rule wrong", m["ap_circle"] > 0.999 and m["ap_iou"] == 0))
    m = ev([box(107.2, 100, 10, 0.9)], [box(100, 100, 10)])
    checks.append(("centre 7.2 px off: wrong for both rules", m["ap_circle"] == 0 and m["ap_iou"] == 0))
    m = ev([box(100, 100, 10, 0.9), box(101, 100, 10, 0.8)], [box(100, 100, 10)])
    checks.append((f"two predictions in one label: one right, one wrong (F1 {m['f1']:.2f}); loose rule counts both "
                   f"(F1 {m['f1_loose']:.2f})", abs(m["f1"] - 2 / 3) < 1e-6 and m["f1_loose"] == 1))
    m = ev([], [box(100, 100, 10)])
    checks.append(("no prediction: AP 0, recall 0", m["ap_iou"] == 0 and m["r"] == 0))
    tmp = "outputs/step11/selftest_ignore"
    os.makedirs(tmp, exist_ok=True)
    mask = np.zeros((RGB_H // 4, RGB_W // 4), np.uint8)
    mask[60:90, 60:90] = 255  # covers x, y = 240..360 px
    cv2.imwrite(f"{tmp}/a_frame_000001.png", mask)
    p, g = [box(100, 100, 10, 0.9), box(300, 300, 10, 0.8), box(600, 600, 10, 0.7)], [box(100, 100, 10)]
    m0, m1 = ev(p, g), ev(p, g, ignore_dir=tmp)
    checks.append((f"ignore region removes 1 of 2 wrong predictions: precision {m0['p']:.2f} -> {m1['p']:.2f}",
                   abs(m0["p"] - 1 / 3) < 1e-6 and abs(m1["p"] - 0.5) < 1e-6 and m1["ignored"] == 1))
    g2 = [box(300, 300, 10)]
    m = ev([box(300, 300, 10, 0.9)], g2, ignore_dir=tmp)
    checks.append(("a right prediction inside an ignore region still counts", m["f1"] == 1))
    for text, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {text}")
    print(f"self-test: {sum(ok for _, ok in checks)} of {len(checks)} passed")
    return all(ok for _, ok in checks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--pred")
    ap.add_argument("--gt")
    ap.add_argument("--ignore")
    ap.add_argument("--conf", type=float, default=0.25, help="fixed confidence threshold for F1 and frame accuracy")
    ap.add_argument("--tail", default="yolov7", choices=["yolov7", "ultralytics"], help="AP curve ending convention")
    ap.add_argument("--images", help="image folder; with --out, example images are written")
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    preds, gts = load_dir(args.pred), load_dir(args.gt)
    extra = set(preds) - set(gts)
    print(f"labels: {len(gts)} images in {args.gt}\npredictions: {len(preds)} files in {args.pred}"
          + (f" ({len(extra)} without a label file, not used)" if extra else ""))
    results, per_image = evaluate(preds, gts, args.ignore, args.conf, args.tail)
    print_table(results, args.conf)
    if args.images and args.out:
        os.makedirs(args.out, exist_ok=True)
        busy = lambda s: len(per_image[s]["gt"]) + int(np.sum(per_image[s]["circle"]["pred"][:, 4] >= args.conf))
        stems = sorted(per_image, key=lambda s: (-busy(s), s))
        picks = [stems[0], stems[len(stems) // 4], stems[len(stems) // 2]]
        for s in picks:
            render(s, per_image[s], args.images, args.out, args.conf)
        print(f"images: {picks} -> {args.out}/")


if __name__ == "__main__":
    main()
