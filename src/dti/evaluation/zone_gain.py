"""
Step 17d of the zone rebuild: does the zone help, judged against the spread between
random seeds?

Every run of setup 1 exists three times (seed offset 0, 1, 2: m3.slurm / m3_b2.slurm /
m3_channels.slurm / m3d.slurm, then m3_seeds.slurm / m3_nozone_seeds.slurm /
m3d_seeds.slurm). Compared with the run without zone channel (nozone, boxes A):
  A human / B2 human   zones from the human DVS annotation (upper bound)
  A auto / B2 auto     zones from our DVS detector (the realistic system)

Decision rule (fixed 2026-10-16, before the seeds were known): a scene counts as a
gain only if the zone run's WORST seed beats the no-zone run's BEST seed (no
overlap; seeds are not paired, seed offset 1 of two different runs has nothing in
common but the number). The zone helps if this holds in at least 4 of 6 scenes;
checked for AP first, then for "right".

Printed for both rules (circle, IoU): AP, found and right (at the best cut-off, which
is chosen on the scored scene itself, so optimistic) as mean +- standard deviation
over the three seeds, the gain over no zone, '*' where there is no overlap, and the
number of scenes that pass. Then B2 against A with human zones (same input, only the
training boxes differ). All on all labelled frames, human RGB labels, ignore regions.

Run: PYTHONPATH=src ./venv/bin/python src/dti/evaluation/zone_gain.py
"""

import numpy as np

from dti.evaluation.metrics import load_dir, evaluate
from dti.geometry.zone_v3 import ALL_SCENES
from dti.evaluation.score_auto_chain import GT, IGNORE, found_right

RUNS = {"nozone": "outputs/m3/m3_{s}_nozone{k}/labels",
        "A human": "outputs/m3/m3_{s}_A{k}/labels",
        "B2 human": "outputs/m3/m3_{s}_B2{k}/labels",
        "A auto": "outputs/m3d/m3d_{s}_A{k}/labels",
        "B2 auto": "outputs/m3d/m3d_{s}_B2{k}/labels"}
SEEDS = ["", "_s1", "_s2"]
ZONED = [n for n in RUNS if n != "nozone"]
METRICS = (("AP", 0), ("found", 1), ("right", 2))


def scores():
    """(scene, run, rule) -> array (3 seeds, 3) of AP, found, right."""
    gts = load_dir(GT)
    out = {}
    for s in ALL_SCENES:
        gt = {k: b for k, b in gts.items() if k.startswith(s + "_frame_")}
        n_gt = sum(len(b) for b in gt.values())
        for name, path in RUNS.items():
            rows = {"circle": [], "iou": []}
            for k in SEEDS:
                pred = {x: b for x, b in load_dir(path.format(s=s, k=k)).items() if x in gt}
                res, per = evaluate(pred, gt, IGNORE)
                for rule in rows:
                    rows[rule].append((res["all"]["ap_" + rule], *found_right(per, rule, n_gt)))
            for rule in rows:
                out[(s, name, rule)] = np.array(rows[rule])
    return out


def main():
    r = scores()
    for rule in ("circle", "iou"):
        for metric, m in METRICS:
            print(f"\n{metric}, {rule} rule: mean +- sd over 3 seeds | gain over no zone, '*' = no overlap")
            print(f"{'scene':8s} " + " ".join(f"{n:>11s}" for n in RUNS) + " | "
                  + " ".join(f"{n:>9s}" for n in ZONED))
            passed = dict.fromkeys(ZONED, 0)
            for s in ALL_SCENES:
                v = {n: r[(s, n, rule)][:, m] for n in RUNS}
                cells = []
                for n in ZONED:
                    ok = v[n].min() > v["nozone"].max()
                    passed[n] += ok
                    cells.append(f"{v[n].mean() - v['nozone'].mean():+.2f}{'*' if ok else ' '}")
                print(f"{s:8s} " + " ".join(f"{v[n].mean():.2f}+-{v[n].std(ddof=1):.2f}".rjust(11) for n in RUNS)
                      + " | " + " ".join(c.rjust(9) for c in cells))
            print("scenes passing (>= 4 of 6 needed): " + ", ".join(f"{n} {passed[n]}/6" for n in ZONED))

    print("\nB2 against A, human zones, AP per seed (same input, only the training boxes differ)")
    for rule in ("circle", "iou"):
        for s in ALL_SCENES:
            a, b = r[(s, "A human", rule)][:, 0], r[(s, "B2 human", rule)][:, 0]
            rel = "B2 > A" if b.min() > a.max() else "A > B2" if a.min() > b.max() else "overlap"
            print(f"  {rule:6s} {s:8s} A {np.round(a, 2).tolist()}  B2 {np.round(b, 2).tolist()}  -> {rel}")
    print("\nfound = of 100 labelled insects, how many got a box; right = of 100 boxes, how many are on an insect; "
          "AP = one number over all cut-offs (no cut-off chosen).")


if __name__ == "__main__":
    main()
