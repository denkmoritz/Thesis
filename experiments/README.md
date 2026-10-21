# Experiments

Every number in the thesis comes from one of these. All results per scene (1–6),
leave-one-scene-out. "found" = of 100 labelled insects, how many got a box; "right" =
of 100 boxes, how many are on an insect (both at the best cut-off of the scored scene,
so optimistic); AP needs no cut-off. Circle rule = box centre near the label; IoU rule
= box also the right size.

| ID | Question | Build / PALMA job | Score | Result | Thesis |
|---|---|---|---|---|---|
| E01 | Is the own DVS detector a fair stand-in for the paper's? | `scripts/06`, `palma/jobs/dvs.slurm` | `scripts/07` | F1 mean 49.6 vs paper 46.1 | Results 1 |
| E02 | Which input channel does the detector use? | `scripts/08`, `m3.slurm` + `m3_channels.slurm` | `score_setup1.py` | motion essential; zone +0.09–0.29 (1 seed) | Results 2 |
| E03 | Does the system work on unseen scenes without human DVS input? | `scripts/09`, `m3d.slurm` | `score_auto_chain.py` | AP circle 0.55–0.89 (B2) vs RGB-only single frame ≈ 0 | Results 2 |
| E04 | Can DVS + motion replace human RGB training labels (B2)? | `scripts/08/09`, `m3_b2.slurm`, `m3d.slurm` | `zone_gain.py` | B2 ≈ A within seed spread, more stable | Results 2 |
| E05 | Does the zone help, judged against seed noise? | `m3_seeds.slurm`, `m3d_seeds.slurm`, `m3_nozone_seeds.slurm` | `scripts/10` | human zones: A 4/6, B2 6/6 pass; detector zones: 0/6, 1/6 | Results 2 |
| E06 | What does RGB add over DVS alone? | – | `scripts/11` | same insects found, RGB position added | Results 2 |
| E07 | Why do detector zones miss labels? | – | `scripts/12` | recall 76 %, placement 8 % (scene 5), size 1 % | Discussion |
| E08 | Does tracking improve DVS detection? | – | `scripts/13` | F1 +13–15 on hand-placed frames in scenes 3, 4, 6 | Results 3 |
| – | Pilot P0: single-frame zone multiply (dead end) | `archive/pilot_p0/` | `archive/pilot_p0/` | AP ≈ 0 in scenes 3–5 → motion added | Results 4 / appendix |

Full numbers: `results/logs/` (printed output of the score scripts).
