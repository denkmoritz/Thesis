# PALMA (GPU cluster)

Jobs run from `/scratch/tmp/mdenk/yolov7` (YOLOv7 repo). Datasets live in
`/scratch/tmp/mdenk/data/insect_dataset_{dvs,m3,m3d}`.

## Once: our additions to YOLOv7

```bash
scp palma/yolov7/*.py palma:/scratch/tmp/mdenk/yolov7/
ssh palma "cd /scratch/tmp/mdenk/yolov7 && python3 apply_zone_hooks.py --check"
```

## Run a job

```bash
scp palma/jobs/<job>.slurm palma:/scratch/tmp/mdenk/yolov7/
ssh palma
cd /scratch/tmp/mdenk/yolov7 && sbatch <job>.slurm && squeue -u mdenk
```

| Job | Runs | What |
|---|---|---|
| `dvs.slurm` | 6 | own DVS detector, LOSO |
| `m3.slurm` | 12 | setup 1, human zones, boxes A / B (old rule) |
| `m3_b2.slurm` | 6 | setup 1, human zones, boxes B2 |
| `m3_channels.slurm` | 18 | channel test (nozone / nomotion / nogrey) |
| `m3d.slurm` | 12 | automatic chain (detector zones), A / B2 |
| `m3_seeds.slurm` | 24 | seeds 1, 2 of A / B2 with human zones |
| `m3_nozone_seeds.slurm` | 12 | seeds 1, 2 of nozone |
| `m3d_seeds.slurm` | 24 | seeds 1, 2 of the automatic chain |

## Check and fetch results

```bash
sacct -j <JOBID> --format=JobID,State,Elapsed | grep -v "\."
for d in runs/test/<pattern>; do echo "$d $(ls $d/labels | wc -l)"; done
# on the Mac, from the repo root:
rsync -av --include='<pattern>/' --include='<pattern>/labels/***' --exclude='*' \
  palma:/scratch/tmp/mdenk/yolov7/runs/test/ outputs/m3/      # m3d_* runs → outputs/m3d/
```

If jobs wait with `ReqNodeNotAvail` or `Priority`: `sinfo -p gpuh200 -N`,
`sinfo -R -p gpuh200`, `scontrol show reservation`. A100 nodes only have PyTorch 2.1.2
(`check_a100_env.sh`, untested); keep all seeds of one comparison on the same GPU type.
