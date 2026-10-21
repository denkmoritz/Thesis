#!/bin/bash
# Automatic chain: setup 1 (grey RGB frame + motion picture + zone) where the zone
# comes from our DVS detector instead of the human DVS annotation, each of the six
# scenes left out in turn, with training boxes A (human RGB labels) and B2 (motion
# blob in the detector zone, no human label).
#
# 12 jobs = 6 left-out scenes x 2 kinds of boxes, started as one array:
#
#   sbatch m3d.slurm                 # all 12
#   sbatch --array=0-1 m3d.slurm     # only left-out scene 1 (A, B2)
#
# Array index -> scene = index / 2, boxes = index % 2 (0 = A, 1 = B2).
# Recipe as m3.slurm (YOLOv7 from COCO, 50 epochs, 1600 px, batch 8, hyp.scratch.p5
# without colour augmentation, --zone-mode off). Dataset: step17_auto_chain.py.
# Predictions of the LAST epoch on the left-out scene: runs/test/m3d_<scene>_<A|B2>/labels.
#
#SBATCH --job-name=m3d
#SBATCH --array=0-11
#SBATCH --partition=gpuh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/tmp/mdenk/yolov7/logs/%x_%A_%a.out
#SBATCH --error=/scratch/tmp/mdenk/yolov7/logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdenk@uni-muenster.de

set -euo pipefail

SCENES=(1_l-l-l 2_l-h-l 3_m-h-h 4_m-m-h 5_h-l-h 6_h-h-h)
BOXES=(A B2)
SCENE="${SCENES[$((SLURM_ARRAY_TASK_ID / 2))]}"
LAB="${BOXES[$((SLURM_ARRAY_TASK_ID % 2))]}"
RUN="m3d_${SCENE}_${LAB}"
DATA_DIR=/scratch/tmp/mdenk/data/insect_dataset_m3d
DATA="$DATA_DIR/folds/${SCENE}_${LAB}.yaml"
HYP="data/hyp.m3d_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

echo "left-out scene : $SCENE"
echo "training boxes : $LAB"
echo "run name       : $RUN"

module purge
module load palma/2024a
module load GCC/13.3.0
module load OpenMPI/5.0.3
module load PyTorch/2.7.1-CUDA-12.6.0

cd /scratch/tmp/mdenk/yolov7

# Preflight
python3 apply_zone_hooks.py --check
[ -f test_zone.py ] || { echo "test_zone.py missing in $(pwd)" >&2; exit 1; }
[ -f "$DATA" ] || { echo "$DATA missing -- rsync dataset_m3 first" >&2; exit 1; }
for d in "runs/train/$RUN" "runs/test/$RUN"; do
  [ ! -e "$d" ] || { echo "$d exists -- remove it first" >&2; exit 1; }
done
first_img=$(head -1 "$DATA_DIR/folds/${SCENE}_${LAB}_train.txt")
[ -f "$first_img" ] || { echo "image missing: $first_img" >&2; exit 1; }
first_lbl=$(echo "$first_img" | sed 's#/images/#/labels/#; s#\.png$#.txt#')
[ -f "$first_lbl" ] || { echo "label missing: $first_lbl" >&2; exit 1; }
echo "train images: $(wc -l < "$DATA_DIR/folds/${SCENE}_${LAB}_train.txt"), test images: $(wc -l < "$DATA_DIR/folds/${SCENE}_${LAB}_test.txt")"

# hyp.scratch.p5 without colour augmentation
sed -E 's/^(hsv_[hsv]):[^#]*/\1: 0.0  /' data/hyp.scratch.p5.yaml > "$HYP"
grep -E '^hsv_' "$HYP"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python3 train.py \
  --weights weights/yolov7.pt \
  --cfg cfg/training/yolov7.yaml \
  --data "$DATA" \
  --hyp "$HYP" \
  --zone-mode off \
  --img-size 1600 1600 \
  --batch-size 8 \
  --epochs 50 \
  --workers 4 \
  --device 0 \
  --name "$RUN" \
  --project /scratch/tmp/mdenk/yolov7/runs/train

srun python3 test_zone.py \
  --weights "runs/train/$RUN/weights/last.pt" \
  --data "$DATA" \
  --zone-mode off \
  --img-size 1600 --batch-size 8 --device 0 \
  --name "$RUN"

rm -f "$HYP"
