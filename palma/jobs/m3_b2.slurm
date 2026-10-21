#!/bin/bash
# Setup 1 with training boxes B2: label-free like B, but the box is the motion blob
# outlined at 25 % of the peak instead of 50 % (step16_b_box_size.py chose this rule
# leave-one-scene-out; step16b_write_b2.py wrote the boxes). Images as in m3.slurm.
#
# 6 jobs = 6 left-out scenes, started as one array:
#
#   sbatch m3_b2.slurm
#
# Array index = left-out scene (0 = 1_l-l-l ... 5 = 6_h-h-h).
# Everything else as in m3.slurm (YOLOv7 from COCO, 50 epochs, 1600 px, batch 8,
# hyp.scratch.p5 without colour augmentation, --zone-mode off). Predictions of the
# LAST epoch on the left-out scene: runs/test/m3_<scene>_B2/labels.
#
#SBATCH --job-name=m3b2
#SBATCH --array=0-5
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
BOXES=(B2)
SCENE="${SCENES[$((SLURM_ARRAY_TASK_ID / 1))]}"
LAB="${BOXES[$((SLURM_ARRAY_TASK_ID % 1))]}"
RUN="m3_${SCENE}_${LAB}"
DATA_DIR=/scratch/tmp/mdenk/data/insect_dataset_m3
DATA="$DATA_DIR/folds/${SCENE}_${LAB}.yaml"
HYP="data/hyp.m3_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

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
