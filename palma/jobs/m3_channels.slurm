#!/bin/bash
# Channel test for setup 1: the same training as m3.slurm with boxes A (human RGB
# labels), but with one of the three input channels set to zero:
#   nozone    grey frame + motion   (no DVS information)
#   nomotion  grey frame + zone     (single frame + DVS zone)
#   nogrey    motion + zone         (no picture of the scene)
#
# 18 jobs = 6 left-out scenes x 3 variants, started as one array:
#
#   sbatch m3_channels.slurm                 # all 18
#   sbatch --array=0-2 m3_channels.slurm     # only left-out scene 1
#
# Array index -> scene = index / 3, variant = index % 3 (0 nozone, 1 nomotion, 2 nogrey).
# Everything else as in m3.slurm (YOLOv7 from COCO, 50 epochs, 1600 px, batch 8,
# hyp.scratch.p5 without colour augmentation, --zone-mode off).
# Dataset: step15c_channel_dataset.py. After training, the predictions of the LAST
# epoch on the left-out scene are saved (runs/test/m3_<scene>_<variant>/labels).
#
#SBATCH --job-name=m3ch
#SBATCH --array=0-17
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
BOXES=(nozone nomotion nogrey)
SCENE="${SCENES[$((SLURM_ARRAY_TASK_ID / 3))]}"
LAB="${BOXES[$((SLURM_ARRAY_TASK_ID % 3))]}"
RUN="m3_${SCENE}_${LAB}"
DATA_DIR=/scratch/tmp/mdenk/data/insect_dataset_m3
DATA="$DATA_DIR/folds/${SCENE}_${LAB}.yaml"
HYP="data/hyp.m3_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

echo "left-out scene : $SCENE"
echo "variant        : $LAB"
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
