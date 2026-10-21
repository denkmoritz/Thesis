#!/bin/bash
# Our own DVS detector: YOLOv7 on DVS frames, each of the six scenes left out in
# turn (as Gebauer et al.). Then the model runs on EVERY frame of the left-out
# scene's DVS video; these detections become the zones later.
#
# 6 jobs, started as one array:
#
#   sbatch dvs.slurm                 # all 6
#   sbatch --array=0 dvs.slurm       # only left-out scene 1
#
# Array index = left-out scene (0 = 1_l-l-l ... 5 = 6_h-h-h).
# Recipe: YOLOv7 from COCO (yolov7.pt), 50 epochs, 1280 px (native DVS width),
# batch 8, hyp.scratch.p5 unchanged. Dataset: step12_dvs_dataset.py.
# Outputs:
#   runs/train/dvs_<scene>/weights/last.pt
#   runs/test/dvs_<scene>/labels        predictions on the labelled frames of the scene
#   runs/detect/dvs_<scene>/labels      predictions on all video frames:
#                                       <scene>_dvs_<n>.txt, n = video frame + 1 (YOLOv7 counts from 1)
#
#SBATCH --job-name=dvs
#SBATCH --array=0-5
#SBATCH --partition=gpuh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/tmp/mdenk/yolov7/logs/%x_%A_%a.out
#SBATCH --error=/scratch/tmp/mdenk/yolov7/logs/%x_%A_%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mdenk@uni-muenster.de

set -euo pipefail

SCENES=(1_l-l-l 2_l-h-l 3_m-h-h 4_m-m-h 5_h-l-h 6_h-h-h)
SCENE="${SCENES[$SLURM_ARRAY_TASK_ID]}"
RUN="dvs_${SCENE}"
DATA_DIR=/scratch/tmp/mdenk/data/insect_dataset_dvs
DATA="$DATA_DIR/folds/${SCENE}.yaml"
VIDEO="/scratch/tmp/mdenk/data/dvs_videos/${SCENE}_dvs.mp4"

echo "left-out scene : $SCENE"
echo "run name       : $RUN"

module purge
module load palma/2024a
module load GCC/13.3.0
module load OpenMPI/5.0.3
module load PyTorch/2.7.1-CUDA-12.6.0

cd /scratch/tmp/mdenk/yolov7

# Preflight
[ -f test_zone.py ] || { echo "test_zone.py missing in $(pwd)" >&2; exit 1; }
[ -f "$DATA" ] || { echo "$DATA missing -- rsync dataset_dvs first" >&2; exit 1; }
[ -f "$VIDEO" ] || { echo "$VIDEO missing -- copy the DVS videos first" >&2; exit 1; }
for d in "runs/train/$RUN" "runs/test/$RUN" "runs/detect/$RUN"; do
  [ ! -e "$d" ] || { echo "$d exists -- remove it first" >&2; exit 1; }
done
echo "train images: $(wc -l < "$DATA_DIR/folds/${SCENE}_train.txt"), val images: $(wc -l < "$DATA_DIR/folds/${SCENE}_val.txt")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun python3 train.py \
  --weights weights/yolov7.pt \
  --cfg cfg/training/yolov7.yaml \
  --data "$DATA" \
  --hyp data/hyp.scratch.p5.yaml \
  --img-size 1280 1280 \
  --batch-size 8 \
  --epochs 50 \
  --workers 4 \
  --device 0 \
  --name "$RUN" \
  --project /scratch/tmp/mdenk/yolov7/runs/train

# predictions on the labelled frames of the left-out scene (one RESULT line)
srun python3 test_zone.py \
  --weights "runs/train/$RUN/weights/last.pt" \
  --data "$DATA" \
  --zone-mode off \
  --img-size 1280 --batch-size 8 --device 0 \
  --name "$RUN"

# predictions on every frame of the left-out scene's video (per-frame output to a file, not the log)
srun python3 detect.py \
  --weights "runs/train/$RUN/weights/last.pt" \
  --source "$VIDEO" \
  --img-size 1280 --conf-thres 0.01 --iou-thres 0.45 \
  --save-txt --save-conf --nosave --no-trace --device 0 \
  --name "$RUN" --project runs/detect > "logs/detect_${RUN}.txt" 2>&1
echo "DETECT $RUN video frames $(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "$VIDEO" 2>/dev/null || echo '?') frames with detections $(ls "runs/detect/$RUN/labels" | wc -l)"
