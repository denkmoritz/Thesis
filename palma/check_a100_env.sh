#!/bin/bash
# Check whether the A100 nodes on PALMA can run our YOLOv7 training.
# Run on an A100 node (see the srun line in the chat), from /scratch/tmp/mdenk/yolov7:
#
#   srun --partition=gpua100 --time=00:10:00 --cpus-per-task=1 --mem=4G --gres=gpu:1 bash check_a100_env.sh
#
# Prints: which modules have to be loaded before PyTorch/2.1.2-CUDA-12.1.1, then loads
# them and checks every Python package YOLOv7 and our scripts import.

MOD=PyTorch/2.1.2-CUDA-12.1.1
module purge
out=$(module spider "$MOD" 2>&1)
echo "=== module spider $MOD"
echo "$out" | sed -n '/You will need to load/,/^$/p;/load all module/,/^ *$/p' | head -15
pre=$(echo "$out" | grep -A2 "load all module" | grep -E '^ +[A-Za-z]' | head -1)
echo "=== loading: $pre $MOD"
module load $pre "$MOD" || { echo "LOAD FAILED"; exit 1; }
module list 2>&1 | tail -5

python3 - <<'EOF'
import sys
print("=== python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "cuda available", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
for m in ["numpy", "cv2", "yaml", "scipy", "matplotlib", "tqdm", "seaborn", "pandas", "PIL", "torchvision"]:
    try:
        mod = __import__(m)
        print(f"{m:12s} ok  {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"{m:12s} MISSING  {e}")
EOF

echo "=== zone hooks and YOLOv7 import"
python3 apply_zone_hooks.py --check 2>&1 | tail -2
python3 -c "import models.yolo, utils.datasets, zone_fusion; print('yolov7 modules import ok')" 2>&1 | tail -2
