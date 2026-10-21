"""
Save the predictions of a trained model with the zone applied -- runs on PALMA.

YOLOv7's own test.py cannot apply a zone: the zone hooks patch train.py and the
dataloader, not test.py. This wrapper builds the validation dataloader exactly as
train.py does (same hook, same zone PNGs) and hands it to YOLOv7's test function,
so inference sees the same input as training. With --zone-mode off it does the
same as `test.py --save-txt --save-conf` and must print the same numbers.

Nothing in the YOLOv7 checkout is changed. Copy this file into the repo root
(/scratch/tmp/mdenk/yolov7/) and run it there:

    python3 test_zone.py --weights runs/train/<run>/weights/last.pt \
        --data <data.yaml> --zone-mode multiply --name <name>

Output: runs/test/<name>/labels/{stem}.txt  (class cx cy w h conf, normalised)
and one line "RESULT <name> images labels P R mAP@.5 mAP@.5:.95".
"""

import argparse
from pathlib import Path

import yaml

import test as yolo_test
from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import check_img_size, colorstr
from utils.torch_utils import select_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--zone-mode", default="off", choices=["off", "multiply"])
    ap.add_argument("--task", default="val", help="key in the data yaml: train, val or test")
    ap.add_argument("--img-size", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--conf-thres", type=float, default=0.001)
    ap.add_argument("--iou-thres", type=float, default=0.65, help="IoU threshold of the NMS (test.py default)")
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/test")
    ap.add_argument("--name", required=True)
    ap.add_argument("--single-cls", action="store_true")
    opt = ap.parse_args()

    save_dir = Path(opt.project) / opt.name
    if save_dir.exists():
        raise SystemExit(f"{save_dir} exists -- remove it or choose another --name")
    (save_dir / "labels").mkdir(parents=True)

    device = select_device(opt.device, batch_size=opt.batch_size)
    model = attempt_load(opt.weights, map_location=device)
    stride = max(int(model.stride.max()), 32)
    imgsz = check_img_size(opt.img_size, s=stride)
    with open(opt.data) as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # same call as test.py, plus the zone mode of the dataloader hook
    dataloader = create_dataloader(data[opt.task], imgsz, opt.batch_size, stride, opt, pad=0.5, rect=True,
                                   prefix=colorstr(f"{opt.task}: "), zone_mode=opt.zone_mode)[0]
    results, _, _ = yolo_test.test(data, batch_size=opt.batch_size, imgsz=imgsz, conf_thres=opt.conf_thres,
                                   iou_thres=opt.iou_thres, single_cls=opt.single_cls, model=model,
                                   dataloader=dataloader, save_dir=save_dir, save_txt=True, save_conf=True,
                                   plots=False)
    p, r, map50, map5095 = results[:4]
    n_files = len(list((save_dir / "labels").glob("*.txt")))
    n_labels = int(sum(len(l) for l in dataloader.dataset.labels))
    print(f"RESULT {opt.name} images {len(dataloader.dataset)} labels {n_labels} "
          f"P {p:.3f} R {r:.3f} mAP@.5 {map50:.4f} mAP@.5:.95 {map5095:.4f}  "
          f"zone {opt.zone_mode}  prediction files {n_files}")


if __name__ == "__main__":
    main()
