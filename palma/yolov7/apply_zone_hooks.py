"""
Apply the zone-fusion hooks to a WongKinYiu/yolov7 checkout.

This is yolov7_zone_fusion.MD sections 1-4 as a script, so the Docker test
image and the PALMA checkout receive byte-identical patches instead of two
hand edits that can drift apart. Run it from the YOLOv7 repo root, next to
zone_fusion.py (the copied yolov7_zone_fusion.py):

    python3 apply_zone_hooks.py           # patch in place, then verify
    python3 apply_zone_hooks.py --check   # verify only, change nothing

Every edit is an exact text replacement that must match exactly once, so an
unexpected checkout (e.g. a differently patched PALMA copy) stops the script
with the name of the failing hook instead of being half-patched. Re-running
it on an already patched checkout changes nothing.
"""

import argparse
import os
import re
import sys

MARK = "# zone-fusion hook"

LOAD_IMAGE_RETURN = (
    "        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized\n"
)

GETITEM_CONVERT = (
    "        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416\n"
    "        img = np.ascontiguousarray(img)\n"
    "\n"
    "        return torch.from_numpy(img), labels_out"
)

AUGMENT_HSV_DEF = "def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):\n"


def replace_once(text, old, new, name):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"[{name}] expected exactly 1 match, found {count}")
    return text.replace(old, new)


def patch_datasets(text):
    # 1. Fuse the zone where the image is first read, so every later
    #    augmentation moves zone and pixels together. Only the uncached
    #    branch needs it: the cache is filled by calling load_image itself.
    text = replace_once(
        text,
        LOAD_IMAGE_RETURN,
        f"        if getattr(self, 'zone_mode', 'off') != 'off':  {MARK}\n"
        "            from zone_fusion import fuse, load_zone\n"
        "            zone = load_zone(self.img_files[index], img.shape[:2])\n"
        "            img = fuse(img, zone, self.zone_mode)\n" + LOAD_IMAGE_RETURN,
        "load_image",
    )

    # 2. Accept zone_mode on the dataset. It is stored on the first line of
    #    __init__ so it is already set when --cache-images fills the cache.
    text = replace_once(
        text,
        "cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=''):\n"
        "        self.img_size = img_size\n",
        "cache_images=False, single_cls=False, stride=32, pad=0.0, prefix='',\n"
        f"                 zone_mode='off'):  {MARK}\n"
        "        self.zone_mode = zone_mode\n"
        "        self.img_size = img_size\n",
        "LoadImagesAndLabels.__init__",
    )

    # 3. Pass zone_mode through create_dataloader.
    match = re.search(r"def create_dataloader\((.*?)\):", text, re.S)
    if not match:
        raise SystemExit("[create_dataloader] definition not found")
    signature = match.group(0)
    text = text.replace(
        signature, signature[:-2] + f", zone_mode='off'):  {MARK}", 1
    )
    start = text.index("def create_dataloader(")
    end = text.index("prefix=prefix)", start)
    text = (
        text[:end]
        + "prefix=prefix, zone_mode=zone_mode)"
        + text[end + len("prefix=prefix)"):]
    )

    # 4. Reverse only the colour channels; img[:, :, ::-1] would move the
    #    zone of a concat input into channel 0.
    text = replace_once(
        text,
        GETITEM_CONVERT,
        f"        from zone_fusion import to_rgb_chw  {MARK}\n"
        "        img = to_rgb_chw(img)  # BGR(+zone) to RGB(+zone), HWC to CHW\n"
        "        img = np.ascontiguousarray(img)\n"
        "\n"
        "        return torch.from_numpy(img), labels_out",
        "__getitem__ channel order",
    )

    # 5. HSV augmentation on the colour channels only, so the concat run
    #    keeps the same colour augmentation as the other two runs.
    text = replace_once(
        text,
        AUGMENT_HSV_DEF,
        AUGMENT_HSV_DEF
        + f"    if img.shape[2] > 3:  {MARK}\n"
        "        # cvtColor cannot write into a strided view, so work on a copy\n"
        "        colour = np.ascontiguousarray(img[..., :3])\n"
        "        augment_hsv(colour, hgain, sgain, vgain)\n"
        "        img[..., :3] = colour\n"
        "        return\n",
        "augment_hsv",
    )
    return text


def patch_train(text):
    text = replace_once(
        text,
        "from utils.plots import",
        f"from zone_fusion import input_channels  {MARK}\nfrom utils.plots import",
        "train.py import",
    )
    count = text.count("ch=3, nc=nc")
    if count != 2:
        raise SystemExit(f"[train.py Model(ch=3)] expected 2 matches, found {count}")
    text = text.replace("ch=3, nc=nc", "ch=input_channels(opt.zone_mode), nc=nc")
    text = replace_once(
        text,
        "quad=opt.quad, prefix=colorstr('train: '))",
        "quad=opt.quad, prefix=colorstr('train: '),\n"
        f"                                            zone_mode=opt.zone_mode)  {MARK}",
        "train.py train dataloader",
    )
    text = replace_once(
        text,
        "pad=0.5, prefix=colorstr('val: '))[0]",
        "pad=0.5, prefix=colorstr('val: '),\n"
        f"                                       zone_mode=opt.zone_mode)[0]  {MARK}",
        "train.py val dataloader",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--weights'",
        "    parser.add_argument('--zone-mode', type=str, default='off',\n"
        "                        choices=['off', 'multiply', 'concat'],\n"
        f"                        help='epipolar zone fusion mode')  {MARK}\n"
        "    parser.add_argument('--weights'",
        "train.py --zone-mode flag",
    )
    # Batch previews assume 3 channels; plot the colour channels only.
    text = replace_once(
        text,
        "args=(imgs, targets, paths, f)",
        "args=(imgs[:, :3], targets, paths, f)",
        "train.py plot_images",
    )
    return text


def patch_test(text):
    text = replace_once(
        text,
        "args=(img, targets, paths, f, names)",
        "args=(img[:, :3], targets, paths, f, names)",
        "test.py plot_images (labels)",
    )
    text = replace_once(
        text,
        "args=(img, output_to_target(out), paths, f, names)",
        "args=(img[:, :3], output_to_target(out), paths, f, names)",
        "test.py plot_images (predictions)",
    )
    return text


def write_zone_hyp(src="data/hyp.scratch.p5.yaml", dst="data/hyp.zone.yaml"):
    """hyp.scratch.p5.yaml with every multi-image augmentation switched off.

    Mosaic, mixup and paste-in all combine several frames, so the zone would
    no longer belong to the pixels it sits on (yolov7_zone_fusion.MD
    section 4). paste_in runs even with mosaic off, so it has to be zeroed
    separately.
    """
    with open(src) as handle:
        text = handle.read()
    for key in ("mosaic", "mixup", "paste_in"):
        text, count = re.subn(
            rf"^{key}:[^\n]*", f"{key}: 0.0  # off for zone runs", text, flags=re.M
        )
        if count != 1:
            raise SystemExit(f"[{dst}] expected one '{key}:' line in {src}, found {count}")
    with open(dst, "w") as handle:
        handle.write(text)


PATCHES = (
    ("utils/datasets.py", patch_datasets, 5),
    ("train.py", patch_train, 4),
)


def check():
    problems = []
    if not os.path.exists("zone_fusion.py"):
        problems.append("zone_fusion.py missing (copy src/yolov7_zone_fusion.py here)")
    else:
        with open("zone_fusion.py") as handle:
            module = handle.read()
        # Guards against copying an old version of the module.
        for name in ("def to_rgb_chw", "missing_ok=False", "torch.zeros_like"):
            if name not in module:
                problems.append(f"zone_fusion.py is outdated (no '{name}')")
    for path, _, expected in PATCHES:
        with open(path) as handle:
            found = handle.read().count(MARK)
        if found != expected:
            problems.append(f"{path}: {found}/{expected} hooks present")
    with open("test.py") as handle:
        if handle.read().count("img[:, :3]") != 2:
            problems.append("test.py: plot_images not patched")
    if not os.path.exists("data/hyp.zone.yaml"):
        problems.append("data/hyp.zone.yaml missing")
    else:
        with open("data/hyp.zone.yaml") as handle:
            hyp = handle.read()
        for key in ("mosaic", "mixup", "paste_in"):
            if not re.search(rf"^{key}: 0\.0\b", hyp, re.M):
                problems.append(f"data/hyp.zone.yaml: {key} is not 0.0")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify the checkout is patched; change nothing")
    args = parser.parse_args()

    if not os.path.exists("train.py") or not os.path.isdir("utils"):
        raise SystemExit("Run this from the YOLOv7 repo root.")

    if not args.check:
        for path, patch, _ in PATCHES + (("test.py", patch_test, 0),):
            with open(path) as handle:
                text = handle.read()
            if MARK in text or "img[:, :3]" in text:
                print(f"{path}: already patched, left unchanged")
                continue
            with open(path, "w") as handle:
                handle.write(patch(text))
            print(f"{path}: patched")
        write_zone_hyp()
        print("data/hyp.zone.yaml: written")

    problems = check()
    for problem in problems:
        print(f"FAIL  {problem}")
    if problems:
        sys.exit(1)
    print("OK    zone hooks, zone_fusion.py and hyp.zone.yaml are in place")


if __name__ == "__main__":
    main()
