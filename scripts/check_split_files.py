"""Check that every sample in a Monodepth2-style KITTI split exists on disk.

For each line "<folder> <frame_idx> <side>", checks the target frame and the
neighbouring frames (default -1, +1) used by self-supervised training.

Usage:
    python scripts/check_split_files.py --data_path data/raw/kitti \
        --split_file data/splits/eigen_zhou/train_files.txt --ext .png
"""
import argparse
import os
import sys

SIDE_MAP = {"l": "image_02", "r": "image_03", "2": "image_02", "3": "image_03"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", required=True)
    p.add_argument("--split_file", required=True)
    p.add_argument("--ext", default=".png")
    p.add_argument("--frame_ids", type=int, nargs="+", default=[0, -1, 1])
    args = p.parse_args()

    with open(args.split_file) as f:
        lines = [l.split() for l in f.read().splitlines() if l.strip()]

    missing = []
    for folder, idx, side in lines:
        for fid in args.frame_ids:
            path = os.path.join(args.data_path, folder, SIDE_MAP[side], "data",
                                f"{int(idx) + fid:010d}{args.ext}")
            if not os.path.isfile(path):
                missing.append(path)

    print(f"{args.split_file}: {len(lines)} samples, "
          f"{len(lines) * len(args.frame_ids)} frames checked, {len(missing)} missing")
    for m in missing[:20]:
        print("  MISSING", m)
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
