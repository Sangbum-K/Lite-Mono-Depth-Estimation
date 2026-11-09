"""Download KITTI raw one archive at a time and keep only what the Eigen-Zhou / Eigen splits use.

- RGB: every frame referenced by train/val (t-1, t, t+1, same camera) and test, converted png -> jpg
  with Monodepth2's settings (quality 92, 4:2:0 chroma). Written to <kitti_out>, the --data_path for training.
- LiDAR: velodyne .bin of val/test frames only, plus calib, written to <velo_out>. Kept out of <kitti_out> on
  purpose: the dataloader loads LiDAR for every training sample if it finds it there (only used for logging).
- Calib txt files go to both roots.

Each archive is deleted after extraction and marked done, so the script can be re-run after a disconnect.
It ends by checking that every needed file exists and exits non-zero if anything is missing.

    python scripts/prepare_kitti.py --kitti_out /content/kitti_data --velo_out /content/kitti_velo
"""
import argparse
import io
import os
import subprocess
import sys
import zipfile
from multiprocessing import Pool

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAM = {"l": "image_02", "r": "image_03"}


def read_split(path):
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def needed_files(split_dir):
    """Zip member paths (png) needed for RGB, and (folder, frame) pairs needed for LiDAR."""
    rgb, velo = set(), set()
    for name, offsets in [("train", (-1, 0, 1)), ("val", (-1, 0, 1)), ("test", (0,))]:
        for folder, frame, side in read_split(os.path.join(split_dir, name + "_files.txt")):
            for d in offsets:
                rgb.add("{}/{}/data/{:010d}.png".format(folder, CAM[side], int(frame) + d))
            if name != "train":
                velo.add("{}/velodyne_points/data/{:010d}.bin".format(folder, int(frame)))
    return rgb, velo


def to_jpg(args):
    zip_path, member, out_root = args
    out = os.path.join(out_root, member[:-4] + ".jpg")
    if os.path.isfile(out):
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        img = Image.open(io.BytesIO(z.read(member)))
        img.save(out + ".tmp", "JPEG", quality=92, subsampling=2)  # = convert -quality 92 -sampling-factor 2x2,1x1,1x1
    os.replace(out + ".tmp", out)


def extract(z, member, out_root):
    out = os.path.join(out_root, member)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out + ".tmp", "wb") as f:
        f.write(z.read(member))
    os.replace(out + ".tmp", out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kitti_out", required=True)
    p.add_argument("--velo_out", help="omit to handle RGB only, e.g. to re-check a restored copy")
    p.add_argument("--tmp", default="/content/kitti_zip")
    p.add_argument("--archives", default=os.path.join(ROOT, "data", "kitti_archives_to_download.txt"))
    p.add_argument("--splits", default=os.path.join(ROOT, "data", "splits"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    a = p.parse_args()

    rgb, velo = needed_files(a.splits)
    if a.velo_out is None:
        velo = set()
    roots = [r for r in (a.kitti_out, a.velo_out) if r]
    drives = {m.split("/")[1] for m in rgb}
    done_dir = os.path.join(a.kitti_out, ".done")
    os.makedirs(done_dir, exist_ok=True)
    os.makedirs(a.tmp, exist_ok=True)
    print("need {} RGB frames, {} LiDAR frames, {} drives".format(len(rgb), len(velo), len(drives)))

    with open(a.archives) as f:
        urls = [u.strip() for u in f if u.strip()]
    for i, url in enumerate(urls, 1):
        name = os.path.basename(url)
        marker = os.path.join(done_dir, name)
        is_calib = name.endswith("_calib.zip")
        if os.path.isfile(marker) or (not is_calib and name[:-4] not in drives):
            continue
        zip_path = os.path.join(a.tmp, name)
        print("[{}/{}] {}".format(i, len(urls), name), flush=True)
        # shortcut: downloads run one after another; overlap with extraction if the one-time prep is too slow
        subprocess.run(["curl", "-fL", "--retry", "5", "-C", "-", "-sS", "-o", zip_path, url], check=True)
        with zipfile.ZipFile(zip_path) as z:
            members = set(z.namelist())
            if is_calib:
                for m in members:
                    if m.endswith(".txt"):
                        for r in roots:
                            extract(z, m, r)
            else:
                for m in members & velo:
                    extract(z, m, a.velo_out)
        if not is_calib:
            jobs = [(zip_path, m, a.kitti_out) for m in sorted(members & rgb)]
            with Pool(a.workers) as pool:
                pool.map(to_jpg, jobs, chunksize=64)
        os.remove(zip_path)
        open(marker, "w").close()

    missing_rgb = [m[:-4] + ".jpg" for m in rgb if not os.path.isfile(os.path.join(a.kitti_out, m[:-4] + ".jpg"))]
    missing_velo = [m for m in velo if not os.path.isfile(os.path.join(a.velo_out, m))]
    dates = {m.split("/")[0] for m in rgb}
    missing_calib = [os.path.join(r, d, c) for r in roots for d in dates
                     for c in ("calib_cam_to_cam.txt", "calib_velo_to_cam.txt")
                     if not os.path.isfile(os.path.join(r, d, c))]
    print("check: RGB {}/{} present, LiDAR {}/{} present, calib missing {}".format(
        len(rgb) - len(missing_rgb), len(rgb), len(velo) - len(missing_velo), len(velo), len(missing_calib)))
    for m in sorted(missing_rgb)[:20] + sorted(missing_velo)[:20] + missing_calib[:20]:
        print("  missing:", m)
    sys.exit(1 if missing_rgb or missing_velo or missing_calib else 0)


if __name__ == "__main__":
    main()
