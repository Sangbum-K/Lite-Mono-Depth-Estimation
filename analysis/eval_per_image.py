"""Per-image depth evaluation with the exact protocol of vendor/Lite-Mono/evaluate_depth.py
(Eigen crop, 1e-3..80 m, per-image median scaling, same 7 metrics), for any split and many checkpoints.

Why not the official script: it only prints the mean, and loads every GT map densely
(4,424 val maps ~ 8 GB). Here GT is stored sparsely and errors are computed per batch.

1) GT, once (needs LiDAR + calib, see scripts/prepare_kitti.py --velo_out):
   python analysis/eval_per_image.py build-gt --split_file data/splits/val_files.txt \
       --velo_root /content/kitti_velo --out val_gt.npz
2) Evaluate one or more weight folders in one pass over the images:
   python analysis/eval_per_image.py eval --split_file data/splits/val_files.txt --gt val_gt.npz \
       --data_path /content/kitti_data --weights run/models/weights_0 run/models/weights_1 ...
   Writes per_image_<split>.csv into each weight folder and prints the mean row per folder.

Self-check: on test_files.txt with the official checkpoint, the mean must equal evaluate_depth.py.
"""
import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "Lite-Mono"))

MIN_DEPTH, MAX_DEPTH = 1e-3, 80
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def pack(maps):
    """Dense GT maps (any iterable, consumed one at a time) -> sparse arrays (only LiDAR pixels are non-zero)."""
    shapes, idx, depth = [], [], []
    for m in maps:
        i = np.flatnonzero(m).astype(np.int32)
        shapes.append(m.shape)
        idx.append(i)
        depth.append(m.ravel()[i].astype(np.float32))
    return {"shapes": np.array(shapes, np.int32),
            "offsets": np.cumsum([0] + [len(i) for i in idx]).astype(np.int64),
            "idx": np.concatenate(idx),
            "depth": np.concatenate(depth)}


def unpack(g, i):
    m = np.zeros(g["shapes"][i], np.float32)
    s, e = g["offsets"][i], g["offsets"][i + 1]
    m.flat[g["idx"][s:e]] = g["depth"][s:e]
    return m


def errors_one(gt, pred_disp, compute_errors, resize):
    """Same steps as evaluate_depth.py for eval_split == 'eigen'."""
    h, w = gt.shape
    pred = 1 / resize(pred_disp, (w, h))
    mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
    crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    mask = np.logical_and(mask, crop_mask)
    pred, gt = pred[mask], gt[mask]
    ratio = np.median(gt) / np.median(pred)
    pred = pred * ratio
    pred[pred < MIN_DEPTH] = MIN_DEPTH
    pred[pred > MAX_DEPTH] = MAX_DEPTH
    return list(compute_errors(gt, pred)) + [ratio, int(mask.sum())]


def build_gt(a):
    from kitti_utils import generate_depth_map

    def maps():  # one dense map in memory at a time (all 4,424 val maps would need ~8 GB)
        for folder, frame, side in (l.split() for l in open(a.split_file) if l.strip()):
            velo = os.path.join(a.velo_root, folder, "velodyne_points/data/{:010d}.bin".format(int(frame)))
            # official export_gt_depth.py: cam 2 with vel_depth=True; the matching camera for right-image lines
            cam = {"l": 2, "r": 3}[side]
            yield generate_depth_map(os.path.join(a.velo_root, folder.split("/")[0]), velo, cam, True)

    g = pack(maps())
    np.savez_compressed(a.out, **g)
    print("saved {} GT maps to {}".format(len(g["shapes"]), a.out))


def evaluate(a):
    import cv2
    import torch
    from torch.utils.data import DataLoader
    import datasets
    import networks
    from layers import disp_to_depth
    from evaluate_depth import compute_errors

    lines = [l.strip() for l in open(a.split_file) if l.strip()]
    g = dict(np.load(a.gt))
    assert len(g["shapes"]) == len(lines), "GT was built from a different split file"

    models = []
    for w in a.weights:
        enc_dict = torch.load(os.path.join(w, "encoder.pth"), map_location="cpu")
        dec_dict = torch.load(os.path.join(w, "depth.pth"), map_location="cpu")
        enc = networks.LiteMono(model=a.model, height=enc_dict["height"], width=enc_dict["width"])
        dec = networks.DepthDecoder(enc.num_ch_enc, scales=range(3))
        # same key filtering as the official evaluator
        enc.load_state_dict({k: v for k, v in enc_dict.items() if k in enc.state_dict()})
        dec.load_state_dict({k: v for k, v in dec_dict.items() if k in dec.state_dict()})
        models.append((enc.cuda().eval(), dec.cuda().eval()))
        size = (enc_dict["height"], enc_dict["width"])
        assert len(models) == 1 or size == prev_size, "all weights must share the input size"
        prev_size = size

    dataset = datasets.KITTIRAWDataset(a.data_path, lines, size[0], size[1], [0], 4, is_train=False)
    loader = DataLoader(dataset, 16, shuffle=False, num_workers=a.num_workers, pin_memory=True)
    rows = [[] for _ in models]
    n = 0
    with torch.no_grad():
        for data in loader:
            x = data[("color", 0, 0)].cuda()
            gts = [unpack(g, n + j) for j in range(x.shape[0])]
            for m, (enc, dec) in enumerate(models):
                disp, _ = disp_to_depth(dec(enc(x))[("disp", 0)], a.min_depth, a.max_depth)
                for j, d in enumerate(disp.cpu()[:, 0].numpy()):
                    rows[m].append([lines[n + j]] + errors_one(gts[j], d, compute_errors, cv2.resize))
            n += x.shape[0]

    name = os.path.splitext(os.path.basename(a.split_file))[0]
    print(("{:>40} | " + "{:>8} | " * 7).format("weights", *METRICS))
    for w, r in zip(a.weights, rows):
        with open(os.path.join(w, "per_image_{}.csv".format(name)), "w", newline="") as f:
            csv.writer(f).writerows([["line"] + METRICS + ["scale_ratio", "n_valid"]] + r)
        mean = np.array([x[1:8] for x in r], dtype=np.float64).mean(0)
        print(("{:>40} | " + "{:8.3f} | " * 7).format(w[-40:], *mean))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-gt")
    b.add_argument("--split_file", required=True)
    b.add_argument("--velo_root", required=True)
    b.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--split_file", required=True)
    e.add_argument("--gt", required=True)
    e.add_argument("--data_path", required=True)
    e.add_argument("--weights", nargs="+", required=True)
    e.add_argument("--model", default="lite-mono")
    e.add_argument("--min_depth", type=float, default=0.1)  # options.py defaults used by evaluate_depth.py
    e.add_argument("--max_depth", type=float, default=100.0)
    e.add_argument("--num_workers", type=int, default=os.cpu_count())
    a = p.parse_args()
    build_gt(a) if a.cmd == "build-gt" else evaluate(a)


if __name__ == "__main__":
    main()
