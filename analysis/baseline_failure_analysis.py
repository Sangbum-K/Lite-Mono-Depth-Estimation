"""Per-image / per-pixel failure analysis of a Lite-Mono checkpoint on the Eigen test split.

Mirrors vendor/Lite-Mono/evaluate_depth.py exactly (batch 16, no post-processing,
disp_to_depth(0.1, 100), resize disp to GT size, Garg crop, depth cap 1e-3..80 m,
per-image median scaling), then additionally saves:

  per_image.csv        7 metrics + scale ratio + GT statistics for all 697 images
  per_drive.csv        per-sequence mean metrics
  pixel_bins.csv       pooled per-pixel errors by GT distance bin and by region proxy
  summary.md           headline numbers, distribution, correlations, selected samples
  figs/{worst,best,random}/*.png   RGB | predicted disparity | per-LiDAR-point AbsRel
  figs/overview_*.png  stacked overview sheets, figs/hist_abs_rel.png, figs/bins.png
  disps_eigen.npy      raw network disparities (reused with --reuse_disps)

Sample selection is fixed by rule (worst/best by per-image AbsRel, random with a fixed
seed drawn before looking at errors), not chosen by eye.

Region proxies (KITTI GT is sparse LiDAR, there is no dense depth-boundary GT):
  - gt_disc : LiDAR point whose 9x9 neighbourhood contains other LiDAR points with a
              depth ratio > 1.25 (GT-based depth-discontinuity proxy; also catches
              LiDAR/camera parallax artifacts at occlusions)
  - rgb_edge: within 2 px of a Canny edge of the RGB image (texture edges included)
"""
import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LITEMONO = os.path.join(ROOT, "vendor", "Lite-Mono")
sys.path.insert(0, LITEMONO)

MIN_DEPTH, MAX_DEPTH = 1e-3, 80.0
DIST_BINS = [(0.0, 20.0), (20.0, 40.0), (40.0, 80.0)]
METRICS = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]


def read_paths_env():
    env = {}
    path = os.path.join(ROOT, "configs", "paths.env")
    if os.path.exists(path):
        for line in open(path):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                env[k] = v
    return env


def compute_errors(gt, pred):
    # identical to evaluate_depth.compute_errors
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt(((np.log(gt) - np.log(pred)) ** 2).mean())
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def predict_disps(weights, data_path, num_workers):
    import torch
    from torch.utils.data import DataLoader
    import datasets
    import networks
    from layers import disp_to_depth
    from utils import readlines

    filenames = readlines(os.path.join(LITEMONO, "splits", "eigen", "test_files.txt"))
    enc_dict = torch.load(os.path.join(weights, "encoder.pth"))
    dec_dict = torch.load(os.path.join(weights, "depth.pth"))
    dataset = datasets.KITTIRAWDataset(data_path, filenames, enc_dict["height"], enc_dict["width"],
                                       [0], 4, is_train=False, img_ext=".png")
    loader = DataLoader(dataset, 16, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last=False)
    encoder = networks.LiteMono(model="lite-mono", height=enc_dict["height"], width=enc_dict["width"])
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    encoder.load_state_dict({k: v for k, v in enc_dict.items() if k in encoder.state_dict()})
    decoder.load_state_dict({k: v for k, v in dec_dict.items() if k in decoder.state_dict()})
    encoder.cuda().eval()
    decoder.cuda().eval()
    out = []
    with torch.no_grad():
        for data in loader:
            disp = decoder(encoder(data[("color", 0, 0)].cuda()))[("disp", 0)]
            disp, _ = disp_to_depth(disp, 0.1, 100.0)  # options.py defaults min/max_depth
            out.append(disp.cpu()[:, 0].numpy())
    return np.concatenate(out)


def eval_mask(gt):
    h, w = gt.shape
    mask = np.logical_and(gt > MIN_DEPTH, gt < MAX_DEPTH)
    crop = np.array([0.40810811 * h, 0.99189189 * h, 0.03594771 * w, 0.96405229 * w]).astype(np.int32)
    crop_mask = np.zeros(mask.shape)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    return np.logical_and(mask, crop_mask)


def scaled_pred(pred_disp, gt, mask):
    """Full-resolution median-scaled predicted depth (same ops as evaluate_depth)."""
    h, w = gt.shape
    depth = 1.0 / cv2.resize(pred_disp, (w, h))
    ratio = np.median(gt[mask]) / np.median(depth[mask])
    depth = np.clip(depth * ratio, MIN_DEPTH, MAX_DEPTH)
    return depth, ratio


def gt_discontinuity(gt, win=9, thr=1.25, sides=False):
    """LiDAR points with a > thr depth jump in their win x win neighbourhood.
    sides=True returns (far_side, near_side): the point is the background (deeper than a
    neighbour by > thr) or the foreground (shallower than a neighbour by > thr)."""
    valid = gt > 0
    k = np.ones((win, win), np.uint8)
    dmax = cv2.dilate(np.where(valid, gt, 0).astype(np.float32), k)
    dmin = -cv2.dilate(np.where(valid, -gt, -1e6).astype(np.float32), k)
    if sides:
        far = valid & (gt > thr * dmin)
        near = valid & (gt * thr < dmax) & ~far
        return far, near
    return valid & ((dmax / np.maximum(dmin, 1e-3)) > thr)


def gt_enclosed(gt, far, reach=20, thr=1.25):
    """Background-side points with nearer LiDAR points within `reach` px on BOTH the left and
    the right of the same row: LiDAR seen through glass, fences or foliage (GT ambiguity),
    as opposed to the outer side of an object contour."""
    big = np.where(gt > 0, gt, 1e6).astype(np.float32)
    k = np.ones((1, reach + 1), np.uint8)
    lmin = cv2.erode(big, k, anchor=(reach, 0), borderType=cv2.BORDER_CONSTANT, borderValue=1e6)
    rmin = cv2.erode(big, k, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT, borderValue=1e6)
    return far & (lmin * thr < gt) & (rmin * thr < gt)


def rgb_edges(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 50, 150)
    return cv2.dilate(edges, np.ones((5, 5), np.uint8)) > 0


def load_rgb(data_path, line):
    folder, frame, side = line.split()
    cam = {"l": "image_02", "r": "image_03"}[side]
    path = os.path.join(data_path, folder, cam, "data", "{:010d}.png".format(int(frame)))
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


class PixelAcc:
    """Pools per-pixel errors (over all images) for one group of pixels."""

    def __init__(self):
        self.n = 0
        self.s = dict(abs_rel=0.0, sq_rel=0.0, sq=0.0, log_sq=0.0, a1=0.0, log_signed=0.0)

    def add(self, gt, pred):
        if gt.size == 0:
            return
        lr = np.log(pred) - np.log(gt)
        self.n += gt.size
        self.s["abs_rel"] += np.sum(np.abs(gt - pred) / gt)
        self.s["sq_rel"] += np.sum((gt - pred) ** 2 / gt)
        self.s["sq"] += np.sum((gt - pred) ** 2)
        self.s["log_sq"] += np.sum(lr ** 2)
        self.s["a1"] += np.sum(np.maximum(gt / pred, pred / gt) < 1.25)
        self.s["log_signed"] += np.sum(lr)

    def row(self):
        n = max(self.n, 1)
        return dict(n_pixels=self.n,
                    abs_rel=self.s["abs_rel"] / n, sq_rel=self.s["sq_rel"] / n,
                    rmse=np.sqrt(self.s["sq"] / n), rmse_log=np.sqrt(self.s["log_sq"] / n),
                    a1=self.s["a1"] / n,
                    mean_log_ratio_pred_over_gt=self.s["log_signed"] / n)


def panel(ax_row, rgb, disp_full, gt, pred, mask, title):
    import matplotlib.pyplot as plt  # noqa: F401
    ax0, ax1, ax2 = ax_row
    ax0.imshow(rgb)
    ax0.set_title(title, fontsize=8, loc="left")
    vmax = np.percentile(disp_full, 95)
    ax1.imshow(disp_full, cmap="magma", vmax=vmax)
    ys, xs = np.nonzero(mask)
    err = np.abs(gt[mask] - pred[mask]) / gt[mask]
    ax2.imshow((rgb * 0.35).astype(np.uint8))
    sc = ax2.scatter(xs, ys, c=err, s=1.2, cmap="jet", vmin=0, vmax=0.5, linewidths=0)
    for a in ax_row:
        a.axis("off")
    return sc


def main():
    env = read_paths_env()
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(ROOT, "outputs/20261109_baseline_unseeded/models/weights_29"))
    p.add_argument("--out", default=os.path.join(ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29"))
    p.add_argument("--data_path", default=env.get("KITTI_ROOT", os.path.join(ROOT, "data/raw/kitti")))
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--n_worst", type=int, default=20)
    p.add_argument("--n_best", type=int, default=5)
    p.add_argument("--n_random", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reuse_disps", action="store_true")
    p.add_argument("--no_figs", action="store_true")
    opt = p.parse_args()
    os.makedirs(opt.out, exist_ok=True)

    lines = [l.strip() for l in open(os.path.join(LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]
    disp_path = os.path.join(opt.out, "disps_eigen.npy")
    if opt.reuse_disps and os.path.exists(disp_path):
        disps = np.load(disp_path)
    else:
        disps = predict_disps(opt.weights, opt.data_path, opt.num_workers)
        np.save(disp_path, disps)
    gts = np.load(os.path.join(LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    assert len(disps) == len(gts) == len(lines)

    # random sample fixed before any error is computed
    rand_idx = sorted(np.random.default_rng(opt.seed).choice(len(lines), opt.n_random, replace=False).tolist())

    bin_acc = {b: PixelAcc() for b in DIST_BINS}
    region_acc = {k: PixelAcc() for k in ["all", "gt_disc", "gt_smooth", "disc_far_side", "disc_far_edge",
                                          "disc_far_enclosed", "disc_near_side", "rgb_edge", "rgb_flat"]}
    rows = []
    for i, line in enumerate(lines):
        gt = gts[i]
        mask = eval_mask(gt)
        pred, ratio = scaled_pred(disps[i], gt, mask)
        g, pr = gt[mask], pred[mask]
        m = compute_errors(g, pr)
        disc = gt_discontinuity(gt)[mask]
        edge = rgb_edges(load_rgb(opt.data_path, line))[mask]
        region_acc["all"].add(g, pr)
        region_acc["gt_disc"].add(g[disc], pr[disc])
        region_acc["gt_smooth"].add(g[~disc], pr[~disc])
        far_s, near_s = gt_discontinuity(gt, sides=True)
        enc = gt_enclosed(gt, far_s)[mask]
        far_s, near_s = far_s[mask], near_s[mask]
        region_acc["disc_far_side"].add(g[far_s], pr[far_s])
        region_acc["disc_far_edge"].add(g[far_s & ~enc], pr[far_s & ~enc])
        region_acc["disc_far_enclosed"].add(g[enc], pr[enc])
        region_acc["disc_near_side"].add(g[near_s], pr[near_s])
        region_acc["rgb_edge"].add(g[edge], pr[edge])
        region_acc["rgb_flat"].add(g[~edge], pr[~edge])
        row = dict(idx=i, folder=line.split()[0], frame=int(line.split()[1]), side=line.split()[2])
        row.update({k: float(v) for k, v in zip(METRICS, m)})
        row.update(ratio=float(ratio), n_valid=int(mask.sum()), gt_median=float(np.median(g)),
                   frac_far40=float(np.mean(g >= 40)), frac_gt_disc=float(disc.mean()),
                   frac_rgb_edge=float(edge.mean()),
                   mean_log_ratio=float(np.mean(np.log(pr) - np.log(g))))
        for lo, hi in DIST_BINS:
            sel = (g >= lo) & (g < hi) if hi < MAX_DEPTH else (g >= lo)
            bin_acc[(lo, hi)].add(g[sel], pr[sel])
            row["abs_rel_{:g}_{:g}".format(lo, hi)] = float(np.mean(np.abs(g[sel] - pr[sel]) / g[sel])) if sel.any() else ""
        rows.append(row)

    # ---- per-image csv + sanity check against evaluate_depth ----
    with open(os.path.join(opt.out, "per_image.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    M = np.array([[r[k] for k in METRICS] for r in rows])
    mean = M.mean(0)
    ratios = np.array([r["ratio"] for r in rows])

    # ---- per-drive ----
    drives = {}
    for r in rows:
        drives.setdefault(r["folder"].split("/")[1], []).append(r)
    drive_rows = []
    for d, rs in drives.items():
        dr = dict(drive=d, n_images=len(rs))
        dr.update({k: float(np.mean([r[k] for r in rs])) for k in METRICS})
        drive_rows.append(dr)
    drive_rows.sort(key=lambda r: -r["abs_rel"])
    with open(os.path.join(opt.out, "per_drive.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(drive_rows[0].keys()))
        w.writeheader()
        w.writerows(drive_rows)

    # ---- pooled pixel stats ----
    pix_rows = []
    for (lo, hi), acc in bin_acc.items():
        pix_rows.append(dict(group="dist_{:g}-{:g}m".format(lo, hi), **acc.row()))
    for k, acc in region_acc.items():
        pix_rows.append(dict(group="region_" + k, **acc.row()))
    total_abs = region_acc["all"].s["abs_rel"]
    accs = list(bin_acc.values()) + list(region_acc.values())
    for r, acc in zip(pix_rows, accs):
        r["frac_pixels"] = acc.n / region_acc["all"].n
        r["abs_rel_share"] = acc.s["abs_rel"] / total_abs
    with open(os.path.join(opt.out, "pixel_bins.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pix_rows[0].keys()))
        w.writeheader()
        w.writerows(pix_rows)

    # ---- selection ----
    order = np.argsort(M[:, 0])
    worst_idx = order[::-1][:opt.n_worst].tolist()
    best_idx = order[:opt.n_best].tolist()

    # ---- correlations (Spearman via ranks) ----
    absrel = M[:, 0]

    def spearman(x):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(absrel))
        return float(np.corrcoef(rx, ry)[0, 1])

    corr = {
        "|log(ratio / median ratio)|": spearman(np.abs(np.log(ratios / np.median(ratios)))),
        "frac GT >= 40 m": spearman(np.array([r["frac_far40"] for r in rows])),
        "GT median depth": spearman(np.array([r["gt_median"] for r in rows])),
        "frac GT-discontinuity points": spearman(np.array([r["frac_gt_disc"] for r in rows])),
        "frac RGB-edge points": spearman(np.array([r["frac_rgb_edge"] for r in rows])),
        "n valid LiDAR points": spearman(np.array([r["n_valid"] for r in rows])),
    }

    # ---- summary ----
    q = np.percentile(absrel, [50, 75, 90, 95, 99])
    top10 = absrel[order[::-1][:int(round(0.1 * len(absrel)))]]
    md = ["# Failure analysis: {}".format(os.path.relpath(opt.weights, ROOT)), "",
          "Protocol: Eigen test 697, median scaling per image, cap 80 m, Garg crop, no post-processing.", "",
          "## Mean over images (must match evaluate_depth.py)", "",
          "| " + " | ".join(METRICS) + " |", "|" + "---:|" * len(METRICS),
          "| " + " | ".join("{:.3f}".format(v) for v in mean) + " |", "",
          "Scale ratio: median {:.3f}, std(ratio/median) {:.3f}".format(np.median(ratios), np.std(ratios / np.median(ratios))), "",
          "## Per-image AbsRel distribution", "",
          "mean {:.3f}, median {:.3f}, p75 {:.3f}, p90 {:.3f}, p95 {:.3f}, p99 {:.3f}, max {:.3f}".format(
              absrel.mean(), q[0], q[1], q[2], q[3], q[4], absrel.max()),
          "Worst 10% of images ({}) hold {:.1%} of the summed AbsRel.".format(len(top10), top10.sum() / absrel.sum()),
          "Mean AbsRel without worst 10%: {:.3f}".format(absrel[order[:len(absrel) - len(top10)]].mean()), "",
          "## Pooled per-pixel errors", "",
          "| group | n_pixels | % pixels | % of AbsRel sum | abs_rel | sq_rel | rmse | rmse_log | a1 | mean log(pred/gt) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in pix_rows:
        md.append("| {group} | {n_pixels} | {fp:.1%} | {fs:.1%} | {abs_rel:.3f} | {sq_rel:.3f} | {rmse:.3f} | {rmse_log:.3f} | {a1:.3f} | {mean_log_ratio_pred_over_gt:+.3f} |".format(
            fp=r["frac_pixels"], fs=r["abs_rel_share"], **r))
    md += ["", "mean log(pred/gt) < 0: underestimates depth (predicts too close).",
           "gt_disc = LiDAR point with a >1.25x depth jump inside its 9x9 neighbourhood (includes LiDAR/camera parallax artifacts).",
           "disc_far_side / disc_near_side = background / foreground side of such a jump.",
           "disc_far_enclosed = background-side point with nearer points within 20 px on both sides of its row "
           "(LiDAR through glass/fence/foliage, GT ambiguity); disc_far_edge = the rest (outer side of a contour).",
           "rgb_edge = within 2 px of a Canny edge (texture edges included). Neither is a true depth-boundary GT.", "",
           "## Spearman correlation with per-image AbsRel", ""]
    md += ["- {}: {:+.3f}".format(k, v) for k, v in corr.items()]
    md += ["", "## Per-drive (worst 5 by mean AbsRel)", "", "| drive | n | abs_rel | sq_rel | a1 |", "|---|---:|---:|---:|---:|"]
    md += ["| {drive} | {n_images} | {abs_rel:.3f} | {sq_rel:.3f} | {a1:.3f} |".format(**r) for r in drive_rows[:5]]
    md += ["", "## Selected samples", "",
           "random (seed {}): {}".format(opt.seed, rand_idx), "", "| set | rank | idx | file | abs_rel | a1 | ratio |", "|---|---:|---:|---|---:|---:|---:|"]
    for name, ids in [("worst", worst_idx), ("best", best_idx), ("random", rand_idx)]:
        for k, i in enumerate(ids):
            r = rows[i]
            md.append("| {} | {} | {} | {} {} {} | {:.3f} | {:.3f} | {:.2f} |".format(
                name, k + 1, i, r["folder"], r["frame"], r["side"], r["abs_rel"], r["a1"], r["ratio"]))
    open(os.path.join(opt.out, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(dict(mean=dict(zip(METRICS, mean.tolist())), worst=worst_idx, best=best_idx, random=rand_idx,
                   spearman=corr), open(os.path.join(opt.out, "summary.json"), "w"), indent=1)
    print("mean metrics:", " ".join("{}={:.3f}".format(k, v) for k, v in zip(METRICS, mean)))

    if opt.no_figs:
        return

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = os.path.join(opt.out, "figs")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(absrel, bins=60, color="#4a6fa5")
    ax.axvline(absrel.mean(), color="k", ls="--", lw=1, label="mean {:.3f}".format(absrel.mean()))
    ax.axvline(np.median(absrel), color="r", ls=":", lw=1, label="median {:.3f}".format(np.median(absrel)))
    ax.set_xlabel("per-image AbsRel")
    ax.set_ylabel("# images")
    ax.legend()
    fig.tight_layout()
    os.makedirs(figdir, exist_ok=True)
    fig.savefig(os.path.join(figdir, "hist_abs_rel.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    labels = [r["group"].replace("dist_", "") for r in pix_rows[:len(DIST_BINS)]]
    ax[0].bar(labels, [r["abs_rel"] for r in pix_rows[:len(DIST_BINS)]], color="#4a6fa5")
    ax[0].set_title("AbsRel by GT distance (pooled pixels)", fontsize=9)
    ax[1].bar(labels, [r["n_pixels"] for r in pix_rows[:len(DIST_BINS)]], color="#999")
    ax[1].set_title("# LiDAR points", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "bins.png"), dpi=150)
    plt.close(fig)

    def render(i):
        gt = gts[i]
        mask = eval_mask(gt)
        pred, _ = scaled_pred(disps[i], gt, mask)
        rgb = load_rgb(opt.data_path, lines[i])
        disp_full = cv2.resize(disps[i], (gt.shape[1], gt.shape[0]))
        return rgb, disp_full, gt, pred, mask

    def title(i, tag):
        r = rows[i]
        return "{} | idx {} | {} {} | AbsRel {:.3f}  d1 {:.3f}".format(
            tag, i, r["folder"].split("/")[1].replace("_sync", ""), r["frame"], r["abs_rel"], r["a1"])

    for name, ids in [("worst", worst_idx), ("best", best_idx), ("random", rand_idx)]:
        d = os.path.join(figdir, name)
        os.makedirs(d, exist_ok=True)
        for k, i in enumerate(ids):
            fig, axs = plt.subplots(3, 1, figsize=(9, 8.4))
            sc = panel(axs, *render(i), title=title(i, "{} #{}".format(name, k + 1)))
            fig.colorbar(sc, ax=axs[2], fraction=0.02, pad=0.01, label="AbsRel")
            fig.tight_layout()
            fig.savefig(os.path.join(d, "{:02d}_idx{:03d}.png".format(k + 1, i)), dpi=110)
            plt.close(fig)
        # overview sheet: one row per image, columns RGB | disparity | error
        n = min(len(ids), 8)
        fig, axs = plt.subplots(n, 3, figsize=(15, 1.75 * n))
        axs = np.atleast_2d(axs)
        for k, i in enumerate(ids[:n]):
            panel(axs[k], *render(i), title=title(i, "{} #{}".format(name, k + 1)))
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "overview_{}.png".format(name)), dpi=110)
        plt.close(fig)
        # signed error sheet: red = predicted too far, blue = predicted too close
        fig, axs = plt.subplots(n, 1, figsize=(8, 1.9 * n))
        axs = np.atleast_1d(axs)
        for k, i in enumerate(ids[:n]):
            rgb, _, gt, pred, mask = render(i)
            lr = np.log(pred[mask]) - np.log(gt[mask])
            ys, xs = np.nonzero(mask)
            axs[k].imshow((rgb * 0.35).astype(np.uint8))
            sc = axs[k].scatter(xs, ys, c=lr, s=1.0, cmap="coolwarm", vmin=-0.5, vmax=0.5, linewidths=0)
            axs[k].axis("off")
            far = gt[mask] >= 40
            axs[k].set_title("{} | mean log(pred/gt) {:+.3f}, >=40m {}".format(
                title(i, "{} #{}".format(name, k + 1)), lr.mean(),
                "{:+.3f}".format(lr[far].mean()) if far.any() else "n/a"), fontsize=7, loc="left")
        fig.colorbar(sc, ax=list(axs), fraction=0.02, pad=0.01, label="log(pred/gt)  red=too far, blue=too close")
        fig.savefig(os.path.join(figdir, "signed_{}.png".format(name)), dpi=110, bbox_inches="tight")
        plt.close(fig)
    print("wrote", opt.out)


if __name__ == "__main__":
    main()
