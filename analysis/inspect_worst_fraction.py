"""Inspect the worst X% of Eigen test images (by per-image AbsRel) from a finished
baseline_failure_analysis.py run.

Outputs (in <analysis_dir>/worst<pct>/):
  group_stats.md   worst X% vs the rest: image-level factors, pooled pixel errors by
                   distance bin / region, drive distribution
  sheets/sheet_XX.png   10 images per sheet in rank order: RGB | disparity | signed error
                   (log(pred/gt), red = predicted too far, blue = too close)
  labels_template.csv   one row per image for manual failure-type labelling
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

DEFAULT_DIR = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", default=DEFAULT_DIR)
    p.add_argument("--pct", type=float, default=30.0)
    p.add_argument("--per_sheet", type=int, default=10)
    p.add_argument("--data_path", default=B.read_paths_env().get("KITTI_ROOT"))
    opt = p.parse_args()
    out = os.path.join(opt.analysis_dir, "worst{:g}".format(opt.pct))
    os.makedirs(os.path.join(out, "sheets"), exist_ok=True)

    rows = list(csv.DictReader(open(os.path.join(opt.analysis_dir, "per_image.csv"))))
    absrel = np.array([float(r["abs_rel"]) for r in rows])
    order = np.argsort(-absrel)
    n_top = int(round(len(rows) * opt.pct / 100))
    top = order[:n_top]
    rest = order[n_top:]
    is_top = np.zeros(len(rows), bool)
    is_top[top] = True

    disps = np.load(os.path.join(opt.analysis_dir, "disps_eigen.npy"))
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    lines = [l.strip() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]

    # ---- pooled pixel errors per group ----
    groups = {"worst": is_top, "rest": ~is_top}
    acc = {g: {k: B.PixelAcc() for k in ["all", "0-20", "20-40", "40-80", "gt_disc", "gt_smooth"]} for g in groups}
    for i in range(len(rows)):
        g = "worst" if is_top[i] else "rest"
        gt = gts[i]
        mask = B.eval_mask(gt)
        pred, _ = B.scaled_pred(disps[i], gt, mask)
        gv, pv = gt[mask], pred[mask]
        disc = B.gt_discontinuity(gt)[mask]
        a = acc[g]
        a["all"].add(gv, pv)
        a["gt_disc"].add(gv[disc], pv[disc])
        a["gt_smooth"].add(gv[~disc], pv[~disc])
        for (lo, hi), k in zip(B.DIST_BINS, ["0-20", "20-40", "40-80"]):
            sel = (gv >= lo) & ((gv < hi) if hi < B.MAX_DEPTH else True)
            a[k].add(gv[sel], pv[sel])

    def f(r, k):
        return np.array([float(x[k]) for x in r])

    sel_top = [rows[i] for i in top]
    sel_rest = [rows[i] for i in rest]
    md = ["# Worst {:g}% ({} images, AbsRel >= {:.3f}) vs rest ({} images)".format(
        opt.pct, n_top, absrel[top].min(), len(rest)), "",
        "## Image-level factors (mean)", "", "| factor | worst | rest |", "|---|---:|---:|"]
    for k in ["abs_rel", "a1", "ratio", "n_valid", "gt_median", "frac_far40", "frac_gt_disc", "frac_rgb_edge", "mean_log_ratio"]:
        md.append("| {} | {:.3f} | {:.3f} |".format(k, f(sel_top, k).mean(), f(sel_rest, k).mean()))
    md.append("| abs(log(ratio/median)) | {:.3f} | {:.3f} |".format(
        np.abs(np.log(f(sel_top, "ratio") / np.median(f(rows, "ratio")))).mean(),
        np.abs(np.log(f(sel_rest, "ratio") / np.median(f(rows, "ratio")))).mean()))
    md += ["", "## Pooled pixel errors", "",
           "| group | region | n_pixels | % of group pixels | % of group AbsRel sum | abs_rel | a1 | mean log(pred/gt) |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        tot_n, tot_s = acc[g]["all"].n, acc[g]["all"].s["abs_rel"]
        for k, a in acc[g].items():
            r = a.row()
            md.append("| {} | {} | {} | {:.1%} | {:.1%} | {:.3f} | {:.3f} | {:+.3f} |".format(
                g, k, a.n, a.n / tot_n, a.s["abs_rel"] / tot_s, r["abs_rel"], r["a1"], r["mean_log_ratio_pred_over_gt"]))
    drive_top = Counter(r["folder"].split("/")[1] for r in sel_top)
    drive_all = Counter(r["folder"].split("/")[1] for r in rows)
    md += ["", "## Drives most over-represented in the worst group", "",
           "| drive | in worst | total | share in worst |", "|---|---:|---:|---:|"]
    for d, c in sorted(drive_top.items(), key=lambda x: (-x[1] / drive_all[x[0]], -x[1]))[:15]:
        md.append("| {} | {} | {} | {:.0%} |".format(d, c, drive_all[d], c / drive_all[d]))
    open(os.path.join(out, "group_stats.md"), "w").write("\n".join(md) + "\n")

    with open(os.path.join(out, "labels_template.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "idx", "drive", "frame", "abs_rel", "a1", "mean_log_ratio", "labels", "note"])
        for k, i in enumerate(top):
            r = rows[i]
            w.writerow([k + 1, i, r["folder"].split("/")[1].replace("_sync", ""), r["frame"],
                        "{:.3f}".format(float(r["abs_rel"])), "{:.3f}".format(float(r["a1"])),
                        "{:+.3f}".format(float(r["mean_log_ratio"])), "", ""])

    # ---- sheets ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2

    for s in range(0, n_top, opt.per_sheet):
        ids = top[s:s + opt.per_sheet]
        fig, axs = plt.subplots(len(ids), 3, figsize=(15, 1.6 * len(ids)))
        axs = np.atleast_2d(axs)
        for k, i in enumerate(ids):
            gt = gts[i]
            mask = B.eval_mask(gt)
            pred, _ = B.scaled_pred(disps[i], gt, mask)
            rgb = B.load_rgb(opt.data_path, lines[i])
            disp = cv2.resize(disps[i], (gt.shape[1], gt.shape[0]))
            lr = np.log(pred[mask]) - np.log(gt[mask])
            ys, xs = np.nonzero(mask)
            r = rows[i]
            axs[k, 0].imshow(rgb)
            axs[k, 0].set_title("#{} idx {} {} {} AbsRel {:.3f} d1 {:.3f}".format(
                s + k + 1, i, r["folder"].split("/")[1].replace("_sync", "").replace("2011_", ""),
                r["frame"], float(r["abs_rel"]), float(r["a1"])), fontsize=8, loc="left")
            axs[k, 1].imshow(disp, cmap="magma", vmax=np.percentile(disp, 95))
            axs[k, 2].imshow((rgb * 0.35).astype(np.uint8))
            axs[k, 2].scatter(xs, ys, c=lr, s=0.8, cmap="coolwarm", vmin=-0.5, vmax=0.5, linewidths=0)
            axs[k, 2].set_title("log(pred/gt) mean {:+.3f}  red=too far blue=too close".format(lr.mean()), fontsize=7, loc="left")
            for a in axs[k]:
                a.axis("off")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "sheets", "sheet_{:02d}.png".format(s // opt.per_sheet + 1)), dpi=75)
        plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
