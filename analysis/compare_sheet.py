"""Side-by-side sheet: Lite-Mono vs a reference model on chosen Eigen test images.

Columns: photo | Lite-Mono depth | reference depth | Lite-Mono signed error | reference signed error
(signed error = log(pred/GT) at LiDAR points after per-image median scaling;
red = predicted too far, blue = predicted too close). Same colour scales for both models.

Usage: compare_sheet.py --ref_dir outputs/depth_pro_2/s/analysis --ref_name "Depth Pro 2-S" \
       --idx_file outputs/depth_pro_2/compare_s/only_lm.npy --n 8 --out sheet.png
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

LM = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")
DMIN, DMAX = 3.0, 80.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref_dir", required=True)
    p.add_argument("--ref_name", required=True)
    p.add_argument("--idx", type=int, nargs="*", default=[])
    p.add_argument("--idx_file", default=None)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--title", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--data_path", default=B.read_paths_env().get("KITTI_ROOT"))
    opt = p.parse_args()
    ids = list(opt.idx) + (np.load(opt.idx_file).tolist() if opt.idx_file else [])
    ids = ids[:opt.n]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cm, ce = plt.get_cmap("turbo"), plt.get_cmap("coolwarm")

    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    lines = [l.strip() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]
    dl = np.load(os.path.join(LM, "disps_eigen.npy"), mmap_mode="r")
    dr = np.load(os.path.join(opt.ref_dir, "disps_eigen.npy"), mmap_mode="r")

    def depth_rgb(dep):
        x = (np.log(np.clip(dep, DMIN, DMAX)) - np.log(DMIN)) / (np.log(DMAX) - np.log(DMIN))
        return cm(1 - x)[..., :3]

    def err_rgb(rgb, gt, m, pred):
        lr = np.zeros_like(gt); lr[m] = np.log(pred[m]) - np.log(gt[m])
        img = ce(np.clip(lr, -0.5, 0.5) + 0.5)[..., :3]
        big = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        fill = np.stack([cv2.dilate(np.where(m, img[..., c], 0).astype(np.float32), np.ones((3, 3), np.uint8)) for c in range(3)], -1)
        return np.where(big[..., None], fill, 0.25 + rgb * 0.3)

    n = len(ids)
    fig, axs = plt.subplots(n, 5, figsize=(16, 1.25 * n + 0.6))
    axs = np.atleast_2d(axs)
    for k, i in enumerate(ids):
        gt = gts[i]
        m = B.eval_mask(gt)
        pl, _ = B.scaled_pred(np.asarray(dl[i]), gt, m)
        pr, _ = B.scaled_pred(np.asarray(dr[i]), gt, m)
        rgb = B.load_rgb(opt.data_path, lines[i]).astype(np.float32) / 255
        al = np.mean(np.abs(gt[m] - pl[m]) / gt[m]); ar = np.mean(np.abs(gt[m] - pr[m]) / gt[m])
        H, W = gt.shape
        t = int(0.40810811 * H) - 40
        panels = [rgb, depth_rgb(pl), depth_rgb(pr), err_rgb(rgb, gt, m, pl), err_rgb(rgb, gt, m, pr)]
        for j, im in enumerate(panels):
            axs[k, j].imshow(im[max(t, 0):])
            axs[k, j].axis("off")
        drv = lines[i].split()[0].split("/")[1].replace("_sync", "").replace("2011_", "")
        axs[k, 0].set_title("idx {} {} f{}".format(i, drv, int(lines[i].split()[1])), fontsize=7, loc="left")
        axs[k, 3].set_title("AbsRel {:.3f}".format(al), fontsize=7, loc="left")
        axs[k, 4].set_title("AbsRel {:.3f}".format(ar), fontsize=7, loc="left")
    heads = ["photo", "Lite-Mono depth", "{} depth".format(opt.ref_name),
             "Lite-Mono error (red far / blue close)", "{} error".format(opt.ref_name)]
    for j, h in enumerate(heads):
        axs[0, j].set_title(h + "\n" + axs[0, j].get_title(), fontsize=8, loc="left")
    if opt.title:
        fig.suptitle(opt.title, fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(opt.out, dpi=110)
    print("wrote", opt.out)


if __name__ == "__main__":
    main()
