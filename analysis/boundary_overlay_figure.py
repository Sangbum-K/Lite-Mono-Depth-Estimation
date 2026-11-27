"""Report figure: true (LiDAR) depth edges vs predicted (Lite-Mono) depth edges drawn on images.

For each case, three zoomed panels side by side:
  1. LiDAR ground-truth depth
  2. Lite-Mono predicted depth (median-scaled), with its depth edges marked (red)
  3. the photo with the predicted edges
(LiDAR-derived "true edge" markers were removed on purpose: LiDAR jumps also come from
glass, fences and sensor noise, so they are not a reliable boundary ground truth.)

Predicted edges: Canny on the log of the predicted depth (fixed 3-80 m range), i.e. the line
where the predicted depth changes fastest.

Cases are the same rule-based selection as boundary_zoom_figures.py --select edge
(edges measured by edge_offset.py: located, offset 8-24 px, depth ratio >= 2, one per drive).
Output: <analysis_dir>/boundary_overlay/overlay_a4.{png,pdf}
"""
import argparse
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

DEFAULT_DIR = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")
DMIN, DMAX = 3.0, 80.0
CASE_NAMES = {213: "car roof vs. building", 364: "car roof vs. far street", 68: "tree trunk vs. far field",
              480: "parked vehicle edge", 529: "pedestrian vs. street"}  # written after looking at the cases


def select_cases(analysis_dir, lines, n, min_off=8.0, max_off=24.0, min_ratio=2.0):
    er = [r for r in csv.DictReader(open(os.path.join(analysis_dir, "edges", "edges.csv")))
          if r["status"] == "located" and min_off <= float(r["offset_px"]) <= max_off
          and float(r["d_bg"]) / float(r["d_fg"]) >= min_ratio]
    er.sort(key=lambda r: -float(r["d_bg"]) / float(r["d_fg"]))
    chosen, drives = [], set()
    for r in er:
        i = int(r["idx"])
        drv = lines[i].split()[0]
        if drv in drives:
            continue
        drives.add(drv)
        chosen.append(dict(i=i, row=int(r["row"]), xj=float(r["x_jump"]), off=float(r["offset_px"]),
                           dfg=float(r["d_fg"]), dbg=float(r["d_bg"])))
        if len(chosen) == n:
            break
    return chosen


def lidar_edges(gt, mask, jump=1.25, max_gap=3):
    """(ys, xs) of the midpoints of all LiDAR depth jumps."""
    ys, xs = [], []
    for y in np.nonzero(mask.any(1))[0]:
        x = np.nonzero(mask[y])[0]
        if len(x) < 2:
            continue
        dd = gt[y, x]
        r = np.maximum(dd[1:], dd[:-1]) / np.minimum(dd[1:], dd[:-1])
        j = np.nonzero((np.diff(x) <= max_gap) & (r > jump))[0]
        ys += [y] * len(j)
        xs += list(0.5 * (x[j] + x[j + 1]))
    return np.array(ys), np.array(xs)


def pred_edges(pred):
    lg = (np.log(np.clip(pred, DMIN, DMAX)) - np.log(DMIN)) / (np.log(DMAX) - np.log(DMIN))
    img = cv2.GaussianBlur((lg * 255).astype(np.uint8), (3, 3), 0)
    return cv2.Canny(img, 25, 50) > 0


def depth_rgb(depth, valid, cmap):
    x = (np.log(np.clip(depth, DMIN, DMAX)) - np.log(DMIN)) / (np.log(DMAX) - np.log(DMIN))
    rgb = cmap(1 - x)[..., :3]
    rgb[~valid] = 1.0
    return rgb


def single_case(c, gts, disps, lines, opt, out, cm):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import NullLocator

    G, R = "#00c853", "#ff1744"
    i, cw, ch = c["i"], opt.crop_w, opt.crop_h
    gt = gts[i]
    mask = B.eval_mask(gt)
    pred, _ = B.scaled_pred(disps[i], gt, mask)
    rgb = B.load_rgb(opt.data_path, lines[i]).astype(np.float32) / 255
    H, W = gt.shape
    t, b, l, r = int(0.40810811 * H), int(0.99189189 * H), int(0.03594771 * W), int(0.96405229 * W)
    y0 = int(np.clip(c["row"] - ch // 2, t, b - ch))
    x0 = int(np.clip(round(c["xj"]) - cw // 2, l, r - cw))
    sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    ey, ex = lidar_edges(gt, mask)
    inside = (ey >= y0) & (ey < y0 + ch) & (ex >= x0) & (ex < x0 + cw)
    ey, ex = ey[inside] - y0, ex[inside] - x0
    pe = cv2.dilate(pred_edges(pred)[sl].astype(np.uint8), np.ones((2, 2), np.uint8)) > 0
    m = mask[sl]
    py, px = np.nonzero(m)

    # LiDAR panel with points enlarged 3x3 for visibility
    gvis = depth_rgb(gt[sl], m, cm)
    grow = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gfill = cv2.dilate(np.where(m, gt[sl], 0).astype(np.float32), np.ones((3, 3), np.uint8))
    gvis = np.where((grow & ~m)[..., None], depth_rgb(gfill, grow, cm), gvis)
    pvis = depth_rgb(pred[sl], np.ones_like(m), cm)

    def edges(ax, red=True, green=False):
        if red:
            ov = np.zeros(pe.shape + (4,))
            ov[pe] = (1.0, 0.09, 0.27, 1.0)
            ax.imshow(ov, interpolation="nearest", zorder=3)
        if green:
            ax.scatter(ex, ey, s=11, c=G, edgecolors="k", linewidths=0.35, zorder=4)

    fig, axs = plt.subplots(2, 2, figsize=(6.7, 3.45))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.93, bottom=0.17, wspace=0.025, hspace=0.2)
    axs[0, 0].imshow(rgb[sl], interpolation="nearest"); edges(axs[0, 0])
    axs[0, 1].imshow(gvis, interpolation="nearest"); edges(axs[0, 1], red=False)
    axs[1, 0].imshow(pvis, interpolation="nearest"); edges(axs[1, 0], green=False)
    axs[1, 1].imshow(pvis, interpolation="nearest")
    axs[1, 1].scatter(px, py, s=3.5, marker="s", c=depth_rgb(gt[sl][m], np.ones(m.sum(), bool), cm),
                      linewidths=0, zorder=2)
    edges(axs[1, 1], green=False)
    titles = ["(a) photo + predicted edge", "(b) true depth (LiDAR)",
              "(c) predicted depth (Lite-Mono) + predicted edges",
              "(d) LiDAR points on the prediction (stand out where wrong)"]
    for a, ttl in zip(axs.ravel(), titles):
        a.set_title(ttl, fontsize=7.5, loc="left", pad=2)
        a.set_xticks([]); a.set_yticks([])
        for sp in a.spines.values():
            sp.set_linewidth(0.5)
    drv = lines[i].split()[0].split("/")[1].replace("_sync", "")
    fig.suptitle("car roof vs. far street: object {:.0f} m, background {:.0f} m   (idx {}, {} frame {})".format(
        c["dfg"], c["dbg"], i, drv, int(lines[i].split()[1])), fontsize=8, x=0.005, ha="left", y=0.995)
    fig.legend(handles=[Line2D([], [], color=R, lw=1.6, label="predicted edge (Lite-Mono)")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=7, frameon=False)
    cax = fig.add_axes([0.62, 0.075, 0.36, 0.02])
    cb = fig.colorbar(ScalarMappable(norm=LogNorm(DMIN, DMAX), cmap=cm.reversed()), cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticks([3, 5, 10, 20, 40, 80]); cb.set_ticklabels(["3", "5", "10", "20", "40", "80"])
    cb.ax.tick_params(labelsize=6.5); cb.set_label("depth (m)", fontsize=7, labelpad=1)
    name = "single_idx{:03d}".format(i)
    fig.savefig(os.path.join(out, name + ".png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(out, name + ".pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".png"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", default=DEFAULT_DIR)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--crop_w", type=int, default=260)
    p.add_argument("--crop_h", type=int, default=110)
    p.add_argument("--data_path", default=B.read_paths_env().get("KITTI_ROOT"))
    p.add_argument("--single", type=int, default=-1, help="idx of one case: 2x2 figure incl. GT-on-prediction overlay")
    opt = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import NullLocator

    cm = plt.get_cmap("turbo")
    out = os.path.join(opt.analysis_dir, "boundary_overlay")
    os.makedirs(out, exist_ok=True)
    disps = np.load(os.path.join(opt.analysis_dir, "disps_eigen.npy"))
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    lines = [l.strip() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]
    cases = select_cases(opt.analysis_dir, lines, opt.n)
    print("cases:", [c["i"] for c in cases])
    if opt.single >= 0:
        c = [c for c in cases if c["i"] == opt.single]
        assert c, "idx {} is not among the selected cases".format(opt.single)
        single_case(c[0], gts, disps, lines, opt, out, cm)
        return

    cw, ch = opt.crop_w, opt.crop_h
    n = len(cases)
    fig, axs = plt.subplots(n, 3, figsize=(6.7, 1.3 * n + 0.7))
    fig.subplots_adjust(left=0.005, right=0.995, top=0.965, bottom=0.085, wspace=0.025, hspace=0.24)
    axs = np.atleast_2d(axs)
    G, R = "#00c853", "#ff1744"
    for k, c in enumerate(cases):
        i = c["i"]
        gt = gts[i]
        mask = B.eval_mask(gt)
        pred, _ = B.scaled_pred(disps[i], gt, mask)
        rgb = B.load_rgb(opt.data_path, lines[i]).astype(np.float32) / 255
        H, W = gt.shape
        t, b, l, r = int(0.40810811 * H), int(0.99189189 * H), int(0.03594771 * W), int(0.96405229 * W)
        y0 = int(np.clip(c["row"] - ch // 2, t, b - ch))
        x0 = int(np.clip(round(c["xj"]) - cw // 2, l, r - cw))
        sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))

        ey, ex = lidar_edges(gt, mask)
        inside = (ey >= y0) & (ey < y0 + ch) & (ex >= x0) & (ex < x0 + cw)
        ey, ex = ey[inside] - y0, ex[inside] - x0
        pe = pred_edges(pred)[sl]
        pe_thick = cv2.dilate(pe.astype(np.uint8), np.ones((2, 2), np.uint8)) > 0

        # 1. LiDAR depth (points enlarged 3x3 for visibility) + true edges
        m = mask[sl]
        gvis = depth_rgb(gt[sl], m, cm)
        grow = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        gfill = cv2.dilate(np.where(m, gt[sl], 0).astype(np.float32), np.ones((3, 3), np.uint8))
        gvis = np.where((grow & ~m)[..., None], depth_rgb(gfill, grow, cm), gvis)
        axs[k, 0].imshow(gvis, interpolation="nearest")
        # 2. prediction + predicted edges
        pvis = depth_rgb(pred[sl], np.ones_like(m), cm)
        pvis[pe_thick] = (1.0, 0.09, 0.27)
        axs[k, 1].imshow(pvis, interpolation="nearest")
        # 3. photo with both
        photo = rgb[sl].copy()
        photo[pe_thick] = (1.0, 0.09, 0.27)
        axs[k, 2].imshow(photo, interpolation="nearest")
        for a in axs[k]:
            a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values():
                sp.set_linewidth(0.5)
        drv = lines[i].split()[0].split("/")[1].replace("_sync", "")
        axs[k, 0].text(0, 1.03, "({}) {}: object {:.0f} m, background {:.0f} m".format(
            k + 1, CASE_NAMES.get(i, "case"), c["dfg"], c["dbg"]),
            transform=axs[k, 0].transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        axs[k, 2].text(1, 1.03, "idx {} | {} frame {}".format(i, drv, int(lines[i].split()[1])),
                       transform=axs[k, 2].transAxes, fontsize=6, color="0.35", ha="right", va="bottom")
    for j, ttl in enumerate(["true depth (LiDAR)", "predicted depth (Lite-Mono)", "photo + predicted edge"]):
        axs[0, j].set_title(ttl, fontsize=8, pad=13)

    fig.legend(handles=[Line2D([], [], color=R, lw=2, label="predicted edge (Lite-Mono)")],
               loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=7, frameon=False)
    cax = fig.add_axes([0.62, 0.035, 0.36, 0.012])
    cb = fig.colorbar(ScalarMappable(norm=LogNorm(DMIN, DMAX), cmap=cm.reversed()), cax=cax, orientation="horizontal")
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticks([3, 5, 10, 20, 40, 80]); cb.set_ticklabels(["3", "5", "10", "20", "40", "80"])
    cb.ax.tick_params(labelsize=6.5); cb.set_label("depth (m)", fontsize=7, labelpad=1)
    fig.savefig(os.path.join(out, "overlay_a4.png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(out, "overlay_a4.pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
