"""Report figures: full-image GT vs prediction and pixel-level zooms at depth boundaries.

Uses the predictions saved by baseline_failure_analysis.py (disps_eigen.npy) and the same
evaluation protocol (Garg crop, 80 m cap, per-image median scaling). Everything is shown at
KITTI GT resolution (about 1242x375); the network output (640x192) is resized as in evaluation.

Selection (fixed rules, no hand-picking):
  --select abs_rel   : the N images with the highest per-image AbsRel
  --select boundary  : the N images with the largest summed AbsRel at GT depth-discontinuity points
Zoom window: the crop_w x crop_h window (inside the evaluated area) with the largest summed
AbsRel at discontinuity points. Profile: the LiDAR row inside that window with the largest
summed boundary AbsRel.

Outputs (in <analysis_dir>/boundary_zoom/<select>/):
  idxNNN.png   one detailed figure per image
  summary.png  compact sheet of all N images (zoom panels + profile)
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

DEFAULT_DIR = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")
DMIN, DMAX = 3.0, 80.0  # depth colour range (log scale)
# Short descriptions of the auto-selected edge cases, written after looking at them (figure captions only).
CASE_NAMES = {213: "car roof vs. building", 364: "car roof vs. far street", 68: "tree trunk vs. far field",
              480: "parked vehicle edge", 529: "pedestrian vs. street"}
CLS_COLORS = {"smooth": (0.62, 0.62, 0.62), "far": (0.95, 0.55, 0.0), "near": (0.0, 0.62, 0.9), "enc": (0.6, 0.2, 0.75)}


def eval_box(h, w):
    return (int(0.40810811 * h), int(0.99189189 * h), int(0.03594771 * w), int(0.96405229 * w))


def prepare(i, disps, gts):
    gt = gts[i]
    mask = B.eval_mask(gt)
    pred, ratio = B.scaled_pred(disps[i], gt, mask)
    far, near = B.gt_discontinuity(gt, sides=True)
    enc = B.gt_enclosed(gt, far) & mask
    far, near = far & mask & ~enc, near & mask
    err = np.zeros_like(gt, dtype=np.float32)
    err[mask] = np.abs(gt[mask] - pred[mask]) / gt[mask]
    lr = np.zeros_like(gt, dtype=np.float32)
    lr[mask] = np.log(pred[mask]) - np.log(gt[mask])
    return dict(gt=gt, mask=mask, pred=pred, far=far, near=near, enc=enc, err=err, lr=lr)


def window_scores(d, cw, ch, max_disc_frac):
    """Per-window score = summed AbsRel at background-side boundary points, only for 'clean'
    windows (few boundary points among LiDAR points, i.e. an object edge rather than
    see-through clutter such as fences or foliage, where LiDAR GT itself is ambiguous)."""
    h, w = d["gt"].shape
    t, b, l, r = eval_box(h, w)

    def box(x):
        return cv2.boxFilter(x.astype(np.float32), -1, (cw, ch), normalize=False, borderType=cv2.BORDER_CONSTANT)

    s = box(np.where(d["far"], d["err"], 0))
    n_valid = box(d["mask"])
    n_disc = box(d["far"] | d["near"] | d["enc"])
    ok = (n_valid >= 0.04 * cw * ch) & (n_disc <= max_disc_frac * np.maximum(n_valid, 1))
    valid = np.zeros_like(s, bool)
    valid[t + ch // 2:b - ch // 2, l + cw // 2:r - cw // 2] = True
    s[~(valid & ok)] = -1
    return s


def find_window(d, cw, ch, max_disc_frac):
    s = window_scores(d, cw, ch, max_disc_frac)
    cy, cx = np.unravel_index(np.argmax(s), s.shape)
    return cy - ch // 2, cx - cw // 2, float(s[cy, cx])  # top, left, score


def find_row(d, y0, x0, cw, ch):
    """Row with the largest boundary error that has both background- and foreground-side GT."""
    sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
    e = np.where(d["far"][sl], d["err"][sl], 0).sum(1)
    both = (d["far"][sl].sum(1) >= 2) & (d["near"][sl].sum(1) >= 2) & (d["mask"][sl].sum(1) >= 15)
    if both.any():
        e[~both] = -1
    return y0 + int(np.argmax(e))


def depth_rgb(depth, valid, cmap):
    """Colour depth on a log scale; invalid pixels white."""
    x = (np.log(np.clip(depth, DMIN, DMAX)) - np.log(DMIN)) / (np.log(DMAX) - np.log(DMIN))
    rgb = cmap(1 - x)[..., :3]  # near = bright end of the map
    rgb[~valid] = 1.0
    return rgb


def sparse_vis(img, valid, k=3):
    """Dilate sparse LiDAR pixels for full-image display only (zooms stay 1 pixel)."""
    out = img.copy()
    if k > 1:
        idx = cv2.dilate(valid.astype(np.uint8), np.ones((k, k), np.uint8)) > 0
        for c in range(3):
            ch = img[..., c].copy()
            ch[~valid] = 0
            num = cv2.dilate(ch.astype(np.float32), np.ones((k, k), np.uint8))
            out[..., c] = np.where(idx & ~valid, num, out[..., c])
    return out


def class_rgb(d, sl):
    m = d["mask"][sl]
    img = np.ones(m.shape + (3,))
    img[m] = CLS_COLORS["smooth"]
    img[d["far"][sl]] = CLS_COLORS["far"]
    img[d["near"][sl]] = CLS_COLORS["near"]
    img[d["enc"][sl]] = CLS_COLORS["enc"]
    return img


def gt_on_pred(d, sl, pred_full, cmap):
    """Prediction colours with each LiDAR pixel repainted in its true-depth colour (same scale):
    points blend in where the prediction is right and stand out where it is wrong."""
    m = d["mask"][sl]
    img = pred_full[sl].copy()
    img[m] = depth_rgb(d["gt"][sl], m, cmap)[m]
    return img


def signed_rgb(d, sl, cmap):
    m = d["mask"][sl]
    v = np.clip(d["lr"][sl], -0.5, 0.5)
    img = cmap((v + 0.5))[..., :3]
    img[~m] = 1.0
    return img


# Plain-language names used in every legend.
L_PRED = "prediction (Lite-Mono)"
L_FG = "true depth: object (LiDAR)"
L_BG = "true depth: background (LiDAR)"
L_OTHER = "true depth: other (LiDAR)"
L_GLASS = "true depth seen through glass (LiDAR)"
L_TRUE_EDGE = "true edge"
L_PRED_EDGE = "predicted edge"
L_BLEED = "bleeding: background predicted as object"


def legend_handles(include_glass=False):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], color="k", lw=1.6, label=L_PRED),
         Line2D([], [], marker="^", ls="", ms=6, color=CLS_COLORS["near"], mec="k", mew=0.4, label=L_FG),
         Line2D([], [], marker="v", ls="", ms=6, color=CLS_COLORS["far"], mec="k", mew=0.4, label=L_BG),
         Line2D([], [], marker="o", ls="", ms=5, color=CLS_COLORS["smooth"], mec="k", mew=0.4, label=L_OTHER)]
    if include_glass:
        h.append(Line2D([], [], marker="s", ls="", ms=5, color=CLS_COLORS["enc"], mec="k", mew=0.4, label=L_GLASS))
    h += [Line2D([], [], color="tab:green", lw=1.4, ls="--", label=L_TRUE_EDGE),
          Line2D([], [], color="tab:red", lw=1.4, ls="--", label=L_PRED_EDGE),
          Patch(color="tab:red", alpha=0.18, label=L_BLEED)]
    return h


def draw_profile(ax, d, row, x0, cw, e=None, fs=7, ms=14, xlabel=True):
    """Depth along one image row: prediction (line) vs LiDAR points, with the measured edges."""
    from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
    xs = np.arange(x0, x0 + cw)
    pr = d["pred"][row, xs]
    m = d["mask"][row, xs]
    gtr = d["gt"][row, xs]
    if e:
        xp = e["xj"] + e["s"] * e["off"]
        ax.axvspan(min(e["xj"], xp), max(e["xj"], xp), color="tab:red", alpha=0.18, lw=0, zorder=0)
        ax.axvline(e["xj"], color="tab:green", lw=1.3, ls="--", zorder=1)
        ax.axvline(xp, color="tab:red", lw=1.3, ls="--", zorder=1)
    ax.plot(xs, pr, color="k", lw=1.6, zorder=2)
    groups = [(m & ~d["far"][row, xs] & ~d["near"][row, xs] & ~d["enc"][row, xs], "smooth", "o"),
              (d["far"][row, xs], "far", "v"), (d["near"][row, xs], "near", "^"), (d["enc"][row, xs], "enc", "s")]
    for sel, key, mk in groups:
        if sel.any():
            ax.scatter(xs[sel], gtr[sel], s=ms, marker=mk, color=CLS_COLORS[key], edgecolors="k", linewidths=0.35, zorder=3)
    ax.set_yscale("log")
    vals = np.concatenate([pr, gtr[m]])
    lo, hi = max(vals.min() / 1.25, 1.0), min(vals.max() * 1.25, 100.0)
    ax.set_ylim(lo, hi)
    ticks = [t for t in [2, 3, 5, 10, 20, 30, 50, 80] if lo <= t <= hi]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "{:g}".format(v)))
    ax.set_xlim(x0, x0 + cw - 1)
    if e:
        xp = e["xj"] + e["s"] * e["off"]
        from matplotlib.transforms import blended_transform_factory
        ax.text(0.5 * (e["xj"] + xp), 1.02, "{:.0f} px".format(e["off"]), ha="center", va="bottom",
                fontsize=fs, color="tab:red", fontweight="bold",
                transform=blended_transform_factory(ax.transData, ax.transAxes))
    if xlabel:
        ax.set_xlabel("image column (px)", fontsize=fs)
    ax.set_ylabel("depth (m)", fontsize=fs)
    ax.tick_params(labelsize=fs - 1, length=2)
    ax.grid(alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", default=DEFAULT_DIR)
    p.add_argument("--select", choices=["abs_rel", "boundary", "edge"], default="edge")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--crop_w", type=int, default=160)
    p.add_argument("--crop_h", type=int, default=0, help="0 = 52 for --select edge, else 72")
    p.add_argument("--max_disc_frac", type=float, default=0.3)
    p.add_argument("--one_per_drive", type=int, default=1)
    p.add_argument("--min_off", type=float, default=8.0)
    p.add_argument("--max_off", type=float, default=24.0)
    p.add_argument("--min_ratio", type=float, default=2.0)
    p.add_argument("--data_path", default=B.read_paths_env().get("KITTI_ROOT"))
    opt = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Patch
    from matplotlib.lines import Line2D
    from matplotlib.colors import LogNorm
    from matplotlib.cm import ScalarMappable

    cm_depth = plt.get_cmap("turbo")
    cm_err = plt.get_cmap("coolwarm")
    out = os.path.join(opt.analysis_dir, "boundary_zoom", opt.select)
    os.makedirs(out, exist_ok=True)

    disps = np.load(os.path.join(opt.analysis_dir, "disps_eigen.npy"))
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    lines = [l.strip() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]

    cw, ch = opt.crop_w, (opt.crop_h if opt.crop_h > 0 else (52 if opt.select == "edge" else 72))
    import csv
    edge_of = {}
    if opt.select == "edge":
        # edges measured by edge_offset.py: located edges pushed furthest into the background
        # representative, clearly visible cases: offset in [min_off, max_off] GT px and a large jump
        er = [r for r in csv.DictReader(open(os.path.join(opt.analysis_dir, "edges", "edges.csv")))
              if r["status"] == "located" and opt.min_off <= float(r["offset_px"]) <= opt.max_off
              and float(r["d_bg"]) / float(r["d_fg"]) >= opt.min_ratio]
        er.sort(key=lambda r: -float(r["d_bg"]) / float(r["d_fg"]))
        order = []
        for r in er:
            i = int(r["idx"])
            if i not in edge_of:
                edge_of[i] = dict(row=int(r["row"]), xj=float(r["x_jump"]), s=int(r["bg_dir"]),
                                  off=float(r["offset_px"]), dfg=float(r["d_fg"]), dbg=float(r["d_bg"]))
                order.append(i)
        score = np.full(len(gts), -1.0)
        for k, i in enumerate(order):
            score[i] = len(order) - k
    elif opt.select == "abs_rel":
        rows = list(csv.DictReader(open(os.path.join(opt.analysis_dir, "per_image.csv"))))
        score = np.array([float(r["abs_rel"]) for r in rows])
    else:
        score = np.array([find_window(prepare(i, disps, gts), cw, ch, opt.max_disc_frac)[2] for i in range(len(gts))])
    chosen, drives = [], set()
    for i in np.argsort(-score):
        drv = lines[i].split()[0]
        if opt.select != "abs_rel" and opt.one_per_drive and drv in drives:
            continue
        chosen.append(int(i))
        drives.add(drv)
        if len(chosen) == opt.n:
            break
    print("selected:", chosen, [edge_of[i] for i in chosen] if edge_of else "")

    items = []
    for rank, i in enumerate(chosen):
        d = prepare(i, disps, gts)
        rgb = B.load_rgb(opt.data_path, lines[i]).astype(np.float32) / 255
        e = edge_of.get(i)
        if e:
            H, W = d["gt"].shape
            row = e["row"]
            y0 = int(np.clip(row - ch // 2, 0, H - ch))
            x0 = int(np.clip(round(e["xj"]) - cw // 2, 0, W - cw))
        else:
            y0, x0, _ = find_window(d, cw, ch, opt.max_disc_frac)
            row = find_row(d, y0, x0, cw, ch)
        sl = (slice(y0, y0 + ch), slice(x0, x0 + cw))
        g, m, pr = d["gt"], d["mask"], d["pred"]
        a = d["err"][m].mean()
        bm = d["far"] | d["near"] | d["enc"]
        tag = "{} | {} frame {} | image AbsRel {:.3f}".format(
            "idx {}".format(i), lines[i].split()[0].split("/")[1].replace("_sync", ""), int(lines[i].split()[1]), a)
        items.append(dict(i=i, d=d, rgb=rgb, y0=y0, x0=x0, row=row, sl=sl, tag=tag, e=e))

        # ---------- detailed figure ----------
        gt_full = sparse_vis(depth_rgb(g, m, cm_depth), m, 3)
        pred_full = depth_rgb(pr, np.ones_like(m), cm_depth)
        t, b_, l, r = eval_box(*g.shape)
        y_top = max(t - 50, 0)
        err_full = cm_err(np.clip(d["lr"], -0.5, 0.5) + 0.5)[..., :3]
        err_full = sparse_vis(np.where(m[..., None], err_full, 1.0), m, 3)
        err_full = np.where((cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)[..., None],
                            err_full, 0.25 + rgb * 0.35)

        fig = plt.figure(figsize=(13, 7.6))
        gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.15], width_ratios=[1, 1, 1, 1, 0.06],
                              hspace=0.35, wspace=0.06)
        top_panels = [(gs[0, :2], rgb, "(a) input RGB  (box = zoom window)"),
                      (gs[0, 2:4], gt_full, "(b) ground truth depth (LiDAR, sparse; points enlarged)"),
                      (gs[1, :2], pred_full, "(c) Lite-Mono predicted depth (median-scaled)"),
                      (gs[1, 2:4], err_full, "(d) signed error log(pred/GT) at GT points: red = too far, blue = too close")]
        for k, (spec, im, ttl) in enumerate(top_panels):
            ax = fig.add_subplot(spec)
            ax.imshow(im)
            ax.add_patch(Rectangle((x0, y0), cw, ch, fill=False, ec="red" if k == 0 else "k", lw=1.4))
            ax.set_xlim(l, r); ax.set_ylim(b_, y_top); ax.axis("off")
            ax.set_title(ttl, fontsize=8.5, loc="left")
        cax = fig.add_subplot(gs[0:2, 4])
        cb = fig.colorbar(ScalarMappable(norm=LogNorm(DMIN, DMAX), cmap=cm_depth.reversed()), cax=cax)
        cb.set_label("depth (m), panels b, c, f, g", fontsize=7); cb.ax.tick_params(labelsize=6)

        zooms = [(rgb[sl], "(e) zoom: RGB"), (depth_rgb(g[sl], m[sl], cm_depth), "(f) zoom: GT, 1 px = 1 LiDAR point"),
                 (pred_full[sl], "(g) zoom: prediction"),
                 (gt_on_pred(d, sl, pred_full, cm_depth), "(h) zoom: LiDAR points on prediction")]
        for k, (im, ttl) in enumerate(zooms):
            ax = fig.add_subplot(gs[2, k])
            ax.imshow(im, interpolation="nearest")
            ax.set_title(ttl, fontsize=8.5, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(tag, fontsize=9.5, x=0.12, y=0.95, ha="left")
        fig.savefig(os.path.join(out, "idx{:03d}_zoom.png".format(i)), dpi=150, bbox_inches="tight")
        plt.close(fig)


    # ---------- A4 summary: per case, 3 zoomed images on top and the row profile below ----------
    # Sized for an A4 page with ~2 cm margins (17 x 25 cm); fonts stay >= 6.5 pt at 100 %.
    W_IN, H_IN = 6.7, 10.1
    n = len(items)
    fig = plt.figure(figsize=(W_IN, H_IN))
    outer = fig.add_gridspec(n + 1, 1, height_ratios=[1] * n + [0.4], hspace=0.42,
                             left=0.085, right=0.99, top=0.975, bottom=0.03)
    img_aspect = ch / cw
    for k, it in enumerate(items):
        d, sl, y0, x0, row, e = it["d"], it["sl"], it["y0"], it["x0"], it["row"], it["e"]
        g, m = d["gt"], d["mask"]
        inner = outer[k].subgridspec(2, 3, height_ratios=[1.2, 1], hspace=0.2, wspace=0.03)
        panels = [it["rgb"][sl], depth_rgb(g[sl], m[sl], cm_depth),
                  depth_rgb(d["pred"][sl], np.ones_like(m[sl]), cm_depth)]
        heads = ["input image", "true depth (LiDAR, 1 px = 1 point)", "predicted depth (Lite-Mono)"]
        for j, im in enumerate(panels):
            ax = fig.add_subplot(inner[0, j])
            ax.imshow(im, interpolation="nearest", aspect="equal")
            ax.axhline(row - y0, color="white" if j == 0 else "k", lw=0.6 if j == 1 else 0.8,
                       alpha=0.45 if j == 1 else 1.0, ls=(0, (3, 2)))
            if e:
                xp = e["xj"] + e["s"] * e["off"]
                for xv, col in [(e["xj"], "tab:green"), (xp, "tab:red")]:
                    ax.plot([xv - x0, xv - x0], [row - y0 - 9, row - y0 + 9], color=col, lw=1.8)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
            if k == 0:
                ax.set_title(heads[j], fontsize=7.5, pad=3)
        name = CASE_NAMES.get(it["i"], "")
        drv = lines[it["i"]].split()[0].split("/")[1].replace("_sync", "")
        extra = "  object {:.0f} m | background {:.0f} m".format(e["dfg"], e["dbg"]) if e else ""
        fig.axes[-3].text(0, 1.04 if k else 1.22, "({}) {}{}".format(k + 1, name, extra),
            transform=fig.axes[-3].transAxes, fontsize=7.5, fontweight="bold", va="bottom")
        fig.axes[-1].text(1, 1.04 if k else 1.22, "idx {} | {} frame {}".format(it["i"], drv, int(lines[it["i"]].split()[1])),
                          transform=fig.axes[-1].transAxes, fontsize=6, color="0.35", ha="right", va="bottom")
        ax = fig.add_subplot(inner[1, :])
        draw_profile(ax, d, row, x0, cw, e, fs=7, ms=12, xlabel=(k == n - 1))
    # legend + depth colour bar
    foot = outer[n].subgridspec(1, 2, width_ratios=[3.2, 1], wspace=0.15)
    lax = fig.add_subplot(foot[0]); lax.axis("off")
    lax.legend(handles=legend_handles(), loc="center", ncol=2, fontsize=6.8, frameon=False,
               title="graph: depth along the dashed line in the images   |   image ticks: green = true edge, red = predicted edge",
               title_fontsize=6.8)
    cax = fig.add_subplot(foot[1])
    cb = fig.colorbar(ScalarMappable(norm=LogNorm(DMIN, DMAX), cmap=cm_depth.reversed()), cax=cax, orientation="horizontal")
    from matplotlib.ticker import NullLocator
    cb.ax.xaxis.set_minor_locator(NullLocator())
    cb.set_ticks([3, 5, 10, 20, 40, 80]); cb.set_ticklabels(["3", "5", "10", "20", "40", "80"])
    cb.ax.tick_params(labelsize=6.5); cb.set_label("image depth colour (m)", fontsize=6.8)
    pos = cax.get_position()
    cax.set_position([pos.x0, pos.y0 + pos.height * 0.45, pos.width, 0.008])
    fig.savefig(os.path.join(out, "summary_a4.png"), dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(out, "summary_a4.pdf"), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
