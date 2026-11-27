"""Pixel-level edge displacement of the predicted depth at LiDAR depth jumps (Eigen test).

For every LiDAR scan row, a GT edge is a pair of adjacent valid points (gap <= MAX_GAP px)
whose depths differ by more than JUMP (1.25x). The edge is kept only when both sides are
consistent surfaces:
  - foreground side: >= 3 points within FG_REACH px, all within 1.15x of the foreground depth
  - background side: >= 5 points within BG_REACH px, all within 1.15x of the background depth
Along the same row the median-scaled prediction is searched for the first pixel (from the
foreground side) where log depth reaches the midpoint between log(fg) and log(bg): the
predicted edge position.
  offset_px > 0 : predicted edge lies inside the background (foreground "fattens" outward)
  offset_px < 0 : predicted edge lies inside the foreground
  status 'bg_missed' : prediction never reaches the midpoint within BG_REACH px; typically LiDAR
                       seeing through glass/fence where the image shows the foreground surface
  status 'fg_missed' : prediction is already past the midpoint FG_REACH px inside the
                       foreground, i.e. the foreground object is not separated at this row
  status 'located'   : offset_px is measured
Distances are in KITTI GT pixels (about 1.94 GT px per network output px at 640 wide).

Outputs in <analysis_dir>/edges/: edges.csv (one row per edge), summary.md
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

DEFAULT_DIR = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")
JUMP, MAX_GAP, FG_REACH, BG_REACH, SURF = 1.25, 3, 15, 40, 1.15


def edges_in_image(gt, mask, pred):
    out = []
    lp_img = np.log(pred)
    for y in np.nonzero(mask.any(1))[0]:
        xs = np.nonzero(mask[y])[0]
        if len(xs) < 10:
            continue
        ds = gt[y, xs]
        gap = np.diff(xs)
        ratio = np.maximum(ds[1:], ds[:-1]) / np.minimum(ds[1:], ds[:-1])
        for j in np.nonzero((gap <= MAX_GAP) & (ratio > JUMP))[0]:
            if ds[j + 1] > ds[j]:  # background to the right
                s, xf, xb, dfg, dbg = 1, xs[j], xs[j + 1], ds[j], ds[j + 1]
            else:
                s, xf, xb, dfg, dbg = -1, xs[j + 1], xs[j], ds[j + 1], ds[j]
            fg_sel = (s * (xf - xs) >= 0) & (s * (xf - xs) <= FG_REACH)
            bg_sel = (s * (xs - xb) >= 0) & (s * (xs - xb) <= BG_REACH)
            fgv, bgv = ds[fg_sel], ds[bg_sel]
            if len(fgv) < 3 or len(bgv) < 5:
                continue
            if np.any(np.abs(np.log(fgv / dfg)) > np.log(SURF)) or np.any(np.abs(np.log(bgv / dbg)) > np.log(SURF)):
                continue
            xj = 0.5 * (xf + xb)
            x_lo, x_hi = (xf - FG_REACH, xb + BG_REACH) if s == 1 else (xb - BG_REACH, xf + FG_REACH)
            x_lo, x_hi = max(int(x_lo), 0), min(int(x_hi), gt.shape[1] - 1)
            path = np.arange(x_lo, x_hi + 1)[::s]  # from the foreground side toward the background
            mid = 0.5 * (np.log(dfg) + np.log(dbg))
            hit = np.nonzero(lp_img[y, path] >= mid)[0]
            if not len(hit):
                offset, status = np.nan, "bg_missed"
            elif hit[0] == 0:
                offset, status = np.nan, "fg_missed"
            else:
                offset, status = s * (path[hit[0]] - xj), "located"
            out.append((int(y), float(xj), s, float(dfg), float(dbg), float(offset), status))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--analysis_dir", default=DEFAULT_DIR)
    opt = p.parse_args()
    out = os.path.join(opt.analysis_dir, "edges")
    os.makedirs(out, exist_ok=True)
    disps = np.load(os.path.join(opt.analysis_dir, "disps_eigen.npy"))
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    rows = []
    for i in range(len(gts)):
        gt = gts[i]
        mask = B.eval_mask(gt)
        pred, _ = B.scaled_pred(disps[i], gt, mask)
        for e in edges_in_image(gt, mask, pred):
            rows.append((i,) + e)
    keys = ["idx", "row", "x_jump", "bg_dir", "d_fg", "d_bg", "offset_px", "status"]
    with open(os.path.join(out, "edges.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        w.writerows(rows)

    dfg = np.array([r[4] for r in rows])
    off = np.array([r[6] for r in rows])
    st = np.array([r[7] for r in rows])

    def breakdown(sel):
        n = sel.sum()
        o = off[sel & (st == "located")]
        return [n, (sel & (st == "located") & (np.abs(off) <= 4)).sum() / n,
                (sel & (st == "located") & (off > 4)).sum() / n, (sel & (st == "located") & (off < -4)).sum() / n,
                (sel & (st == "fg_missed")).sum() / n, (sel & (st == "bg_missed")).sum() / n,
                np.median(o) if len(o) else np.nan]

    md = ["# Predicted edge displacement at LiDAR depth jumps", "",
          "{} clean edges in {} images (both sides consistent surfaces). Offsets in GT px "
          "(~1.94 GT px per network px).".format(len(rows), len(set(r[0] for r in rows))), "",
          "| fg depth | edges | located within +-4 px | pushed into background > 4 px | pushed into foreground > 4 px "
          "| foreground missed | background never reached | median located offset |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, sel in [("all", np.ones(len(rows), bool)), ("0-10 m", dfg < 10), ("10-20 m", (dfg >= 10) & (dfg < 20)),
                      ("20-80 m", dfg >= 20)]:
        b = breakdown(sel)
        md.append("| {} | {} | {:.1%} | {:.1%} | {:.1%} | {:.1%} | {:.1%} | {:+.1f} |".format(name, *b))
    md += ["", "foreground missed = prediction already at background depth 15 px inside the foreground;",
           "background never reached = prediction stays near foreground depth for 40 px into the background "
           "(mostly LiDAR through glass/fence: GT ambiguity, not necessarily a model error)."]
    open(os.path.join(out, "summary.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
