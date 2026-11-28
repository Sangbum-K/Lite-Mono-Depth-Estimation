"""Driving-oriented metrics for the 5-epoch pilots (report section 6.1 criteria), from saved predictions.

Uses <run>/analysis/weights_4/disps_eigen.npy and edges/edges.csv written by
baseline_failure_analysis.py / edge_offset.py (no network is run). Same protocol as the Eigen
evaluation: Garg crop, cap 80 m, per-image median scaling. Only LiDAR points closer than 20 m
(GT) are used, split into 0-10 m and 10-20 m.

  AbsRel                 : on all near LiDAR points
  farther >25%           : pred > 1.25 gt (object reported farther than it is: late braking)
  nearer >25%            : pred < gt / 1.25 (reported nearer: conservative, false obstacles)
  object side, farther   : near-side points of a GT depth jump (the object's own contour) with
                           pred > 1.25 gt, i.e. the object edge is eaten by the background
  free space blocked     : far-side points next to a near object (outer contour, not glass/fence)
                           with pred < gt / 1.25, i.e. the gap beside the object looks occupied
  edge outcomes          : edges.csv rows with foreground depth < 20 m (within 4 px / bleed into
                           background > 4 px / foreground missed)

Differences to baseline ep5 come with a paired bootstrap 95% interval over the 697 test images.
This covers test-set sampling only, not training randomness (runs are unseeded, one run each).

Output: outputs/pilots_ad.md
"""
import csv
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

OUT = os.path.join(B.ROOT, "outputs")
RUNS = [("Baseline ep5", "20261109_baseline_unseeded", "weights_4"),
        ("B smooth 1e-4", "20261127_smooth1e-4_e5", "weights_4"),
        ("C gain 10", "20261127_smoothgain10_e5", "weights_4"),
        ("A-lite", "20261127_hires_lite_e5", "weights_4"),
        ("A", "20261127_hires_decoder_e5", "weights_4"),
        ("E DPT 24", "20261128_dpt24_e5", "weights_4"),
        ("(ref) Baseline ep30", "20261109_baseline_unseeded", "weights_29")]
BINS = [(0, 10), (10, 20)]
T = 1.25
N_BOOT = 1000


def near_neighbour_depth(gt, win=9):
    valid = gt > 0
    k = np.ones((win, win), np.uint8)
    return -cv2.dilate(np.where(valid, -gt, -1e6).astype(np.float32), k)


def per_image_counts(disps, gts):
    """Return dict name -> (num[N], den[N]) so that metric = sum(num) / sum(den)."""
    keys = []
    for lo, hi in BINS:
        keys += ["absrel_{}".format(lo), "farther_{}".format(lo), "nearer_{}".format(lo),
                 "objside_farther_{}".format(lo), "freespace_{}".format(lo)]
    num = {k: np.zeros(len(gts)) for k in keys}
    den = {k: np.zeros(len(gts)) for k in keys}
    for i in range(len(gts)):
        gt = gts[i]
        mask = B.eval_mask(gt)
        pred, _ = B.scaled_pred(disps[i], gt, mask)
        far_s, near_s = B.gt_discontinuity(gt, sides=True)
        enc = B.gt_enclosed(gt, far_s)
        dmin = near_neighbour_depth(gt)
        for lo, hi in BINS:
            sel = mask & (gt >= lo) & (gt < hi)
            g, p = gt[sel], pred[sel]
            num["absrel_{}".format(lo)][i], den["absrel_{}".format(lo)][i] = (np.abs(g - p) / g).sum(), sel.sum()
            num["farther_{}".format(lo)][i], den["farther_{}".format(lo)][i] = (p > T * g).sum(), sel.sum()
            num["nearer_{}".format(lo)][i], den["nearer_{}".format(lo)][i] = (p * T < g).sum(), sel.sum()
            s = sel & near_s
            num["objside_farther_{}".format(lo)][i], den["objside_farther_{}".format(lo)][i] = \
                (pred[s] > T * gt[s]).sum(), s.sum()
            # background next to an object whose own depth is in this bin
            s = mask & far_s & ~enc & (dmin >= lo) & (dmin < hi)
            num["freespace_{}".format(lo)][i], den["freespace_{}".format(lo)][i] = \
                (pred[s] * T < gt[s]).sum(), s.sum()
    return num, den


def edge_counts(path, n_img):
    rows = list(csv.DictReader(open(path)))
    out = {}
    for lo, hi in BINS:
        rs = [r for r in rows if lo <= float(r["d_fg"]) < hi]
        for name, f in [("within4", lambda r: r["status"] == "located" and abs(float(r["offset_px"])) <= 4),
                        ("bleed", lambda r: r["status"] == "located" and float(r["offset_px"]) > 4),
                        ("fgmiss", lambda r: r["status"] == "fg_missed")]:
            nu, de = np.zeros(n_img), np.zeros(n_img)
            for r in rs:
                de[int(r["idx"])] += 1
                nu[int(r["idx"])] += f(r)
            out["{}_{}".format(name, lo)] = (nu, de)
    return out


def main():
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    res = {}
    for label, run, w in RUNS:
        ana = os.path.join(OUT, run, "analysis", w)
        disps = np.load(os.path.join(ana, "disps_eigen.npy"))
        num, den = per_image_counts(disps, gts)
        c = {k: (num[k], den[k]) for k in num}
        c.update(edge_counts(os.path.join(ana, "edges", "edges.csv"), len(gts)))
        res[label] = c
        print("done", label, flush=True)

    rng = np.random.default_rng(0)
    boot = rng.integers(0, len(gts), size=(N_BOOT, len(gts)))
    base = res[RUNS[0][0]]

    def val(c, k, idx=None):
        nu, de = c[k]
        if idx is None:
            return nu.sum() / de.sum()
        return nu[idx].sum(1) / de[idx].sum(1)

    rows = [("AbsRel", "absrel", False), ("farther >25% (late braking)", "farther", True),
            ("nearer >25% (false obstacle)", "nearer", True),
            ("object side of edge, farther >25%", "objside_farther", True),
            ("free space next to object blocked", "freespace", True),
            ("edge within 4 px", "within4", True), ("edge bleed into bg >4 px", "bleed", True),
            ("object edge missed", "fgmiss", True)]
    md = ["# Driving-oriented metrics, LiDAR points < 20 m (Eigen test, median scaling)", "",
          "Cell: value, then difference to Baseline ep5 with paired bootstrap 95% interval over test images",
          "(test-set sampling only; training randomness not included: one unseeded run each).", ""]
    for lo, hi in BINS:
        md += ["## {}-{} m".format(lo, hi), "",
               "| metric | " + " | ".join(r[0] for r in RUNS) + " |",
               "|---|" + "---:|" * len(RUNS)]
        for name, key, pct in rows:
            k = "{}_{}".format(key, lo)
            cells = []
            for label, _, _ in RUNS:
                v = val(res[label], k)
                s = "{:.1%}".format(v) if pct else "{:.3f}".format(v)
                if label != RUNS[0][0]:
                    d = val(res[label], k, boot) - val(base, k, boot)
                    lo_ci, hi_ci = np.percentile(d, [2.5, 97.5])
                    dv = v - val(base, k)
                    sig = "*" if lo_ci > 0 or hi_ci < 0 else ""
                    if pct:
                        s += " ({:+.1f} [{:+.1f},{:+.1f}]{})".format(100 * dv, 100 * lo_ci, 100 * hi_ci, sig)
                    else:
                        s += " ({:+.3f} [{:+.3f},{:+.3f}]{})".format(dv, lo_ci, hi_ci, sig)
                cells.append(s)
            n = int(base[k][1].sum())
            md.append("| {} (n={}) | ".format(name, n) + " | ".join(cells) + " |")
        md.append("")
    md.append("* = interval excludes 0. Percent differences are in percentage points.")
    path = os.path.join(OUT, "pilots_ad.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", path)


if __name__ == "__main__":
    main()
