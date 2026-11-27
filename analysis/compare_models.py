"""Paired per-image comparison of Lite-Mono with reference models (e.g. Depth Pro 2) on Eigen test.

Inputs: analysis dirs written by baseline_failure_analysis.py (per_image.csv, summary.md) and
edge_offset.py (edges/edges.csv) for each model. All models use the same evaluation code.

Outputs in --out:
  compare.md         overall / distance bins / boundary regions / edge outcomes side by side,
                     Lite-Mono worst-30% subset, "only Lite-Mono fails" vs "all models fail" lists
  scatter_<ref>.png  per-image AbsRel, Lite-Mono vs reference
"""
import argparse
import csv
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

LM = os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")


def per_image(d):
    rows = list(csv.DictReader(open(os.path.join(d, "per_image.csv"))))
    return rows, np.array([float(r["abs_rel"]) for r in rows]), np.array([float(r["a1"]) for r in rows])


def pooled(d):
    """group -> (abs_rel, a1, mean log ratio) from summary.md."""
    out = {}
    for line in open(os.path.join(d, "summary.md")):
        m = re.match(r"\| (dist_\S+|region_\S+) \| \d+ \| [\d.]+% \| [\d.]+% \| ([\d.]+) \| [\d.]+ \| [\d.]+ \| [\d.]+ \| ([\d.]+) \| ([+-][\d.]+) \|", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out


def edge_outcomes(d):
    rows = list(csv.DictReader(open(os.path.join(d, "edges", "edges.csv"))))
    st = np.array([r["status"] for r in rows])
    off = np.array([float(r["offset_px"]) if r["offset_px"] not in ("", "nan") else np.nan for r in rows])
    n = len(rows)
    loc = st == "located"
    return {"within +-4 px": (loc & (np.abs(off) <= 4)).sum() / n, "into background > 4 px": (loc & (off > 4)).sum() / n,
            "into foreground > 4 px": (loc & (off < -4)).sum() / n, "foreground missed": (st == "fg_missed").mean(),
            "background never reached": (st == "bg_missed").mean()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", nargs="+", required=True, help="name=analysis_dir")
    p.add_argument("--out", default=os.path.join(B.ROOT, "outputs/depth_pro_2/compare"))
    p.add_argument("--worst_pct", type=float, default=30)
    opt = p.parse_args()
    os.makedirs(opt.out, exist_ok=True)
    models = [("Lite-Mono", LM)] + [tuple(r.split("=", 1)) for r in opt.ref]

    data = {n: per_image(d) for n, d in models}
    rows_lm, a_lm, _ = data["Lite-Mono"]
    n = len(a_lm)
    worst = np.argsort(-a_lm)[:int(round(n * opt.worst_pct / 100))]
    md = ["# Lite-Mono vs reference models (Eigen test, same evaluation code, median scaling)", ""]

    md += ["## Overall and Lite-Mono worst {:g}% subset ({} images)".format(opt.worst_pct, len(worst)), "",
           "| model | AbsRel all | d1 all | AbsRel on LM-worst | d1 on LM-worst | images better than LM |",
           "|---|---:|---:|---:|---:|---:|"]
    for name, _ in models:
        _, a, d1 = data[name]
        md.append("| {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {} |".format(
            name, a.mean(), d1.mean(), a[worst].mean(), d1[worst].mean(),
            "-" if name == "Lite-Mono" else "{:.0%}".format((a < a_lm).mean())))

    pooled_all = {name: pooled(d) for name, d in models}
    groups = ["dist_0-20m", "dist_20-40m", "dist_40-80m", "region_gt_smooth", "region_disc_far_edge",
              "region_disc_far_enclosed", "region_disc_near_side"]
    md += ["", "## Pooled pixel AbsRel (mean log(pred/GT); < 0 = too close)", "",
           "| group | " + " | ".join(n for n, _ in models) + " |", "|---|" + "---:|" * len(models)]
    for g in groups:
        md.append("| {} | ".format(g) + " | ".join("{:.3f} ({:+.3f})".format(pooled_all[n][g][0], pooled_all[n][g][2])
                                                  for n, _ in models) + " |")

    eo = {name: edge_outcomes(d) for name, d in models}
    md += ["", "## Edge outcomes at the same 3263 LiDAR jumps (edge_offset.py)", "",
           "| outcome | " + " | ".join(n for n, _ in models) + " |", "|---|" + "---:|" * len(models)]
    for k in eo["Lite-Mono"]:
        md.append("| {} | ".format(k) + " | ".join("{:.1%}".format(eo[n][k]) for n, _ in models) + " |")

    # image categories against the strongest reference (last one given)
    rname = models[-1][0]
    _, a_ref, _ = data[rname]
    thr_lm = np.percentile(a_lm, 100 - opt.worst_pct)
    thr_ref = np.percentile(a_ref, 100 - opt.worst_pct)
    only_lm = np.nonzero((a_lm >= thr_lm) & (a_ref < np.percentile(a_ref, 50)))[0]
    both = np.nonzero((a_lm >= thr_lm) & (a_ref >= thr_ref))[0]
    only_lm = only_lm[np.argsort(-(a_lm[only_lm] - a_ref[only_lm]))]
    both = both[np.argsort(-(a_lm[both] + a_ref[both]))]

    def fmt(ids, k=15):
        out = ["| idx | drive | frame | AbsRel LM | AbsRel {} |".format(rname), "|---:|---|---:|---:|---:|"]
        for i in ids[:k]:
            r = rows_lm[i]
            out.append("| {} | {} | {} | {:.3f} | {:.3f} |".format(i, r["folder"].split("/")[1].replace("_sync", ""),
                                                                  r["frame"], a_lm[i], a_ref[i]))
        return out

    from collections import Counter
    md += ["", "## Image categories (vs {})".format(rname), "",
           "- LM worst {:g}% but {} better than its median: {} images (Lite-Mono-specific failures)".format(opt.worst_pct, rname, len(only_lm)),
           "- LM worst {:g}% and {} worst {:g}%: {} images (hard for both: scene or GT problems)".format(opt.worst_pct, rname, opt.worst_pct, len(both)),
           "- drives, LM-specific: {}".format(dict(Counter(rows_lm[i]["folder"].split("/")[1][-9:-5] for i in only_lm).most_common(6))),
           "- drives, both: {}".format(dict(Counter(rows_lm[i]["folder"].split("/")[1][-9:-5] for i in both).most_common(6))),
           "", "### Largest Lite-Mono-specific failures", ""] + fmt(only_lm) + ["", "### Hardest for both", ""] + fmt(both)
    open(os.path.join(opt.out, "compare.md"), "w").write("\n".join(md) + "\n")
    np.save(os.path.join(opt.out, "only_lm.npy"), only_lm)
    np.save(os.path.join(opt.out, "both.npy"), both)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, _ in models[1:]:
        a = data[name][1]
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.scatter(a, a_lm, s=6, alpha=0.5, color="#4a6fa5", linewidths=0)
        lim = max(a.max(), a_lm.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", lw=0.8)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("per-image AbsRel, {}".format(name)); ax.set_ylabel("per-image AbsRel, Lite-Mono")
        ax.set_title("above the line: Lite-Mono worse ({:.0%} of images)".format((a_lm > a).mean()), fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(opt.out, "scatter_{}.png".format(re.sub(r"\W+", "_", name))), dpi=150)
        plt.close(fig)
    print("\n".join(md))


if __name__ == "__main__":
    main()
