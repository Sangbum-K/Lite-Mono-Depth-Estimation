"""Table of the 5-epoch pilot runs against the baseline epoch-5 checkpoint (weights_4).

For each run it reads <analysis_dir>/per_image.csv (baseline_failure_analysis.py), summary.md
(pooled region / distance rows) and edges/edges.csv (edge_offset.py); all runs go through the
same evaluation code. Missing runs are listed as "-".

Usage: .venv/bin/python analysis/pilot_summary.py [--out outputs/pilots_summary.md]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402
from compare_models import edge_outcomes, pooled  # noqa: E402

RUNS = [  # (label, hypothesis, analysis dir relative to outputs/)
    ("Baseline (ep5)", "-", "20261109_baseline_unseeded/analysis/weights_4"),
    ("B smoothness 1e-4", "2", "20261127_smooth1e-4_e5/analysis/weights_4"),
    ("C edge gain 10", "2", "20261127_smoothgain10_e5/analysis/weights_4"),
    ("A-lite skip8+subpixel", "1", "20261127_hires_lite_e5/analysis/weights_4"),
    ("A skip48+deconv", "1", "20261127_hires_decoder_e5/analysis/weights_4"),
    ("Baseline rerun (ep5)", "-", "20261128_baseline_rerun_e5/analysis/weights_4"),
    ("E DPT decoder dim24", "1", "20261128_dpt24_e5/analysis/weights_4"),
]
COST = {"Baseline (ep5)": "5.03G", "B smoothness 1e-4": "+0%", "C edge gain 10": "+0%",
        "A-lite skip8+subpixel": "+1.7%", "A skip48+deconv": "+10.3%", "Baseline rerun (ep5)": "+0%",
        "E DPT decoder dim24": "+2.0%"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(B.ROOT, "outputs/pilots_summary.md"))
    opt = p.parse_args()
    rows = []
    for label, hyp, rel in RUNS:
        d = os.path.join(B.ROOT, "outputs", rel)
        if not os.path.exists(os.path.join(d, "per_image.csv")):
            rows.append([label, hyp, COST[label]] + ["-"] * 9)
            continue
        r = list(csv.DictReader(open(os.path.join(d, "per_image.csv"))))
        m = {k: np.mean([float(x[k]) for x in r]) for k in ("abs_rel", "sq_rel", "rmse", "a1")}
        pl = pooled(d)
        has_edges = os.path.exists(os.path.join(d, "edges", "edges.csv"))
        eo = edge_outcomes(d) if has_edges else None
        f = lambda k: "{:.1%}".format(eo[k]) if eo else "-"
        rows.append([label, hyp, COST[label], "{:.3f}".format(m["abs_rel"]), "{:.3f}".format(m["sq_rel"]),
                     "{:.3f}".format(m["rmse"]), "{:.3f}".format(m["a1"]),
                     "{:.3f}".format(pl["dist_0-20m"][0]), "{:.3f}".format(pl["region_disc_near_side"][0]),
                     f("within +-4 px"), f("into background > 4 px"), f("foreground missed")])
    head = ["run", "hyp.", "MACs", "AbsRel", "SqRel", "RMSE", "d1", "AbsRel 0-20m", "AbsRel object side of edge",
            "edge within 4px", "bleed into bg >4px", "fg missed"]
    md = ["# 5-epoch pilots vs baseline epoch 5 (Eigen test, same evaluation code)", "",
          "| " + " | ".join(head) + " |", "|---|---|---|" + "---:|" * (len(head) - 3)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    md += ["", "Runs are unseeded: treat differences smaller than the epoch-to-epoch spread of the baseline as noise."]
    open(opt.out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
