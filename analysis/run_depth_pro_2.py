"""Run Apple Depth Pro 2 on the Eigen test split (reference model for the failure analysis).

Run with the isolated env:  third_party/.venv-dp2/bin/python analysis/run_depth_pro_2.py --model depthpro2-s

- Images are fed at native KITTI resolution (as recommended by Depth Pro 2), no resizing.
- The focal length is taken from the KITTI calibration (P_rect_02[0,0]) instead of being
  estimated, so metric depth can also be checked without median scaling.
- Saves <out>/disps_eigen.npy = inverse depth (1/m) at GT resolution, float32, zero-padded to the
  largest KITTI image size (cropped back by baseline_failure_analysis.disp_at). This is the
  format baseline_failure_analysis.py / edge_offset.py read with --reuse_disps, so both
  models go through exactly the same evaluation code (median scaling, 80 m, Garg crop).
- Also prints metric (unscaled) Eigen numbers.

Depth Pro 2: Hu et al., arXiv:2611.26603 (2026); code github.com/apple-aiml-research/ml-depth-pro-2
(commit aa389cb); weights under the Apple ML Research Model license (research use only).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402


def kitti_focal(data_path, date):
    for line in open(os.path.join(data_path, date, "calib_cam_to_cam.txt")):
        if line.startswith("P_rect_02:"):
            return float(line.split()[1])
    raise ValueError("P_rect_02 not found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="depthpro2-s", choices=["depthpro2-s", "depthpro2-b", "depthpro2-l", "depthpro2-h"])
    p.add_argument("--out", default=None)
    p.add_argument("--data_path", default=B.read_paths_env().get("KITTI_ROOT"))
    opt = p.parse_args()
    out = opt.out or os.path.join(B.ROOT, "outputs", "depth_pro_2", opt.model.replace("depthpro2-", ""), "analysis")
    os.makedirs(out, exist_ok=True)

    from depth_pro_2 import create_model_and_transforms
    model, transform = create_model_and_transforms(opt.model, device="cuda")
    fov_ids = {id(q) for q in model.fov.parameters()} if model.fov is not None else set()
    n_params = sum(q.numel() for q in model.parameters() if id(q) not in fov_ids) / 1e6

    lines = [l.strip() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]
    gts = np.load(os.path.join(B.LITEMONO, "splits", "eigen", "gt_depths.npz"),
                  fix_imports=True, encoding="latin1", allow_pickle=True)["data"]
    inv = []
    metric_err = []
    times = []
    torch.cuda.reset_peak_memory_stats()
    for k, line in enumerate(lines):
        rgb = B.load_rgb(opt.data_path, line)
        gt = gts[k]
        assert rgb.shape[:2] == gt.shape, (rgb.shape, gt.shape)
        f_px = kitti_focal(opt.data_path, line.split()[0].split("/")[0])
        torch.cuda.synchronize(); t0 = time.time()
        pred = model.infer(transform(rgb), f_px=f_px)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        depth = pred["depth"].float().cpu().numpy()
        inv.append((1.0 / np.clip(depth, 1e-3, 1e6)).astype(np.float32))
        mask = B.eval_mask(gt)
        d = np.clip(depth[mask], B.MIN_DEPTH, B.MAX_DEPTH)
        metric_err.append(B.compute_errors(gt[mask], d))
        if k % 100 == 0:
            print("{}/{}  {:.3f}s/img".format(k, len(lines), np.mean(times[1:]) if len(times) > 1 else times[0]), flush=True)
    # KITTI image sizes differ by date: zero-pad to the largest; B.disp_at crops back exactly
    H = max(x.shape[0] for x in inv); W = max(x.shape[1] for x in inv)
    arr = np.zeros((len(inv), H, W), np.float32)
    for k, x in enumerate(inv):
        arr[k, :x.shape[0], :x.shape[1]] = x
    np.save(os.path.join(out, "disps_eigen.npy"), arr)
    me = np.array(metric_err).mean(0)
    msg = ["model {} ({:.1f}M params excl. FoV head), native resolution, KITTI focal".format(opt.model, n_params),
           "time {:.3f} s/img (median, batch 1, bf16), peak GPU mem {:.2f} GB".format(
               np.median(times[1:]), torch.cuda.max_memory_allocated() / 1e9),
           "metric (no median scaling): " + " ".join("{}={:.3f}".format(k, v) for k, v in zip(B.METRICS, me))]
    open(os.path.join(out, "run_info.txt"), "w").write("\n".join(msg) + "\n")
    print("\n".join(msg))


if __name__ == "__main__":
    main()
