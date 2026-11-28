"""Does the self-supervised photometric loss itself prefer edge fattening? (hypothesis 2 test, no training)

For every LiDAR depth edge found by edge_offset.py (Eigen test, foreground depth d_fg, background
depth d_bg), pixels along the same row are given either d_fg or d_bg, and the training loss of
Lite-Mono / Monodepth2 is evaluated at that pixel exactly as in training:
  640x192 Lanczos-resized frames, KITTI intrinsics, pose from the trained pose network,
  0.85 * SSIM(3x3) + 0.15 * L1, minimum over source frames, identity loss for auto-masking.
The pixel's 3x3 SSIM patch is warped with the same depth (fronto-parallel patch). Depths are
converted to network scale with the per-image median-scaling ratio of the same checkpoint, so
pose and depth scales agree.

  k > 0 : k network pixels into the background (GT: background, true depth d_bg)
  k < 0 : inside the foreground object (GT: foreground, sanity check: loss should prefer d_fg)
  "prefers object depth" : min_s loss(d_fg) < min_s loss(d_bg)
Pixels where the two depths move the warp by < 0.5 px in every source (no parallax, e.g. car
stopped) are counted as "indistinguishable" and left out of the preference rates.

Source sets: training frames {-1, +1}, and {-2, -1, +1, +2} (poses chained from adjacent pairs)
to predict whether more source frames would remove a preference for fattening.

Also: for each edge, whether the loss prefers object depth on the first 1-4 background pixels,
against whether the trained model's prediction bleeds (> 4 px) at that edge.

Usage: .venv/bin/python analysis/loss_preference.py [--weights .../weights_29] [--analysis_dir ...]
Output: <analysis_dir>/loss_preference.md (+ loss_preference.csv, one row per edge pixel)
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import baseline_failure_analysis as B  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import networks  # noqa: E402  (vendor/Lite-Mono on sys.path via B)
from layers import transformation_from_parameters  # noqa: E402

W, H = 640, 192
KS = list(range(-6, 0)) + list(range(1, 17))
K_BINS = [("object -6..-1", -6, -1), ("bg 1-2", 1, 2), ("bg 3-4", 3, 4), ("bg 5-8", 5, 8), ("bg 9-16", 9, 16)]
SETS = {"train {-1,+1}": (-1, 1), "more {-2,-1,+1,+2}": (-2, -1, 1, 2)}
C1, C2 = 0.01 ** 2, 0.03 ** 2


def load_img(data_path, folder, frame, side):
    p = os.path.join(data_path, folder, "image_0{}".format({"l": 2, "r": 3}[side]), "data", "{:010d}.png".format(frame))
    if not os.path.exists(p):
        return None
    im = Image.open(p).convert("RGB").resize((W, H), Image.LANCZOS)
    return torch.from_numpy(np.asarray(im, np.float32) / 255.).permute(2, 0, 1)


class Pose:
    def __init__(self, weights):
        self.enc = networks.ResnetEncoder(18, False, num_input_images=2)
        self.dec = networks.PoseDecoder(self.enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
        for m, f in ((self.enc, "pose_encoder.pth"), (self.dec, "pose.pth")):
            sd = torch.load(os.path.join(weights, f), map_location="cpu")
            m.load_state_dict({k: v for k, v in sd.items() if k in m.state_dict()})
            m.cuda().eval()

    @torch.no_grad()
    def __call__(self, a, b, invert):
        """Same call as Trainer.predict_poses: frames in temporal order (a earlier than b)."""
        x = torch.cat([a, b], 0)[None].cuda()
        axisangle, translation = self.dec([self.enc(x)])
        return transformation_from_parameters(axisangle[:, 0], translation[:, 0], invert=invert)[0]


def intrinsics():
    K = np.array([[0.58, 0, 0.5, 0], [0, 1.92, 0.5, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    K[0, :] *= W
    K[1, :] *= H
    return torch.from_numpy(K).cuda(), torch.from_numpy(np.linalg.pinv(K)).cuda()


OFF = torch.tensor([(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], dtype=torch.float32).cuda()


def sample(img, uv):
    """img (3,H,W) cuda, uv (...,2) pixel coords -> (...,3), bilinear, border padding, align_corners=True."""
    g = torch.stack([uv[..., 0] / (W - 1), uv[..., 1] / (H - 1)], -1) * 2 - 1
    s = F.grid_sample(img[None], g.reshape(1, -1, 1, 2), mode="bilinear", padding_mode="border", align_corners=True)
    return s[0, :, :, 0].t().reshape(*uv.shape[:-1], 3)


def photometric(tgt, src, K, invK, T, px, depth):
    """Loss at integer pixels px (Q,2) when every pixel of its 3x3 patch has depth `depth` (Q,).
    Returns loss (Q,) and the warped centre coordinate (Q,2)."""
    nb = (px[:, None, :] + OFF[None]).clamp(min=0)
    nb[..., 0] = nb[..., 0].clamp(max=W - 1)
    nb[..., 1] = nb[..., 1].clamp(max=H - 1)
    t = tgt[:, nb[..., 1].long(), nb[..., 0].long()].permute(1, 2, 0)  # Q,9,3
    if T is None:  # identity (auto-mask) loss
        w, uv = src[:, nb[..., 1].long(), nb[..., 0].long()].permute(1, 2, 0), px
    else:
        hom = torch.cat([nb, torch.ones_like(nb[..., :1])], -1)  # Q,9,3
        cam = (invK[:3, :3] @ hom.reshape(-1, 3).t()) * depth.repeat_interleave(9)[None]
        cam = (T @ torch.cat([cam, torch.ones_like(cam[:1])], 0))[:3]
        p = K[:3, :3] @ cam
        uvs = (p[:2] / (p[2:3] + 1e-7)).t().reshape(-1, 9, 2)
        w, uv = sample(src, uvs), uvs[:, 4]
    mx, my = t.mean(1), w.mean(1)
    sx = (t ** 2).mean(1) - mx ** 2
    sy = (w ** 2).mean(1) - my ** 2
    sxy = (t * w).mean(1) - mx * my
    ssim = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx ** 2 + my ** 2 + C1) * (sx + sy + C2))
    l_ssim = ((1 - ssim) / 2).clamp(0, 1).mean(1)
    l1 = (t[:, 4] - w[:, 4]).abs().mean(1)
    return 0.85 * l_ssim + 0.15 * l1, uv


def main():
    env = B.read_paths_env()
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/models/weights_29"))
    p.add_argument("--analysis_dir", default=os.path.join(B.ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29"))
    p.add_argument("--data_path", default=env.get("KITTI_ROOT", os.path.join(B.ROOT, "data/raw/kitti")))
    opt = p.parse_args()

    lines = [l.split() for l in open(os.path.join(B.LITEMONO, "splits", "eigen", "test_files.txt")) if l.strip()]
    # per-image median-scaling ratio saved by baseline_failure_analysis.py (avoids loading 1.3 GB of GT)
    ratios = {int(r["idx"]): float(r["ratio"]) for r in csv.DictReader(open(os.path.join(opt.analysis_dir, "per_image.csv")))}
    edges = list(csv.DictReader(open(os.path.join(opt.analysis_dir, "edges", "edges.csv"))))
    by_img = {}
    for e in edges:
        by_img.setdefault(int(e["idx"]), []).append(e)
    pose = Pose(opt.weights)
    K, invK = intrinsics()

    out = []  # one row per (edge, k)
    n_skip = 0
    for i, (folder, frame, side) in enumerate(lines):
        if i not in by_img:
            continue
        frame = int(frame)
        imgs = {o: load_img(opt.data_path, folder, frame + o, side) for o in (-2, -1, 0, 1, 2)}
        if any(imgs[o] is None for o in (-1, 0, 1)):
            n_skip += len(by_img[i])
            continue
        gpu = {o: v.cuda() for o, v in imgs.items() if v is not None}
        T = {-1: pose(imgs[-1], imgs[0], True), 1: pose(imgs[0], imgs[1], False)}
        if -2 in gpu:
            T[-2] = pose(imgs[-2], imgs[-1], True) @ T[-1]
        if 2 in gpu:
            T[2] = pose(imgs[1], imgs[2], False) @ T[1]
        ratio = ratios[i]
        gw, gh = Image.open(os.path.join(opt.data_path, folder, "image_02", "data", "{:010d}.png".format(frame))).size
        es = by_img[i]
        q_px, q_dfg, q_dbg, q_meta = [], [], [], []
        for e in es:
            y = int(round((int(e["row"]) + 0.5) * H / gh - 0.5))
            x0 = (float(e["x_jump"]) + 0.5) * W / gw - 0.5
            s = int(e["bg_dir"])
            # colour contrast across the edge: |I(object, k=-2) - I(background, k=+2)|, mean over RGB
            xa, xb = int(round(x0 - s * 1.5)), int(round(x0 + s * 1.5))
            e["contrast"] = float((imgs[0][:, min(max(y, 0), H - 1), min(max(xa, 0), W - 1)]
                                   - imgs[0][:, min(max(y, 0), H - 1), min(max(xb, 0), W - 1)]).abs().mean())
            for k in KS:
                x = int(round(x0 + s * (k - 0.5 if k > 0 else k + 0.5)))
                if 0 <= x < W and 0 <= y < H:
                    q_px.append((x, y))
                    q_dfg.append(float(e["d_fg"]) / ratio)
                    q_dbg.append(float(e["d_bg"]) / ratio)
                    q_meta.append((e, k, x))
        if not q_px:
            continue
        px = torch.tensor(q_px, dtype=torch.float32).cuda()
        dfg = torch.tensor(q_dfg).cuda()
        dbg = torch.tensor(q_dbg).cuda()
        L = {}
        with torch.no_grad():
            for o in T:
                lf, uvf = photometric(gpu[0], gpu[o], K, invK, T[o], px, dfg)
                lb, uvb = photometric(gpu[0], gpu[o], K, invK, T[o], px, dbg)
                li, _ = photometric(gpu[0], gpu[o], K, invK, None, px, dbg)
                L[o] = (lf, lb, li, (uvf - uvb).norm(dim=1))
        for name, srcs in SETS.items():
            srcs = [o for o in srcs if o in L]
            if name.startswith("more") and len(srcs) < 4:
                continue
            lf = torch.stack([L[o][0] for o in srcs]).min(0).values.cpu().numpy()
            lb = torch.stack([L[o][1] for o in srcs]).min(0).values.cpu().numpy()
            li = torch.stack([L[o][2] for o in srcs]).min(0).values.cpu().numpy()
            par = torch.stack([L[o][3] for o in srcs]).max(0).values.cpu().numpy()
            for j, (e, k, x) in enumerate(q_meta):
                outer = (int(e["bg_dir"]) > 0) == (x > W / 2)  # background on the side away from the image centre
                out.append(dict(idx=i, row=e["row"], x_jump=e["x_jump"], k=k, set=name, d_fg=float(e["d_fg"]),
                                status=e["status"], offset_px=e["offset_px"], outer=int(outer), contrast=e["contrast"],
                                loss_fg=float(lf[j]), loss_bg=float(lb[j]), loss_id=float(li[j]), parallax=float(par[j])))
        if i % 100 == 0:
            print("image", i, flush=True)

    with open(os.path.join(opt.analysis_dir, "loss_preference.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    md = summarize(out, n_skip, opt)
    open(os.path.join(opt.analysis_dir, "loss_preference.md"), "w").write(md)
    print(md)


def summarize(out, n_skip, opt):
    def arr(rows, key):
        return np.array([r[key] for r in rows])

    md = ["# Does the photometric loss prefer the object's depth beside the object?", "",
          "weights: {}  (edges and median-scaling from {})".format(opt.weights, opt.analysis_dir),
          "Edges skipped (missing neighbour frames): {}".format(n_skip), "",
          "Cell: share of pixels where min_s loss(object depth) < min_s loss(background depth), among pixels",
          "where the two depths move the warp by >= 0.5 px; in brackets: share auto-masked (identity loss lower",
          "than both). k = network pixels from the LiDAR edge; k < 0 is the object itself (sanity check).", ""]
    for near_name, near in (("all edges", lambda r: True), ("object < 20 m", lambda r: r["d_fg"] < 20)):
        md += ["## {}".format(near_name), "", "| source frames | side | " + " | ".join(b[0] for b in K_BINS) + " |",
               "|---|---|" + "---:|" * len(K_BINS)]
        for name in SETS:
            for side_name, side in (("both", None), ("outer", 1), ("inner", 0)):
                cells = []
                for _, lo, hi in K_BINS:
                    rs = [r for r in out if r["set"] == name and lo <= r["k"] <= hi and near(r)
                          and (side is None or r["outer"] == side)]
                    rs_p = [r for r in rs if r["parallax"] >= 0.5]
                    if not rs_p:
                        cells.append("-")
                        continue
                    pref = np.mean(arr(rs_p, "loss_fg") < arr(rs_p, "loss_bg"))
                    masked = np.mean(arr(rs_p, "loss_id") < np.minimum(arr(rs_p, "loss_fg"), arr(rs_p, "loss_bg")))
                    cells.append("{:.0%} ({:.0%}) n={}".format(pref, masked, len(rs_p)))
                md.append("| {} | {} | ".format(name, side_name) + " | ".join(cells) + " |")
        md.append("")

    md += ["## Loss preference near the edge vs. the trained model's bleeding", "",
           "Per edge: 'loss prefers object' = on bg pixels k = 1..4 (with parallax), object depth wins on more than half.",
           "", "| source frames | edges | loss prefers object | P(model bleeds > 4 px), loss prefers object | "
           "P(model bleeds), loss prefers background | P(object missed), prefers object | P(object missed), prefers bg |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    cons = np.array([r["contrast"] for r in out if r["set"] == "train {-1,+1}" and r["k"] == 1])
    t1, t2 = np.percentile(cons, [33.3, 66.7])
    strata = [("", lambda c: True), (", low contrast", lambda c: c < t1), (", mid contrast", lambda c: t1 <= c < t2),
              (", high contrast", lambda c: c >= t2)]
    for (sname, sfun), name in [(st, n) for n in SETS for st in strata]:
        per = {}
        for r in out:
            if r["set"] == name and 1 <= r["k"] <= 4 and r["parallax"] >= 0.5 and sfun(r["contrast"]):
                per.setdefault((r["idx"], r["row"], r["x_jump"]), []).append(r)
        if not per:
            continue
        lp, bleed, miss = [], [], []
        for rs in per.values():
            lp.append(np.mean(arr(rs, "loss_fg") < arr(rs, "loss_bg")) > 0.5)
            r0 = rs[0]
            loc = r0["status"] == "located"
            bleed.append(loc and float(r0["offset_px"]) > 4)
            miss.append(r0["status"] == "fg_missed")
        lp, bleed, miss = map(np.array, (lp, bleed, miss))
        md.append("| {} | {} | {:.0%} | {:.0%} | {:.0%} | {:.0%} | {:.0%} |".format(
            name + sname, len(lp), lp.mean(), bleed[lp].mean(), bleed[~lp].mean(), miss[lp].mean(), miss[~lp].mean()))
    md.append("")
    md.append("Contrast terciles (mean |RGB| difference across the edge, 0-1): {:.3f}, {:.3f}".format(t1, t2))
    return "\n".join(md) + "\n"


if __name__ == "__main__":
    main()
