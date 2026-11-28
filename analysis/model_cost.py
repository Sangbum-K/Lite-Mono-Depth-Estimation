"""Parameters, MACs and inference time of the depth network variants (encoder + depth decoder, 640x192).

Baseline, the 5-epoch pilot decoders (A, A-lite) and the Depth Pro 2 / DPT decoder (--dpt_decoder)
at several widths. Pose network not included (not used at test time).
  MACs: thop.  Time: batch 1, FP32 PyTorch eager, random weights (time does not depend on values).
  GPU: cudnn.benchmark, WARM warm-up runs, then N_GPU runs timed with CUDA events; median and p90.
       Also batch 16, reported per image (batch time / 16), the setting of Table 4 in the Lite-Mono
       paper (TITAN Xp 4.5 ms, Jetson Xavier 20.0 ms; batch 1 in the official README: 9.1 / 24.6 ms).
  CPU: torch.set_num_threads(CPU_THREADS), N_CPU runs after warm-up; median.
The GPU is the training desktop GPU, not an in-vehicle computer: compare variants relative to the
baseline, absolute numbers do not transfer to embedded hardware.
Output: outputs/model_cost.md
Usage: .venv/bin/python analysis/model_cost.py   (nothing else should run on the GPU)
"""
import os
import sys
import time

import numpy as np
import torch
from thop import profile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "Lite-Mono"))
import networks  # noqa: E402

H, W = 192, 640
DPT = [(64, 32), (32, 16), (24, 16), (16, 16)]  # (dpt_dim, dpt_head_ch); (64, 32) = Depth Pro 2-S widths
WARM, N_GPU, N_CPU, CPU_THREADS = 50, 500, 50, 4


def build(stem, make_dec):
    torch.manual_seed(0)
    enc = networks.LiteMono(model="lite-mono", height=H, width=W, return_stem=stem).eval()
    return enc, make_dec(enc).eval()


def macs_params(stem, make_dec):
    enc, dec = build(stem, make_dec)
    x = torch.randn(1, 3, H, W)
    with torch.no_grad():
        f = enc(x)
        out = dec(f)
    assert tuple(out[("disp", 0)].shape[-2:]) == (H, W)
    me, _ = profile(enc, inputs=(x,), verbose=False)
    md, _ = profile(dec, inputs=(f,), verbose=False)
    return me + md, sum(p.numel() for m in (enc, dec) for p in m.parameters())


def time_gpu(stem, make_dec, bs=1):
    enc, dec = build(stem, make_dec)
    enc, dec = enc.cuda(), dec.cuda()
    x = torch.randn(bs, 3, H, W, device="cuda")
    ts = []
    with torch.no_grad():
        for k in range(WARM + N_GPU):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            dec(enc(x))[("disp", 0)]
            b.record()
            torch.cuda.synchronize()
            if k >= WARM:
                ts.append(a.elapsed_time(b) / bs)
    return float(np.median(ts)), float(np.percentile(ts, 90))


def time_cpu(stem, make_dec):
    enc, dec = build(stem, make_dec)
    x = torch.randn(1, 3, H, W)
    ts = []
    with torch.no_grad():
        for k in range(5 + N_CPU):
            t0 = time.perf_counter()
            dec(enc(x))
            if k >= 5:
                ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts))


def main():
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(CPU_THREADS)
    variants = [("Baseline (Lite-Mono)", "", False, lambda e: networks.DepthDecoder(e.num_ch_enc, range(3))),
                ("A", "--hires_decoder", True, lambda e: networks.HiResDepthDecoder(e.num_ch_enc, range(3))),
                ("A-lite", "--hires_decoder --hires_stem_ch 8 --hires_upsample subpixel", True,
                 lambda e: networks.HiResDepthDecoder(e.num_ch_enc, range(3), stem_ch=8, upsample_head="subpixel"))]
    for d, h in DPT:
        variants.append(("E: DPT dim {} head {}".format(d, h), "--dpt_decoder --dpt_dim {} --dpt_head_ch {}".format(d, h),
                         False, lambda e, d=d, h=h: networks.DPTDepthDecoder(e.num_ch_enc, range(3), dim=d, head_ch=h)))
    rows = []
    for name, opts, stem, mk in variants:
        m, p = macs_params(stem, mk)
        g50, g90 = time_gpu(stem, mk)
        g16, _ = time_gpu(stem, mk, bs=16)
        c50 = time_cpu(stem, mk)
        rows.append((name, opts, m, p, g50, g90, g16, c50))
        print(name, "done", flush=True)
    _, _, m0, p0, g0, _, b0, c0 = rows[0]
    md = ["# Depth network cost ({}x{}, encoder + depth decoder)".format(W, H), "",
          "MACs: thop. Time: batch 1, FP32 PyTorch, GPU {} (median / p90 of {} runs), CPU {} threads "
          "(median of {} runs). Desktop hardware: compare relative to the baseline.".format(
              torch.cuda.get_device_name(0), N_GPU, CPU_THREADS, N_CPU), "",
          "| variant | options | MACs | vs base | params | vs base | GPU ms, batch 1 (median / p90) | vs base | GPU ms/image, batch 16 | vs base | CPU ms, batch 1 | vs base |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, opts, m, p, g50, g90, g16, c50 in rows:
        md.append("| {} | {} | {:.3f}G | {:+.1f}% | {:,} | {:+.2f}% | {:.2f} / {:.2f} | {:+.0f}% | {:.2f} | {:+.0f}% | {:.1f} | {:+.0f}% |".format(
            name, "`{}`".format(opts) if opts else "(기본)", m / 1e9, 100 * (m / m0 - 1), p, 100 * (p / p0 - 1),
            g50, g90, 100 * (g50 / g0 - 1), g16, 100 * (g16 / b0 - 1), c50, 100 * (c50 / c0 - 1)))
    path = os.path.join(ROOT, "outputs", "model_cost.md")
    open(path, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
