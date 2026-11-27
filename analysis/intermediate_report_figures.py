"""Crop/stack existing analysis figures into the figures used by the intermediate report.

No new computation: every panel is cut from a figure that the analysis scripts already made.
Output: reports/figures/intermediate/fig{1..4}_*.png

  fig1_qualitative.png  overview_random.png rows idx 28 / 121 / 213 (RGB | disparity | AbsRel at LiDAR)
  fig2_overlay.png      boundary_overlay/overlay_a4.png (copied as is)
  fig3_idx529.png       boundary_zoom/edge/idx529_zoom.png (copied as is)
  fig4_depthpro2.png    rows of depth_pro_2/compare/sheet_both_h.png and sheet_only_lm_h.png

Usage: .venv/bin/python analysis/intermediate_report_figures.py
"""
import os
import shutil

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A29 = os.path.join(ROOT, "outputs/20261109_baseline_unseeded/analysis/weights_29")
DP2 = os.path.join(ROOT, "outputs/depth_pro_2/compare")
OUT = os.path.join(ROOT, "reports/figures/intermediate")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def vstack(parts, pad=0, bg="white"):
    w = max(p.width for p in parts)
    h = sum(p.height for p in parts) + pad * (len(parts) - 1)
    out = Image.new("RGB", (w, h), bg)
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + pad
    return out


def label_strip(text, width, height=30, size=17):
    im = Image.new("RGB", (width, height), "white")
    ImageDraw.Draw(im).text((16, 6), text, fill="black", font=ImageFont.truetype(FONT_B, size))
    return im


def fig1():
    src = Image.open(os.path.join(A29, "figs/overview_random.png")).convert("RGB")
    w = src.width
    # (title band, image band) in source pixels, found by scanning for white separators
    rows = [(203, 382), (582, 761), (962, 1141)]  # random #2 idx 28, #4 idx 121, #6 idx 213
    head = Image.new("RGB", (w, 30), "white")
    d = ImageDraw.Draw(head)
    f = ImageFont.truetype(FONT, 17)
    for x, t in [(16, "input RGB"), (561, "predicted disparity (Lite-Mono)"),
                 (1106, "AbsRel at LiDAR points (blue 0 - red 0.5)")]:
        d.text((x, 6), t, fill="black", font=f)
    parts = [head] + [src.crop((0, a, w, b + 8)) for a, b in rows]
    vstack(parts).save(os.path.join(OUT, "fig1_qualitative.png"))


def fig4():
    both = Image.open(os.path.join(DP2, "sheet_both_h.png")).convert("RGB")
    only = Image.open(os.path.join(DP2, "sheet_only_lm_h.png")).convert("RGB")
    w = both.width
    header = both.crop((0, 28, w, 54))
    # (title top, image bottom) per row; row titles sit ~18 px above each image band
    both_rows = [(1015, 1114), (460, 559)]   # idx 364, idx 549
    only_rows = [(321, 420), (460, 559)]     # idx 168, idx 675
    parts = [header,
             label_strip("(a) hard for both models (scene or LiDAR GT problem)", w)]
    parts += [both.crop((0, a, w, b)) for a, b in both_rows]
    parts += [label_strip("(b) Lite-Mono-specific failures (Depth Pro 2-H+ is fine)", w)]
    parts += [only.crop((0, a, w, b)) for a, b in only_rows]
    vstack(parts).save(os.path.join(OUT, "fig4_depthpro2.png"))


def main():
    os.makedirs(OUT, exist_ok=True)
    fig1()
    shutil.copy(os.path.join(A29, "boundary_overlay/overlay_a4.png"), os.path.join(OUT, "fig2_overlay.png"))
    shutil.copy(os.path.join(A29, "boundary_zoom/edge/idx529_zoom.png"), os.path.join(OUT, "fig3_idx529.png"))
    fig4()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
