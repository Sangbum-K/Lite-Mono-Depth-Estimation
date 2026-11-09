"""python analysis/test_eval_per_image.py  (numpy only; the GPU path is checked against evaluate_depth.py on Colab)"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_per_image import errors_one, pack, unpack

rng = np.random.default_rng(0)
maps = []
for h, w in [(375, 1242), (370, 1224)]:  # KITTI sizes differ per date
    m = np.zeros((h, w), np.float32)
    pts = rng.choice(h * w, 3000, replace=False)
    m.flat[pts] = rng.uniform(1, 90, 3000)  # some beyond the 80 m cap
    maps.append(m)

g = pack(maps)
for i, m in enumerate(maps):
    assert np.array_equal(unpack(g, i), m), "sparse round trip changed GT"

# a prediction that is the GT scaled by 2.5 must score perfectly after median scaling
gt = maps[0].copy()
gt[gt == 0] = 10.0  # dense GT so the identity resize below is exact
abs_rel = lambda gt, pred: (np.mean(np.abs(gt - pred) / gt),) * 7
out = errors_one(gt, 1 / (2.5 * gt), abs_rel, lambda x, size: x)
assert out[0] < 1e-6 and abs(out[7] - 1 / 2.5) < 1e-6, out
h, w = gt.shape
crop_px = (int(0.99189189 * h) - int(0.40810811 * h)) * (int(0.96405229 * w) - int(0.03594771 * w))
assert out[8] <= crop_px and out[8] == ((gt > 1e-3) & (gt < 80))[int(0.40810811 * h):int(0.99189189 * h),
                                                                    int(0.03594771 * w):int(0.96405229 * w)].sum()
print("ok")
