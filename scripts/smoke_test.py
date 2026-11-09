"""Checks that need no KITTI data. Run on the Colab GPU after installing requirements.

    python scripts/smoke_test.py --pretrained /content/drive/MyDrive/lite_mono/pretrained

1. official KITTI weights load into the vendored code with no missing keys, ~3.1M params (paper)
2. ImageNet encoder weights load the way trainer.load_pretrain() does
3. PoseNet ResNet18 ImageNet init downloads (compat fix in resnet_encoder.py)
4. the KITTI dataset pipeline runs on a synthetic drive (compat fix in mono_dataset.py)
5. one forward/backward of depth + pose nets at batch 12, 640x192 on the GPU
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "Lite-Mono"))
import datasets  # noqa: E402
import networks  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--pretrained", required=True, help="folder with lite-mono-pretrain.pth and lite-mono_640x192/")
a = p.parse_args()

enc = networks.LiteMono(model="lite-mono", height=192, width=640)
dec = networks.DepthDecoder(enc.num_ch_enc, scales=range(3))
for name, model in [("encoder", enc), ("depth", dec)]:
    sd = torch.load(os.path.join(a.pretrained, "lite-mono_640x192", name + ".pth"), map_location="cpu")
    own = model.state_dict()
    missing = [k for k in own if k not in sd]
    extra = [k for k in sd if k not in own and not k.endswith(("total_ops", "total_params"))]
    assert not missing, "{}: keys missing from official weights: {}".format(name, missing[:5])
    assert all(sd[k].shape == own[k].shape for k in own), name + ": shape mismatch"
    model.load_state_dict({k: v for k, v in sd.items() if k in own})
    print("1. {}: all {} keys loaded, ignored extra keys {}".format(name, len(own), extra))
n_params = sum(t.numel() for m in (enc, dec) for t in m.parameters())
print("   trainable params encoder+decoder = {:,} (paper: 3.1M)".format(n_params))

pre = torch.load(os.path.join(a.pretrained, "lite-mono-pretrain.pth"), map_location="cpu", weights_only=False)["model"]
fresh = networks.LiteMono(model="lite-mono", height=192, width=640).state_dict()
used = [k for k in pre if k in fresh and not k.startswith("norm")]
print("2. ImageNet pretrain: {} of {} encoder tensors initialised, not covered: {}".format(
    len(used), len(fresh), [k for k in fresh if k not in used][:5]))

pose_enc = networks.ResnetEncoder(18, True, num_input_images=2)
pose_dec = networks.PoseDecoder(pose_enc.num_ch_enc, num_input_features=1, num_frames_to_predict_for=2)
print("3. PoseNet ResNet18 ImageNet weights loaded")

with tempfile.TemporaryDirectory() as root:
    folder = "2011_09_26/2011_09_26_drive_0001_sync"
    os.makedirs(os.path.join(root, folder, "image_02", "data"))
    for f in range(3):
        Image.fromarray(np.random.randint(0, 255, (375, 1242, 3), np.uint8)).save(
            os.path.join(root, folder, "image_02", "data", "{:010d}.jpg".format(f)))
    ds = datasets.KITTIRAWDataset(root, ["{} 1 l".format(folder)], 192, 640, [0, -1, 1], 4, is_train=True)
    item = ds[0]
    print("4. dataset item:", tuple(item[("color_aug", 0, 0)].shape), "depth_gt loaded:", "depth_gt" in item)

dev = torch.device("cuda")
enc, dec, pose_enc, pose_dec = (m.to(dev).train() for m in (enc, dec, pose_enc, pose_dec))
torch.cuda.reset_peak_memory_stats()
x = torch.rand(12, 3, 192, 640, device=dev)
out = dec(enc(x))
pose = pose_dec([pose_enc(torch.cat([x, x], 1))])
loss = sum(v.mean() for v in out.values()) + pose[0].mean() + pose[1].mean()
loss.backward()
assert all(torch.isfinite(v).all() for v in out.values())
print("5. forward/backward OK, outputs", {k: tuple(v.shape) for k, v in out.items()})
print("   peak allocated {:.2f} GB (networks only; the real training step adds warping/SSIM, see train_state.json)"
      .format(torch.cuda.max_memory_allocated() / 2**30))
print("smoke test passed")
