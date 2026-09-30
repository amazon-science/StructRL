"""Convert an RLinf openpi pi05 full_weights.pt into a serve_policy-loadable
PyTorch checkpoint dir (single model.safetensors, 812 keys):
  - drop the 8 PPO value_head.* keys (eval-unused)
  - drop paligemma...language_model.embed_tokens.weight (tied in the base
    layout; base safetensors does not carry it)
  - copy config.json + robocasa365/ + assets/ norm_stats from the base SFT
    pytorch ckpt (norm stats are training-invariant).
Usage: python convert_pi05_fullweights_to_pytorch.py <full_weights.pt> <out_dir>
"""
import json
import os
import shutil
import struct
import sys

import torch
from safetensors.torch import save_file

FW = sys.argv[1]
OUT = sys.argv[2]
# SFT base ckpt providing config/norm_stats and the reference weight layout.
# Override with PI05_SFT_BASE; default assumes the openpi checkpoints root.
BASE = os.environ.get(
    "PI05_SFT_BASE",
    os.path.join(
        os.environ.get("OPENPI_CKPT_ROOT", "checkpoints"),
        "pi05_robocasa_finetune_target_composite_seen/cs16_2n_20260725/99999_pytorch",
    ),
)

with open(os.path.join(BASE, "model.safetensors"), "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
base_keys = {k for k in hdr if k != "__metadata__"}

sd = torch.load(FW, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()

drop = sorted(k for k in sd if k not in base_keys)
bad = [k for k in drop if ".value_head." not in k and "value_head." not in k
       and "embed_tokens" not in k]
assert not bad, f"unexpected extra keys: {bad}"
missing = sorted(base_keys - set(sd))
assert not missing, f"missing keys vs base: {missing[:10]}"
print(f"dropping {len(drop)} keys ({[k.split('.')[0] for k in drop[:3]]}...); "
      f"keeping {len(base_keys)}")

os.makedirs(OUT, exist_ok=True)
tensors = {k: sd[k].contiguous() for k in base_keys}
save_file(tensors, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
shutil.copy(os.path.join(BASE, "config.json"), os.path.join(OUT, "config.json"))
for sub in ["robocasa365", "assets"]:
    src = os.path.join(BASE, sub)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(OUT, sub), dirs_exist_ok=True)
# serve_policy also accepts assets/ produced at train time; rl2eval layout has
# both robocasa365/ and assets/ carrying norm_stats.json — mirror it.
if not os.path.isdir(os.path.join(OUT, "assets")):
    os.makedirs(os.path.join(OUT, "assets"), exist_ok=True)
    shutil.copy(os.path.join(BASE, "robocasa365", "norm_stats.json"),
                os.path.join(OUT, "assets", "norm_stats.json"))
print(f"done -> {OUT}")
