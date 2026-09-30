"""Convert an RLinf FSDP full_weights.pt into a HF GR00T checkpoint dir that
groot_official_eval.py can load. Drops the 8 PPO value_head keys (eval-unused),
copies config.json + experiment_cfg + index from the base SFT ckpt, and shards
weights to match the base layout."""
import json, os, shutil, sys, torch
from safetensors.torch import save_file

FW   = sys.argv[1]   # .../global_step_N/actor/model_state_dict/full_weights.pt
BASE = sys.argv[2]   # composite_seen/checkpoint-60000 (HF safetensors)
OUT  = sys.argv[3]   # output HF dir

os.makedirs(OUT, exist_ok=True)
idx = json.load(open(f"{BASE}/model.safetensors.index.json"))
wmap = idx["weight_map"]          # key -> shard filename
shards = sorted(set(wmap.values()))

sd = torch.load(FW, map_location="cpu", weights_only=False)
if hasattr(sd, "state_dict"):
    sd = sd.state_dict()

base_keys = set(wmap.keys())
drop = sorted(k for k in sd if k not in base_keys)
assert all(".value_head." in k for k in drop), f"unexpected extra keys: {drop}"
missing = sorted(base_keys - set(sd.keys()))
assert not missing, f"missing critical keys: {missing[:10]}"
print(f"dropping {len(drop)} value_head keys; {len(base_keys)} keys -> {len(shards)} shards")

# group keys by their base shard, save each shard as safetensors
for shard in shards:
    keys = [k for k, v in wmap.items() if v == shard]
    tensors = {k: sd[k].contiguous() for k in keys}
    save_file(tensors, os.path.join(OUT, shard), metadata={"format": "pt"})
    print(f"  wrote {shard}: {len(keys)} tensors")

# copy non-weight artifacts the loader needs
for f in ["config.json", "model.safetensors.index.json"]:
    shutil.copy(os.path.join(BASE, f), os.path.join(OUT, f))
shutil.copytree(os.path.join(BASE, "experiment_cfg"),
                os.path.join(OUT, "experiment_cfg"), dirs_exist_ok=True)
print(f"done -> {OUT}")
