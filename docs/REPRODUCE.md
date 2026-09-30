# Reproducing the paper

All launchers read `PROJECT_ROOT` (the repository root, see [INSTALL.md](INSTALL.md)); the LIBERO ones also
read `LIBERO_RL_ROOT` and `CKPT_ROOT`. Submit them from the repository root. Training uses 4 nodes x 8
GPUs and evaluation 2 nodes x 8 GPUs (one task per GPU). Environment resets are not seeded, so repeated
evaluations of one checkpoint vary by about 1 to 2 points of macro SR.

Table and figure numbers refer to the arXiv version of the paper.

## RoboCasa365 (16 Composite-Seen tasks)

### Training (`launch/`)

| Paper entry | Launcher | Reward and optimizer |
|---|---|---|
| Table 1, GR00T-N1.5 w/ StructRL | `train_gr00t_structrl_ppo.sbatch` | StructRL reward, PPO |
| Table 1, GR00T-N1.5 w/ Sparse-RL | `train_gr00t_sparse_rl_ppo.sbatch` | terminal binary reward, PPO |
| Table 1, GR00T-N1.5 w/ SimpleVLA-RL | `train_gr00t_simplevla_rl_grpo.sbatch` | binary reward, GRPO (group 8) |
| Table 3, GR00T-N1.5 w/ StructRL (GRPO) | `train_gr00t_structrl_grpo.sbatch` | summed StructRL reward, GRPO |
| Table 1, π0.5 w/ StructRL | `train_pi05_structrl_ppo.sbatch` | StructRL reward, PPO |
| Table 1, π0.5 w/ Sparse-RL | `train_pi05_sparse_rl_ppo.sbatch` | binary reward, PPO |
| Table 1, π0.5 w/ SimpleVLA-RL | `train_pi05_simplevla_rl_grpo.sbatch` | binary reward, GRPO |
| Table 3, π0.5 w/ StructRL (GRPO) | `train_pi05_structrl_grpo.sbatch` | summed StructRL reward, GRPO |
| Fig. 3 (a-d), binary reward | `train_gr00t_sparse_rl_ppo.sbatch` | as Sparse-RL |
| Fig. 3 (a-d), + subtask reward | `../ablations/ungated/train_gr00t_ungated_fixed_ppo.sbatch` | fixed subtask reward, no gating |
| Fig. 3 (a-d), + dynamic pacing | `../ablations/ungated/train_gr00t_ungated_paced_ppo.sbatch` | paced subtask reward, no gating |
| Fig. 3 (a-d), + gating (StructRL) | `train_gr00t_structrl_ppo.sbatch` | full StructRL |
| Fig. 8, beta sensitivity | `../ablations/beta_sweep/train_gr00t_structrl_beta{0.2,1.0,2.0}.sbatch` | beta = 0.2 / 1.0 / 2.0 |
| Fig. 4, Tables 8 to 12 | `../density/train_density.sbatch` | see `density/README.md` |
| π0.5 SFT policy | `sft_pi05.sh` | openpi config `pi05_robocasa_finetune_target_composite_seen` |

`train_gr00t_fixed_subtask_ppo.sbatch` (gated fixed subtask reward) is included for completeness; no
table uses it. The GR00T-N1.5 GRPO launchers put `grpo_diag/` on `PYTHONPATH`, which registers the
advantage type of their config. Checkpoints are written every 10 iterations to
`RLinf/logs/<time>-<RUN_TAG>/<RUN_TAG>/checkpoints/global_step_N/`.

### Checkpoint conversion (`structrl/`)

- GR00T-N1.5: `python structrl/convert_rl_fullweights_to_hf.py <run>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt <SFT checkpoint-60000> <out_dir>`
- π0.5: `PI05_SFT_BASE=<SFT dir> python structrl/convert_pi05_fullweights_to_pytorch.py <full_weights.pt> <out_dir>`

### Evaluation

- GR00T-N1.5, Composite-Seen (Tables 1 and 3, Fig. 3, Fig. 4):
  `sbatch --export=ALL,NAME=<tag>,CKPT=<converted dir> launch/eval_gr00t_composite_seen.sbatch`.
  It runs `structrl/groot_official_eval.py --eval_split target --n_action_steps 16 --n_episodes 100`
  with the official per-task horizons (800 to 2900 steps) and writes
  `eval_out/eval_<tag>/_summary_<tag>.csv`. Macro SR is the mean of the 16 per-task success rates.
  Horizon buckets: 800-1000 = {ScrubCuttingBoard, RinseSinkBasin, KettleBoiling};
  1000-1400 = {WashLettuce, LoadDishwasher, PrepareCoffee, StackBowlsCabinet, SteamInMicrowave};
  1400-2900 = the other 8 tasks.
- GR00T-N1.5, zero-shot (Table 4): `launch/eval_gr00t_composite_unseen.sbatch` (Composite-Unseen tasks)
  and `launch/eval_gr00t_atomic_seen.sbatch` (65 Atomic-Seen tasks), with the same `NAME` and `CKPT`.
- π0.5, Composite-Seen:
  `sbatch --export=ALL,MODEL=pi05,CONFIG=pi05_robocasa_finetune_target_composite_seen,CKPT_DIR=<dir> launch/eval_pi05_composite_seen.sbatch`
  (one policy server per GPU plus a simulator client; an episode counts as a success at its first
  success). `launch/eval_pi05_makeup.sbatch` re-runs missing shards.

## LIBERO (`libero/`)

LIBERO training uses a second RLinf checkout with `patches/rlinf_core.patch` and then
`patches/libero_addon.patch`, at `$LIBERO_RL_ROOT/RLinf`, and the RLinf fork of LIBERO at
`$LIBERO_RL_ROOT/LIBERO` (`scripts/setup_upstream.sh --libero`). Configs come from `libero/configs/` and
reference durations from `libero/timing/`. SFT policies are the public GR00T-N1.5 LIBERO checkpoints
and the public RLinf π0.5 LIBERO checkpoint (see [INSTALL.md](INSTALL.md)).

| Paper entry | Launcher |
|---|---|
| Table 1, GR00T-N1.5 w/ StructRL (LIBERO-Long); Fig. 3 (e-h), + gating | `libero/launch/train_gr00t_long_structrl_ppo.sbatch` |
| Table 1, GR00T-N1.5 w/ Sparse-RL; Fig. 3 (e-h), binary reward | `libero/launch/train_gr00t_long_sparse_rl_ppo.sbatch` |
| Table 1, GR00T-N1.5 w/ SimpleVLA-RL | `libero/launch/train_gr00t_long_simplevla_rl_grpo.sbatch` |
| Table 1, π0.5 w/ StructRL | `libero/launch/train_pi05_long_structrl_ppo.sbatch` (action chunk 10, 10 denoising steps, 1024 max steps) |
| Table 1, π0.5 w/ Sparse-RL | `libero/launch/train_pi05_long_sparse_rl_ppo.sbatch` |
| Table 1, π0.5 w/ SimpleVLA-RL | `libero/launch/train_pi05_long_simplevla_rl_grpo.sbatch` |
| Fig. 3 (e-h), + subtask reward / + dynamic pacing | `ablations/ungated/train_gr00t_libero_long_ungated_fixed_ppo.sbatch` / `..._paced_ppo.sbatch` |
| Table 5, StructRL on LIBERO Spatial / Object / Goal | `SUITE=spatial\|object\|goal sbatch libero/launch/train_gr00t_suite_structrl_ppo.sbatch` |
| Table 5, SimpleVLA-RL on LIBERO Spatial / Object / Goal | `SUITE=spatial\|object\|goal sbatch libero/launch/train_gr00t_suite_simplevla_rl_grpo.sbatch` |

Evaluation:

- GR00T-N1.5, LIBERO-Long: `sbatch --export=ALL,CKPT=<dir> libero/harness/eval_gr00t_long.sbatch`
  (open-loop 16-step chunks, 4 denoising steps, 50 episodes per task, 1024 max steps). Buckets:
  250-340 = tasks {1, 3, 4, 5, 7}, 340-400 = {0, 2, 6}, 400-550 = {8, 9}.
- π0.5, LIBERO-Long: `sbatch --export=ALL,MODEL=pi05,ACTION_CHUNK=10,NUM_STEPS=10,MAX_STEPS=1024,CONFIG_NAME=pi05_libero,CKPT=<dir> libero/harness_openpi/eval_pi05_long.sbatch`,
  then `python libero/harness_openpi/aggregate.py <output dir>`. The client needs `TF_PATCHED`, a copy
  of `transformers` with openpi's `transformers_replace` files applied.
- LIBERO Spatial / Object / Goal: `sbatch --export=ALL,SUITE=libero_spatial,CKPT=<dir> libero/spatial_object_goal/eval_gr00t_suite.sbatch`
  (10 tasks x 50 fixed initial states, official per-suite horizons).

The reference durations in `libero/timing/` were computed with `libero/timing/extract_libero_demo_timing.py`
(LIBERO-Long) and `libero/timing/extract_timing_libero3.py` (Spatial / Object / Goal).

## Not included

The Robometer reward-model baseline (Table 2), the PolicyTrim baseline, and the pi0 and RLDX-1
reference rows of Table 1 are not part of this release.
