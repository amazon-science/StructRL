#!/bin/bash
# One LIBERO-Long task on one GPU for an RLinf openpi (pi0/pi05) ckpt.
# Single-process: openpi PyTorch policy is loaded IN-PROCESS (no server-client),
# then N_TRIALS episodes of one task_id run against it. Invoked via srun steps
# from the 2-node sbatch, one per (task_id -> gpu).
#
# Expects env: TASK_ID, GPU, N_TRIALS, JOBID, CONFIG_NAME, CKPT, ACTION_CHUNK,
#              NUM_STEPS, EXP_NAME, LOG_DIR
set -uo pipefail

# --- isolated bridge env (READ-ONLY path composition, no venv is modified) ---
OPENPI_PY=${HOME}/projects/openpi-robocasa/.venv/bin/python   # openpi torch-port + jax/flax + torch + ray + cuda
RLINF=${LIBERO_RL_ROOT}/RLinf                        # provides rlinf.* + toolkits.eval_scripts_openpi
LIBERO_SRC=${LIBERO_RL_ROOT}/LIBERO                  # editable libero package root
LIBERO_SP=${PROJECT_ROOT}/.venv-rlinf-libero/lib/python3.11/site-packages  # mujoco/robosuite/libero deps
HARNESS=${PROJECT_ROOT}/libero/harness_openpi
LIBERO_CFG=${LIBERO_CONFIG_PATH:-${HOME}/.libero}  # same LIBERO config the GR00T harness uses
TF_PATCHED=${TF_PATCHED:?export TF_PATCHED=<transformers copy with openpi transformers_replace applied>}

export CUDA_VISIBLE_DEVICES=${GPU}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# openpi src is on the venv already (editable); RLinf + libero + sim deps via PYTHONPATH.
# _tf_patched FIRST so the patched transformers shadows the unpatched one in the source venv.
export PYTHONPATH="${TF_PATCHED}:${RLINF}:${LIBERO_SRC}:${LIBERO_SP}"
export LIBERO_CONFIG_PATH="${LIBERO_CFG}"

echo "[t${TASK_ID}] host=$(hostname) GPU=${GPU} CONFIG=${CONFIG_NAME} CHUNK=${ACTION_CHUNK} NUM_STEPS=${NUM_STEPS} at $(date)"

# Run from the scratch assets cwd: this openpi fork eagerly resolves norm_stats
# from CWD-relative "checkpoints/torch/pi0_libero/assets/<asset_id>/" (value is
# discarded — the policy loads real stats from the ckpt — but the path must exist
# or data.create() crashes). Both pi0_libero & pi05_libero use the pi0_libero
# assets dir, so one scratch tree serves both.
ASSETS_CWD="${HARNESS}/_assets_cwd"
cd "${ASSETS_CWD}"
"${OPENPI_PY}" -u "${HARNESS}/run_openpi_libero_single.py" \
    --config_name "${CONFIG_NAME}" \
    --pretrained_path "${CKPT}" \
    --task_suite_name libero_10 \
    --task_id "${TASK_ID}" \
    --num_trials_per_task "${N_TRIALS}" \
    --action_chunk "${ACTION_CHUNK}" \
    --num_steps "${NUM_STEPS}" \
    ${MAX_STEPS:+--max_steps ${MAX_STEPS}} \
    --exp_name "${EXP_NAME}" \
    --log_dir "${LOG_DIR}" \
    --num_save_videos 5

echo "[t${TASK_ID}] DONE rc=$? at $(date)"
