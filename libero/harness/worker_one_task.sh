#!/bin/bash
# D4 VARIANT of worker_one_task.sh — ONLY change vs original: server runs
# --denoising_steps 4 (matches RL training), instead of 8. Everything else
# identical. One LIBERO task on one GPU: GR00T N1.5 server (.venv-groot) +
# clean-LIBERO client (.venv-libero), both pinned to the single visible GPU,
# then N_TRIALS episodes of one task_id. Invoked via srun job steps.
# Expects env: TASK_ID, GPU, N_TRIALS, JOBID
set -uo pipefail

GROOT_ROOT=${PROJECT_ROOT}/Isaac-GR00T
GROOT_PY=${PROJECT_ROOT}/.venv-groot/bin/python
LIBERO_PY=${LIBERO_PY:-${PROJECT_ROOT}/.venv-libero/bin/python}
CKPT=${CKPT:-${CKPT_ROOT}/gr00t-n1.5-libero-long}
DATA_CONFIG=examples.Libero.custom_data_config:LiberoDataConfig
SUITE=libero_10
PORT=$((5900 + TASK_ID))

HARNESS=${PROJECT_ROOT}/libero/harness
SHIM=${HARNESS}/_client_shim
LIBERO_CFG=${LIBERO_CONFIG_PATH:-${HOME}/.libero}
LOGDIR=${HARNESS}/logs

export CUDA_VISIBLE_DEVICES=${GPU}
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[task ${TASK_ID}] host=$(hostname) GPU=${GPU} PORT=${PORT} DENOISING=4 at $(date)"

# ---- server on this GPU (.venv-groot) ----
cd "${GROOT_ROOT}"
"${GROOT_PY}" -u "${GROOT_ROOT}/scripts/inference_service.py" \
    --server --model_path "${CKPT}" \
    --data_config "${DATA_CONFIG}" \
    --embodiment_tag new_embodiment --denoising_steps 4 --port "${PORT}" \
    > "${LOGDIR}/server-${JOBID}_${TASK_ID}.log" 2>&1 &
SPID=$!

for i in $(seq 1 200); do
  (exec 3<>/dev/tcp/127.0.0.1/${PORT}) 2>/dev/null && { exec 3>&- 3<&-; echo "[task ${TASK_ID}] server up"; break; }
  kill -0 $SPID 2>/dev/null || { echo "[task ${TASK_ID}][err] server died:"; tail -30 "${LOGDIR}/server-${JOBID}_${TASK_ID}.log"; exit 1; }
  sleep 3
done

# ---- client: official LIBERO (.venv-libero), single task, isolated cwd ----
RUNDIR=/tmp/libero_task_${JOBID}_${TASK_ID}
mkdir -p "${RUNDIR}"; cd "${RUNDIR}"
echo "[task ${TASK_ID}] rollout start $(date)"
LIBERO_CONFIG_PATH="${LIBERO_CFG}" PYTHONPATH="${SHIM}:${HARNESS}" "${LIBERO_PY}" -u \
    "${HARNESS}/run_libero_eval_single.py" \
    --task_suite_name "${SUITE}" \
    --task_id "${TASK_ID}" \
    --num_trials_per_task "${N_TRIALS}" \
    --port "${PORT}" \
    --headless

echo "[task ${TASK_ID}] killing server"
kill "$SPID" 2>/dev/null || true; sleep 2; kill -9 "$SPID" 2>/dev/null || true
echo "[task ${TASK_ID}] DONE at $(date)"
