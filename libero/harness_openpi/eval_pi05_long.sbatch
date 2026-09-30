#!/bin/bash
# LIBERO-Long (libero_10) offline eval for ONE RLinf openpi ckpt at ONE
# action_chunk, ONE task per GPU. 2 nodes x [8 + 2] GPUs = 10 tasks (1:1).
# N_TRIALS=50 -> 500 episodes (10 task x 50 init). Mirrors the GR00T harness
# eval_sft_d4_openloop.sbatch layout; worker is single-process openpi instead
# of GR00T server-client.
#
# Submit with (MODEL, ACTION_CHUNK, CONFIG_NAME, CKPT passed via --export):
#   sbatch --export=ALL,MODEL=pi0,ACTION_CHUNK=5,CONFIG_NAME=pi0_libero,CKPT=...,NUM_STEPS=10 \
#          --nodelist=<2 nodes> eval_pi05_long.sbatch
#SBATCH --job-name=eval-pi05-long
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repository root>}"
: "${MODEL:?set MODEL=pi0|pi05}"
: "${ACTION_CHUNK:?set ACTION_CHUNK=5|50|10}"
: "${CONFIG_NAME:?set CONFIG_NAME=pi0_libero|pi05_libero}"
: "${CKPT:?set CKPT=/abs/path/to/ckpt_dir}"
NUM_STEPS="${NUM_STEPS:-10}"
N_TRIALS="${N_TRIALS:-50}"
MAX_STEPS="${MAX_STEPS:-}"

HARNESS=${PROJECT_ROOT}/libero/harness_openpi
mkdir -p "${HARNESS}/logs"
export JOBID=${SLURM_JOB_ID}
export EXP_NAME="${MODEL}_c${ACTION_CHUNK}_ns${NUM_STEPS}"
export LOG_DIR="${HARNESS}/eval_out/${EXP_NAME}_${JOBID}"
mkdir -p "${LOG_DIR}"

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "===== ${EXP_NAME} ${JOBID}: A=${NODE_A} B=${NODE_B} CKPT=${CKPT} CHUNK=${ACTION_CHUNK} NUM_STEPS=${NUM_STEPS} at $(date) ====="

# 8 tasks on node A (GPU=TID), 2 tasks on node B (GPU=TID-8) — same split as GR00T harness.
for TID in 0 1 2 3 4 5 6 7; do
  TASK_ID=$TID GPU=$TID N_TRIALS="${N_TRIALS}" CONFIG_NAME="${CONFIG_NAME}" CKPT="${CKPT}" \
    ACTION_CHUNK="${ACTION_CHUNK}" NUM_STEPS="${NUM_STEPS}" MAX_STEPS="${MAX_STEPS}" EXP_NAME="${EXP_NAME}" LOG_DIR="${LOG_DIR}" \
    srun --exclusive -N1 -n1 --gres=gpu:1 -w "${NODE_A}" \
      bash "${HARNESS}/worker_one_task_openpi.sh" \
      > "${LOG_DIR}/task-${JOBID}_${TID}.out" 2>&1 &
done
for TID in 8 9; do
  GPU=$((TID-8))
  TASK_ID=$TID GPU=$GPU N_TRIALS="${N_TRIALS}" CONFIG_NAME="${CONFIG_NAME}" CKPT="${CKPT}" \
    ACTION_CHUNK="${ACTION_CHUNK}" NUM_STEPS="${NUM_STEPS}" MAX_STEPS="${MAX_STEPS}" EXP_NAME="${EXP_NAME}" LOG_DIR="${LOG_DIR}" \
    srun --exclusive -N1 -n1 --gres=gpu:1 -w "${NODE_B}" \
      bash "${HARNESS}/worker_one_task_openpi.sh" \
      > "${LOG_DIR}/task-${JOBID}_${TID}.out" 2>&1 &
done

wait
echo "===== ALL 10 TASKS DONE at $(date) | results under ${LOG_DIR} ====="
# aggregate
${HOME}/projects/openpi-robocasa/.venv/bin/python "${HARNESS}/aggregate.py" "${LOG_DIR}/${EXP_NAME}" || true
