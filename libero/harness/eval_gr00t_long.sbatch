#!/bin/bash
# Universal GR00T LIBERO eval @ MAX_STEPS_OVERRIDE=1024 (RL-training parity).
# Pass CKPT via --export. Copied from eval_universal_520.sbatch, only the
# horizon differs (1024 instead of 520). denoising=4, OPEN-LOOP, 2 nodes,
# N_TRIALS=50 -> 500 episodes.
#SBATCH --job-name=eval-gr00t-long
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repository root>}"
HARNESS=${PROJECT_ROOT}/libero/harness
mkdir -p "${HARNESS}/logs"
export JOBID=${SLURM_JOB_ID}
export N_TRIALS=50
export MAX_STEPS_OVERRIDE=1024
# robosuite-internal horizon must exceed the outer 1024 loop (default 1000
# would raise on the step after 1000). Same fix as harness_openpi horizon=2000.
export LIBERO_ENV_HORIZON=2000
: "${CKPT:?set CKPT via --export}"
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "===== lbev-g1024 ${JOBID}: A=${NODE_A} B=${NODE_B} CKPT=${CKPT} DENOISING=4 OPEN-LOOP MAX_STEPS=1024 at $(date) ====="
for TID in 0 1 2 3 4 5 6 7; do
  TASK_ID=$TID GPU=$TID CKPT="${CKPT}" MAX_STEPS_OVERRIDE=1024 srun --overlap -N1 -n1 -w "${NODE_A}" \
      bash "${HARNESS}/worker_one_task.sh" \
      > "${HARNESS}/logs/task-${JOBID}_${TID}.out" 2>&1 &
done
for TID in 8 9; do
  GPU=$((TID-8))
  TASK_ID=$TID GPU=$GPU CKPT="${CKPT}" MAX_STEPS_OVERRIDE=1024 srun --overlap -N1 -n1 -w "${NODE_B}" \
      bash "${HARNESS}/worker_one_task.sh" \
      > "${HARNESS}/logs/task-${JOBID}_${TID}.out" 2>&1 &
done
wait
echo "===== ALL 10 TASKS DONE at $(date) ====="
