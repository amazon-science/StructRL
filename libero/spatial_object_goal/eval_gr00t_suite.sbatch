#!/bin/bash
# OFFICIAL-protocol eval of a GR00T N1.5 ckpt on one LIBERO suite
# (spatial/object/goal): 10 tasks x 50 fixed inits = exactly 500 episodes,
# official per-suite horizons baked into run_libero_eval_single.py
# (spatial 220 / object 280 / goal 600), OPEN-LOOP full-chunk execution
# (the fixed harness — same lineage as the libero_10 479/500=95.8% run).
#
# Fans out 10 single-GPU srun steps (task 0-7 -> node A, 8-9 -> node B), each
# = GR00T server (.venv-groot) + official-LIBERO client (.venv-libero).
# Everything referenced from libero-eval/ is READ-ONLY; new files live in
# libero3-rl/. Submit with: SUITE=libero_spatial CKPT=<hf ckpt dir> sbatch ...
#SBATCH --job-name=eval-gr00t-suite
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=2
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repository root>}"
: "${SUITE:?must set SUITE=libero_spatial|libero_object|libero_goal}"
: "${CKPT:?must set CKPT=<path to HF ckpt dir>}"

LB3H=${PROJECT_ROOT}/libero/spatial_object_goal
mkdir -p "${LB3H}/logs"
export JOBID=${SLURM_JOB_ID}
export N_TRIALS=50
export SUITE CKPT

NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
NODE_A=${NODES[0]}; NODE_B=${NODES[1]}
echo "===== lb3-off ${JOBID}: suite=${SUITE} ckpt=${CKPT} A=${NODE_A} B=${NODE_B} at $(date) ====="

# task 0-7 on node A (GPU 0-7), task 8-9 on node B (GPU 0-1)
for TID in 0 1 2 3 4 5 6 7; do
  TASK_ID=$TID GPU=$TID srun --exclusive -N1 -n1 --gres=gpu:1 -w "${NODE_A}" \
      bash "${LB3H}/worker_one_task_suite.sh" \
      > "${LB3H}/logs/task-${SUITE}-${JOBID}_${TID}.out" 2>&1 &
done
for TID in 8 9; do
  GPU=$((TID-8))
  TASK_ID=$TID GPU=$GPU srun --exclusive -N1 -n1 --gres=gpu:1 -w "${NODE_B}" \
      bash "${LB3H}/worker_one_task_suite.sh" \
      > "${LB3H}/logs/task-${SUITE}-${JOBID}_${TID}.out" 2>&1 &
done

wait
echo "===== ALL 10 TASKS DONE (${SUITE}) at $(date) ====="
