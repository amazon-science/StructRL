#!/bin/bash
# pi0.5 evaluation on RoboCasa365 Composite-Seen: one policy server per GPU + simulator client,
# 16 tasks x 100 episodes, target split.
#SBATCH --job-name=eval-pi05-composite-seen
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -x
: "${MODEL:?set MODEL=pi0|pi05}"
REPLAN="${REPLAN:-50}"
: "${CKPT_DIR:?set CKPT_DIR}"
: "${CONFIG:?set CONFIG}"
NUM_TRIALS="${NUM_TRIALS:-100}"
DENOISE="${DENOISE:-}"   # optional: denoising steps (empty = model default 10)

REPO=${OPENPI_ROOT}
# fixed 16 composite_seen tasks, index 0..15 == SLURM_PROCID
TASKS=(DeliverStraw GetToastedBread KettleBoiling LoadDishwasher \
       PackIdenticalLunches PreSoakPan PrepareCoffee RinseSinkBasin \
       ScrubCuttingBoard SearingMeat SetUpCuttingStation StackBowlsCabinet \
       SteamInMicrowave StirVegetables StoreLeftoversInBowl WashLettuce)

# Job id in the dir name: two same-second submits of the same MODEL/REPLAN
# otherwise collide into one LOG_DIR and their 16+16 servers/stats interleave.
TS=$(date +%Y%m%d-%H%M%S)
LOG_DIR=$REPO/eval_out/${MODEL}_r${REPLAN}${DENOISE:+_d$DENOISE}_${TS}_j${SLURM_JOB_ID}
mkdir -p "$LOG_DIR"
echo "=== $MODEL replan=$REPLAN denoise=${DENOISE:-default10} | ckpt=$CKPT_DIR | nodes=$SLURM_JOB_NODELIST | out=$LOG_DIR ==="

# srun launches 16 tasks (8/node). Each pins to GPU = SLURM_LOCALID (0..7) and
# task = TASKS[SLURM_PROCID] (0..15). Pass the task list via env (CSV) and index
# inside the worker — bash arrays don't survive the srun boundary.
export TASKS_CSV=$(IFS=,; echo "${TASKS[*]}")
srun --ntasks-per-node=8 --gres=gpu:8 bash -c "
  export GPU=\$SLURM_LOCALID
  IFS=',' read -ra _T <<< \"\$TASKS_CSV\"
  export TASK=\${_T[\$SLURM_PROCID]}
  export MODEL='$MODEL' REPLAN='$REPLAN' CKPT_DIR='$CKPT_DIR' CONFIG='$CONFIG'
  export LOG_DIR='$LOG_DIR' NUM_TRIALS='$NUM_TRIALS' DENOISE='$DENOISE'
  bash $REPO/examples/robocasa/eval_pi05_shard.sh
"
echo "EVAL EXIT: $? | results under $LOG_DIR/evals_1.5/target/"
