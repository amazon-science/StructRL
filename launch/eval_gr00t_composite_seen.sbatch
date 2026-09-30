#!/bin/bash
# GR00T-N1.5 evaluation on RoboCasa365 Composite-Seen: 16 tasks x 100 episodes, target split.
# Usage: sbatch --export=ALL,NAME=<tag>,CKPT=<hf checkpoint dir> eval_gr00t_composite_seen.sbatch
#SBATCH --job-name=eval-gr00t-composite-seen
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repo root>}"
EVAL_PY=$PROJECT_ROOT/structrl/groot_official_eval.py
N_EPISODES=${N_EPISODES:-100}

: "${NAME:?must set NAME}"
: "${CKPT:?must set CKPT}"

OUT_BASE=$PROJECT_ROOT/eval_out/eval_${NAME}
LOG_DIR=$PROJECT_ROOT/logs/eval_${NAME}
mkdir -p "$OUT_BASE" "$LOG_DIR"

TASKS=(
  DeliverStraw:1700 GetToastedBread:2000 KettleBoiling:1000 LoadDishwasher:1200
  PackIdenticalLunches:2600 PreSoakPan:1600 PrepareCoffee:1200 RinseSinkBasin:900
  ScrubCuttingBoard:800 SearingMeat:2900 SetUpCuttingStation:1600 StackBowlsCabinet:1400
  SteamInMicrowave:1400 StirVegetables:1600 StoreLeftoversInBowl:1700 WashLettuce:1100
)

echo "===== EVAL $NAME (2 node, 1 task/GPU) start $(date) ====="
echo "nodes: $SLURM_JOB_NODELIST"
echo "ckpt: $CKPT"
echo "n_episodes/task: $N_EPISODES  (eval_split=target, n_action_steps=16)"

srun --kill-on-bad-exit=0 bash -c '
  source '"$PROJECT_ROOT"'/.venv-groot/bin/activate
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  NID=$SLURM_NODEID
  GPU=$SLURM_LOCALID
  export CUDA_VISIBLE_DEVICES=$GPU
  TASKS=('"${TASKS[*]}"')
  IDX=$(( NID*8 + GPU ))
  TASK_LIST=${TASKS[$IDX]}
  SUMMARY='"$OUT_BASE"'/_part_'"$NAME"'_n${NID}g${GPU}.csv
  VIDEO='"$OUT_BASE"'/'"$NAME"'
  LOG='"$LOG_DIR"'/'"$NAME"'__n${NID}g${GPU}.log
  echo "[node $NID gpu $GPU] $(hostname) task=$TASK_LIST"
  cd /tmp
  python -u '"$EVAL_PY"' \
    --task_list "$TASK_LIST" \
    --n_episodes '"$N_EPISODES"' \
    --eval_split target \
    --n_action_steps 16 \
    --video_dir "$VIDEO" \
    --summary_csv "$SUMMARY" \
    --ckpt_path '"$CKPT"' \
    > "$LOG" 2>&1
  echo "[node $NID gpu $GPU] done rc=$?"
'

echo "===== EVAL done $(date) ====="
merged=$OUT_BASE/_summary_${NAME}.csv
awk 'FNR==1 && NR!=1 {next} {print}' $OUT_BASE/_part_${NAME}_*.csv > "$merged" 2>/dev/null
echo "--- $NAME ---"; cat "$merged" 2>/dev/null
echo "=== macro SR (mean of per-task rate, 4th col) ==="
awk -F, 'NF>=4 && $4!="rate"{s+=$4;c++} END{if(c>0)printf "%-18s macro_SR=%.4f over %d tasks\n","'"$NAME"'",s/c,c}' "$merged" 2>/dev/null
