#!/bin/bash
# GR00T-N1.5 zero-shot evaluation on the RoboCasa365 Composite-Unseen tasks (Table 4).
#SBATCH --job-name=eval-gr00t-composite-unseen
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT=${PROJECT_ROOT}
EVAL_PY=$PROJECT_ROOT/structrl/groot_official_eval.py
N_EPISODES=${N_EPISODES:-100}
NAME=${NAME:?set NAME}
CKPT=${CKPT:?set CKPT}

OUT_BASE=$PROJECT_ROOT/eval_out/eval_${NAME}_step100
LOG_DIR=$PROJECT_ROOT/logs/eval_${NAME}_step100
mkdir -p "$OUT_BASE" "$LOG_DIR"

# 16 composite_UNSEEN tasks WITH official per-task horizons (robocasa dataset_registry).
TASKS=(
  ArrangeBreadBasket:2900 ArrangeTea:1500 BreadSelection:1300 CategorizeCondiments:1100
  CuttingToolSelection:800 GarnishPancake:1800 GatherTableware:1500 HeatKebabSandwich:1800 MakeIceLemonade:2000
  PanTransfer:1200 PortionHotDogs:1500 RecycleBottlesByType:1900 SeparateFreezerRack:1600
  WaffleReheat:2700 WashFruitColander:2100 WeighIngredients:2000
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
  if [ $IDX -ge ${#TASKS[@]} ]; then echo "[node $NID gpu $GPU] no task (idx $IDX)"; exit 0; fi
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

echo "===== EVAL $NAME done $(date) ====="
merged=$OUT_BASE/_summary_${NAME}.csv
awk 'FNR==1 && NR!=1 {next} {print}' $OUT_BASE/_part_${NAME}_*.csv > "$merged" 2>/dev/null
echo "--- $NAME ---"; cat "$merged" 2>/dev/null
awk -F, 'NF>=4 && $4!="rate"{s+=$4;c++} END{if(c>0)printf "%-20s macro_SR=%.4f over %d tasks\n","'"$NAME"'",s/c,c}' "$merged" 2>/dev/null
