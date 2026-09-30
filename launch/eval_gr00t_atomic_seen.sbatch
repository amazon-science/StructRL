#!/bin/bash
# GR00T-N1.5 zero-shot evaluation on the 65 RoboCasa365 Atomic-Seen tasks (Table 4).
#SBATCH --job-name=eval-gr00t-atomic-seen
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
: "${NAME:?set NAME}"
: "${CKPT:?set CKPT (abs path)}"

OUT_BASE=$PROJECT_ROOT/eval_out/eval_${NAME}
LOG_DIR=$PROJECT_ROOT/logs/eval_${NAME}
mkdir -p "$OUT_BASE" "$LOG_DIR"

# 65 atomic tasks WITH official per-task horizons (robocasa ATOMIC_TASK_DATASETS).
TASKS=(
  AdjustToasterOvenTemperature:300 AdjustWaterTemperature:300 CloseBlenderLid:600
  CloseCabinet:500 CloseDishwasher:200 CloseDrawer:300 CloseElectricKettleLid:200
  CloseFridge:600 CloseFridgeDrawer:300 CloseMicrowave:300 CloseOven:300
  CloseStandMixerHead:300 CloseToasterOvenDoor:300 CoffeeServeMug:300 CoffeeSetupMug:400
  NavigateKitchen:300 OpenBlenderLid:300 OpenCabinet:700 OpenDishwasher:300 OpenDrawer:500
  OpenElectricKettleLid:200 OpenFridge:600 OpenFridgeDrawer:300 OpenMicrowave:400 OpenOven:200
  OpenStandMixerHead:300 OpenToasterOvenDoor:300 PickPlaceCabinetToCounter:300
  PickPlaceCounterToBlender:500 PickPlaceCounterToCabinet:500 PickPlaceCounterToDrawer:500
  PickPlaceCounterToMicrowave:700 PickPlaceCounterToOven:400 PickPlaceCounterToSink:400
  PickPlaceCounterToStandMixer:400 PickPlaceCounterToStove:400 PickPlaceCounterToToasterOven:300
  PickPlaceDrawerToCounter:500 PickPlaceFridgeDrawerToShelf:400 PickPlaceFridgeShelfToDrawer:400
  PickPlaceMicrowaveToCounter:500 PickPlaceSinkToCounter:600 PickPlaceStoveToCounter:300
  PickPlaceToasterOvenToCounter:300 PickPlaceToasterToCounter:400 PreheatOven:300
  SlideDishwasherRack:300 SlideOvenRack:300 SlideToasterOvenRack:200 StartCoffeeMachine:200
  TurnOffMicrowave:200 TurnOffSinkFaucet:200 TurnOffStove:500 TurnOnBlender:200
  TurnOnElectricKettle:300 TurnOnMicrowave:300 TurnOnSinkFaucet:400 TurnOnStove:300
  TurnOnToaster:200 TurnOnToasterOven:300 TurnSinkSpout:200 MakeIcedCoffee:500
  CheesyBread:500 LowerHeat:500 PackDessert:400
)

echo "===== EVAL $NAME (2 node, 65 atomic tasks round-robin) start $(date) ====="
echo "nodes: $SLURM_JOB_NODELIST ; n_tasks=${#TASKS[@]}"
echo "ckpt: $CKPT"
echo "n_episodes/task: $N_EPISODES  (eval_split=target, n_action_steps=16)"

srun --kill-on-bad-exit=0 bash -c '
  source '"$PROJECT_ROOT"'/.venv-groot/bin/activate
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
  NID=$SLURM_NODEID
  GPU=$SLURM_LOCALID
  export CUDA_VISIBLE_DEVICES=$GPU
  TASKS=('"${TASKS[*]}"')
  IDX=$(( NID*8 + GPU ))          # 0..15
  NGPU=16
  # round-robin: this GPU takes tasks IDX, IDX+16, IDX+32, ...
  MY=()
  for ((j=IDX; j<${#TASKS[@]}; j+=NGPU)); do MY+=("${TASKS[$j]}"); done
  if [ ${#MY[@]} -eq 0 ]; then echo "[node $NID gpu $GPU] no task (idx $IDX)"; exit 0; fi
  TASK_LIST=$(IFS=,; echo "${MY[*]}")
  SUMMARY='"$OUT_BASE"'/_part_'"$NAME"'_n${NID}g${GPU}.csv
  VIDEO='"$OUT_BASE"'/'"$NAME"'
  LOG='"$LOG_DIR"'/'"$NAME"'__n${NID}g${GPU}.log
  echo "[node $NID gpu $GPU] $(hostname) n_my=${#MY[@]} tasks=$TASK_LIST"
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
