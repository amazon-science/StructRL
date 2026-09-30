#!/bin/bash
# Re-run missing pi0.5 evaluation shards of eval_pi05_composite_seen.sbatch.
#SBATCH --job-name=eval-pi05-makeup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -x
: "${MODEL:?}"; REPLAN="${REPLAN:-50}"; : "${CKPT_DIR:?}"; : "${CONFIG:?}"
: "${LOG_DIR:?set LOG_DIR to the ORIGINAL eval_out dir}"
: "${TASKS_CSV:?task list, <=8. Use ':' separators — sbatch --export eats commas}"
TASKS_CSV="${TASKS_CSV//:/,}"
NUM_TRIALS="${NUM_TRIALS:-100}"

REPO=${OPENPI_ROOT}
IFS=',' read -ra _T <<< "$TASKS_CSV"
NT=${#_T[@]}
echo "=== MAKEUP $MODEL replan=$REPLAN | $NT tasks: $TASKS_CSV | out=$LOG_DIR ==="

export TASKS_CSV MODEL REPLAN CKPT_DIR CONFIG LOG_DIR NUM_TRIALS
srun --ntasks=$NT --ntasks-per-node=$NT --gres=gpu:8 bash -c "
  export GPU=\$SLURM_LOCALID
  IFS=',' read -ra _T <<< \"\$TASKS_CSV\"
  export TASK=\${_T[\$SLURM_PROCID]}
  bash $REPO/examples/robocasa/eval_pi05_shard.sh
"
echo "MAKEUP EXIT: $?"
