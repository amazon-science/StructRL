#!/bin/bash
# SimpleVLA-RL baseline (GRPO, pi0.5) on LIBERO-Long: action chunk 10, 10 denoising steps, 1024 max steps. Table 1.
# Same launcher as train_pi05_long_structrl_ppo.sbatch without the subtask reward.
#SBATCH --job-name=train-pi05-long-simplevla-rl-grpo
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=86:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repository root>}"
PROJECT="${LIBERO_RL_ROOT:?export LIBERO_RL_ROOT=<RLinf + LIBERO checkout for LIBERO>}"
REPO_PATH="${PROJECT}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_DIR="${PROJECT_ROOT}/libero/configs"
CONFIG_NAME="libero_long_grpo_pi05_simplevla"
LIBERO_DIR="${PROJECT}/LIBERO"
RAY_PORT=29512

VENV="${PROJECT_ROOT}/.venv-rlinf-openpi-libero"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${PROJECT}/logs"

RUN_TAG="${RUN_TAG:-train_pi05_long_simplevla_rl_grpo}"
ENV_TRAIN_NUM_ENVS="${ENV_TRAIN_NUM_ENVS:-256}"
TIMING_JSON="${LIBERO_TIMING_JSON:-${PROJECT_ROOT}/libero/timing/libero_long_demo_timing.json}"
# Smoke: MAX_STEPS=2 self-terminates. Resume: RESUME_DIR=<...>/global_step_N.
MAX_STEPS="${MAX_STEPS:-}"
MAX_STEPS_OVERRIDE=""
[ -n "${MAX_STEPS}" ] && MAX_STEPS_OVERRIDE="runner.max_steps=${MAX_STEPS}"
RESUME_DIR="${RESUME_DIR:-}"
RESUME_OVERRIDE=""
[ -n "${RESUME_DIR}" ] && RESUME_OVERRIDE="runner.resume_dir=${RESUME_DIR}"
# Optional overrides for the official-ckpt lr-sweep lines (07-31):
CKPT_PATH="${CKPT_PATH:-}"
CKPT_OVERRIDE=""
[ -n "${CKPT_PATH}" ] && CKPT_OVERRIDE="actor.model.model_path=${CKPT_PATH} rollout.model.model_path=${CKPT_PATH}"
LR="${LR:-}"
LR_OVERRIDE=""
[ -n "${LR}" ] && LR_OVERRIDE="actor.optim.lr=${LR}"

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_DIR}/${CONFIG_NAME}"
echo "venv:   ${VENV}"
echo "total_num_envs: ${ENV_TRAIN_NUM_ENVS}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export USE_TF=0
export TOKENIZERS_PARALLELISM=false
# smooth_band dense reward: env subprocess computes info["dense_reward"] via
# LiberoSmoothBandReward (0.6/(1+ratio) per latch + 2.0 outcome). MUST be set
# before ray start so env actors inherit it.
# RLinf first, then LIBERO.
export PYTHONPATH="${REPO_PATH}:${LIBERO_DIR}:${PYTHONPATH:-}"
unset RAY_ADDRESS

# NCCL multi-node — TCP/Socket (no RDMA), ens NIC (A100 node).
# Literal heredoc ('EOF'): do NOT export interpolated vars here (heredoc trap).
read -r -d '' NCCL_ENV_BLOCK <<'EOF' || true
export NCCL_SOCKET_IFNAME=ens
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
EOF
eval "${NCCL_ENV_BLOCK}"

LOG_DIR="${PROJECT}/logs/$(date +'%Y%m%d-%H%M%S')-${RUN_TAG}"
mkdir -p "${LOG_DIR}"
echo "logs -> ${LOG_DIR}"

# 1) Ray HEAD on this node.
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
MEM_BYTES=$(( 400 * 1024 * 1024 * 1024 ))
echo "[head] starting ray HEAD"
RLINF_NODE_RANK=0 "${RLINF_RAY}" start --head \
  --port="${RAY_PORT}" --node-ip-address="${HEAD_IP}" \
  --num-gpus=8 --memory=${MEM_BYTES} \
  2>&1 | tee "${LOG_DIR}/ray_head.log"

# 2) Ray WORKERS on the other 3 nodes.
srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 \
  --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2 | paste -sd,)" \
  bash -c '
    export RLINF_NODE_RANK=$(( SLURM_PROCID + 1 ))
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl USE_TF=0 TOKENIZERS_PARALLELISM=false
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
    export LIBERO_CONFIG_PATH="'"${LIBERO_CONFIG_PATH}"'"
    export PYTHONPATH="'"${REPO_PATH}:${LIBERO_DIR}"'"
    export PATH="'"${VENV}/bin"':${PATH}"
    '"${NCCL_ENV_BLOCK}"'
    unset RAY_ADDRESS
    echo "[worker rank ${RLINF_NODE_RANK}] $(hostname) -> '"${HEAD_IP}:${RAY_PORT}"'"
    "'"${RLINF_RAY}"'" stop --force >/dev/null 2>&1 || true
    "'"${RLINF_RAY}"'" start --address="'"${HEAD_IP}:${RAY_PORT}"'" \
      --num-gpus=8 --memory='"${MEM_BYTES}"' --block
  ' 2>&1 | tee "${LOG_DIR}/ray_workers.log" &
RAY_WORKERS_PID=$!

# 3) Wait for all 32 GPUs, then launch driver.
echo "Waiting for 32 GPUs to join the Ray cluster..."
PATH="${VENV}/bin:${PATH}" bash "${REPO_PATH}/ray_utils/check_ray.sh" 32

echo "Launching training driver on ${HEAD_NODE}"
cd "${LOG_DIR}"
# shellcheck disable=SC2086
HYDRA_FULL_ERROR=1 "${RLINF_PY}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  cluster.num_nodes=4 \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_TAG}" \
  env.train.total_num_envs="${ENV_TRAIN_NUM_ENVS}" \
  ${MAX_STEPS_OVERRIDE} \
  ${RESUME_OVERRIDE} \
  ${CKPT_OVERRIDE} \
  ${LR_OVERRIDE} \
  2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
