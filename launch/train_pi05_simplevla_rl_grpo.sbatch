#!/bin/bash
# SimpleVLA-RL baseline (GRPO, pi0.5, binary reward) on RoboCasa365 Composite-Seen. Table 1.
#SBATCH --job-name=train-pi05-simplevla-rl-grpo
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=86:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<repo root>}"
REPO_PATH="${PROJECT_ROOT}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_NAME="robocasa365_grpo_openpi_pi05_composite_seen"
RAY_PORT=29529

# ISOLATED venv for OpenPi pi0/pi05 RC365 training (never touch .venv-rlinf or
# openpi-robocasa/.venv).
VENV="${PROJECT_ROOT}/.venv-rlinf-openpi-train"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${REPO_PATH}/logs"

RUN_TAG="${RUN_TAG:-train_pi05_simplevla_rl_grpo}"
ENV_TOTAL_NUM_ENVS="${ENV_TOTAL_NUM_ENVS:-256}"
# venv.py still computes info["dense_reward"], so keep a valid timing json path.
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-${PROJECT_ROOT}/structrl/demo_timing_composite_seen.json}"
# NO STRUCTRL_REWARD_VARIANT: sparse GRPO uses raw 0/1.

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_NAME}  (SimpleVLA GRPO, raw sparse 0/1, chunks=50 denoise=10 lr=5e-6)"
echo "total_num_envs: ${ENV_TOTAL_NUM_ENVS}"

# Clean up leftover processes on all 4 nodes before starting.
srun --overlap --nodes=4 --ntasks=4 --ntasks-per-node=1 bash -c '
  pkill -9 -u "$(id -un)" -f "ray::|raylet|gcs_server|train_embodied|robocasa" 2>/dev/null
  sleep 2
  echo "[cleanup $(hostname)] gpu_procs_left=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)"
' || true

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
export STRUCTRL_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export USE_TF=0
# PROJECT_ROOT first, so that this repository's structrl/ is the one imported.
export PYTHONPATH="${PROJECT_ROOT}:${REPO_PATH}:${PROJECT_ROOT}/robocasa:${PROJECT_ROOT}/robosuite:${PYTHONPATH:-}"

# NCCL multi-node config — TCP/Socket mode (no RDMA), enp NIC (not eth0).
read -r -d '' NCCL_ENV_BLOCK <<'EOF' || true
export NCCL_SOCKET_IFNAME=enp
export NCCL_NET=Socket
# H200 >=4-GPU NVLS multicast crashes nvls.cc (Cuda failure 401); disable.
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
# Do NOT export STRUCTRL_* / PYTHONPATH here (heredoc literal trap).
EOF
eval "${NCCL_ENV_BLOCK}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${RUN_TAG}"
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
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl USE_TF=0
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export STRUCTRL_PROJECT_ROOT="'"${PROJECT_ROOT}"'"
    export STRUCTRL_TIMING_JSON="'"${STRUCTRL_TIMING_JSON}"'"
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
    export PYTHONPATH="'"${PROJECT_ROOT}:${REPO_PATH}:${PROJECT_ROOT}/robocasa:${PROJECT_ROOT}/robosuite"'"
    '"${NCCL_ENV_BLOCK}"'
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
# GRPO knobs (group_size/filter_rewards/adv_type/loss_type/no value head) are
# STATIC in the grpo yaml. The overrides below follow the RoboCasa365 pi0.5 PPO launcher.
HYDRA_FULL_ERROR=1 "${RLINF_PY}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  cluster.num_nodes=4 \
  actor.micro_batch_size=16 \
  actor.model.num_action_chunks=50 \
  actor.model.num_steps=10 \
  actor.optim.lr=5.0e-6 \
  env.train.max_steps_per_rollout_epoch=2900 \
  env.eval.max_steps_per_rollout_epoch=2900 \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_TAG}" \
  env.train.total_num_envs="${ENV_TOTAL_NUM_ENVS}" \
  env.train.dense_reward.enabled=False \
  env.train.done_shift.enabled=True \
  2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
