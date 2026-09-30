#!/bin/bash
# Fig. 3 '+ Subtask Reward' arm on LIBERO-Long: fixed subtask reward WITHOUT gating (LIBERO_NO_PRECEDENCE=1).
#SBATCH --job-name=train-gr00t-libero-long-ungated-fixed-ppo
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
LIBERO_RL="${LIBERO_RL_ROOT:?export LIBERO_RL_ROOT=<LIBERO RLinf root>}"
NOPREC="${PROJECT_ROOT}/ablations/ungated"
REPO_PATH="${LIBERO_RL}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_DIR="${PROJECT_ROOT}/libero/configs"
CONFIG_NAME="libero_long_ppo_gr00t_structrl"
LIBERO_DIR="${LIBERO_RL}/LIBERO"
GROOT_DIR="${PROJECT_ROOT}/Isaac-GR00T"
RAY_PORT="${RAY_PORT:-29569}"

VENV="${PROJECT_ROOT}/.venv-rlinf-libero"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${NOPREC}/logs"

RUN_TAG="${RUN_TAG:-train_gr00t_libero_long_ungated_fixed_ppo}"
ENV_TRAIN_NUM_ENVS="${ENV_TRAIN_NUM_ENVS:-256}"
TIMING_JSON="${LIBERO_TIMING_JSON:-${PROJECT_ROOT}/libero/timing/libero_long_demo_timing.json}"

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_DIR}/${CONFIG_NAME}"
echo "reward_variant: flat + NO_PRECEDENCE (pyhook)"
echo "total_num_envs: ${ENV_TRAIN_NUM_ENVS}"
echo "timing_json: ${TIMING_JSON}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
# pyhook dir FIRST so its sitecustomize loads in every process.
export PYTHONPATH="${NOPREC}:${REPO_PATH}:${LIBERO_DIR}:${GROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_DENSE_REWARD=1
export LIBERO_REWARD_VARIANT=fixed
export LIBERO_NO_PRECEDENCE=1
export LIBERO_TIMING_JSON="${TIMING_JSON}"
unset RAY_ADDRESS

# NCCL multi-node config — TCP/Socket mode (no RDMA), enp NIC (H200 node).
read -r -d '' NCCL_ENV_BLOCK <<'EOF' || true
export NCCL_SOCKET_IFNAME=enp
export NCCL_NET=Socket
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
EOF
eval "${NCCL_ENV_BLOCK}"

LOG_DIR="${NOPREC}/logs/$(date +'%Y%m%d-%H%M%S')-${RUN_TAG}"
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

# 2) Ray WORKERS on the other 3 nodes. Gate + variant + timing json + pyhook
# PYTHONPATH must reach the spawned env workers (ray captures env per node).
srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 \
  --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2 | paste -sd,)" \
  bash -c '
    export RLINF_NODE_RANK=$(( SLURM_PROCID + 1 ))
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export PATH="'"${VENV}/bin"':${PATH}"
    export PYTHONPATH="'"${NOPREC}:${REPO_PATH}:${LIBERO_DIR}:${GROOT_DIR}"':${PYTHONPATH:-}"
    export LIBERO_DENSE_REWARD=1
    export LIBERO_REWARD_VARIANT=fixed
    export LIBERO_NO_PRECEDENCE=1
    export LIBERO_TIMING_JSON="'"${TIMING_JSON}"'"
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
# shellcheck disable=SC2086
HYDRA_FULL_ERROR=1 "${RLINF_PY}" "${EMBODIED_PATH}/train_embodied_agent.py" \
  --config-path "${CONFIG_DIR}" \
  --config-name "${CONFIG_NAME}" \
  cluster.num_nodes=4 \
  runner.max_epochs="${MAX_EPOCHS:-100}" \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_TAG}" \
  env.train.total_num_envs="${ENV_TRAIN_NUM_ENVS}" \
  2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
