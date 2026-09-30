#!/bin/bash
# SimpleVLA-RL baseline (GRPO, GR00T-N1.5) on LIBERO-Long. Table 1.
#SBATCH --job-name=train-gr00t-long-simplevla-rl-grpo
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
# Isolated roots (NOT robocasa365's RLinf/config).
LIBERO_RL="${LIBERO_RL_ROOT:?export LIBERO_RL_ROOT=<RLinf + LIBERO checkout for LIBERO>}"
REPO_PATH="${LIBERO_RL}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
CONFIG_DIR="${PROJECT_ROOT}/libero/configs"
CONFIG_NAME="libero_long_grpo_gr00t_simplevla"
LIBERO_DIR="${LIBERO_RL}/LIBERO"
GROOT_DIR="${PROJECT_ROOT}/Isaac-GR00T"
RAY_PORT=29504

# Isolated venv (clone of .venv-rlinf with robosuite 1.4.0 + libero deps).
VENV="${PROJECT_ROOT}/.venv-rlinf-libero"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${REPO_PATH}/logs" "${LIBERO_RL}/logs"

RUN_TAG="${RUN_TAG:-train_gr00t_long_simplevla_rl_grpo}"
ENV_TRAIN_NUM_ENVS="${ENV_TRAIN_NUM_ENVS:-256}"
# Smoke test: set MAX_STEPS=2 to self-terminate after 2 global steps (no scancel
# needed). Empty (default) = full run (runner.max_steps stays -1 from config).
MAX_STEPS="${MAX_STEPS:-}"
MAX_STEPS_OVERRIDE=""
[ -n "${MAX_STEPS}" ] && MAX_STEPS_OVERRIDE="runner.max_steps=${MAX_STEPS}"

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_DIR}/${CONFIG_NAME}"
echo "venv:   ${VENV}"
echo "total_num_envs: ${ENV_TRAIN_NUM_ENVS}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
# Isolated RLinf source FIRST, then LIBERO + GR00T. No robocasa365 PYTHONPATH.
export PYTHONPATH="${REPO_PATH}:${LIBERO_DIR}:${GROOT_DIR}:${PYTHONPATH:-}"
# Avoid attaching to any stale Ray cluster left in node-local /tmp.
unset RAY_ADDRESS

# NCCL multi-node config — TCP/Socket mode (no RDMA), same as the RoboCasa365 launchers.
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

LOG_DIR="${LIBERO_RL}/logs/$(date +'%Y%m%d-%H%M%S')-${RUN_TAG}"
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
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    # Pin the isolated libero venv FIRST on PATH so ray actors spawn under
    # .venv-rlinf-libero (robosuite 1.4.0), not the RoboCasa365 .venv-rlinf (1.5.2).
    export PATH="'"${VENV}/bin"':${PATH}"
    export PYTHONPATH="'"${REPO_PATH}:${LIBERO_DIR}:${GROOT_DIR}"':${PYTHONPATH:-}"
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
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_TAG}" \
  env.train.total_num_envs="${ENV_TRAIN_NUM_ENVS}" \
  ${MAX_STEPS_OVERRIDE} \
  2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
