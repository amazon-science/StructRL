#!/bin/bash
# Beta sensitivity (appendix): StructRL on RoboCasa365 with beta = 2.0 (STRUCTRL_BAND_CAP=2.0, ablations/beta_sweep/sitecustomize.py).
#SBATCH --job-name=train-gr00t-structrl-beta20
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --time=86:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?export PROJECT_ROOT=<directory holding RLinf, structrl, robocasa, ...>}"
REPO_PATH="${PROJECT_ROOT}/RLinf"
EMBODIED_PATH="${REPO_PATH}/examples/embodiment"
PYHOOK_DIR="${PROJECT_ROOT}/ablations/beta_sweep"
CONFIG_NAME="robocasa365_ppo_gr00t_composite_seen"
RAY_PORT=29500

VENV="${PROJECT_ROOT}/.venv-rlinf"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${REPO_PATH}/logs"

RUN_TAG="${RUN_TAG:-train_gr00t_structrl_beta2.0}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
ENV_TOTAL_NUM_ENVS="${ENV_TOTAL_NUM_ENVS:-256}"
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-${PROJECT_ROOT}/structrl/demo_timing_composite_seen.json}"
# structrl variant + beta_sweep hook: band numerator 0.6 -> 2.0.
export STRUCTRL_REWARD_VARIANT=structrl
export STRUCTRL_BAND_CAP=2.0

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_NAME}"
echo "reward_variant: ${STRUCTRL_REWARD_VARIANT}"
echo "band_cap: ${STRUCTRL_BAND_CAP}"
echo "total_num_envs: ${ENV_TOTAL_NUM_ENVS}"
echo "timing_json: ${STRUCTRL_TIMING_JSON}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
export STRUCTRL_PROJECT_ROOT="${PROJECT_ROOT}"
# PYHOOK_DIR first so beta_sweep sitecustomize auto-imports in every process.
export PYTHONPATH="${PYHOOK_DIR}:${PROJECT_ROOT}:${REPO_PATH}:${PROJECT_ROOT}/robocasa:${PROJECT_ROOT}/robosuite:${PROJECT_ROOT}/Isaac-GR00T:${PYTHONPATH:-}"

# NCCL multi-node config — TCP/Socket mode (no RDMA), enp NIC (H200 node).
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
# Do NOT export STRUCTRL_* / PYTHONPATH here (heredoc literal trap). They are
# exported as expanded literals inside the srun worker block below instead.
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

# 2) Ray WORKERS on the other 3 nodes. STRUCTRL_* must reach the spawned env
# workers, and PYTHONPATH (incl. pyhook, expanded at composition time — NOT via
# the heredoc) so the beta_sweep sitecustomize auto-imports in every worker process.
srun --overlap --nodes=3 --ntasks=3 --ntasks-per-node=1 \
  --nodelist="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n +2 | paste -sd,)" \
  bash -c '
    export RLINF_NODE_RANK=$(( SLURM_PROCID + 1 ))
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export STRUCTRL_REWARD_VARIANT=structrl
    export STRUCTRL_BAND_CAP=2.0
    export PYTHONPATH="'"${PYHOOK_DIR}:${PROJECT_ROOT}:${REPO_PATH}:${PROJECT_ROOT}/robocasa:${PROJECT_ROOT}/robosuite:${PROJECT_ROOT}/Isaac-GR00T"':${PYTHONPATH:-}"
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
  --config-path "${EMBODIED_PATH}/config/" \
  --config-name "${CONFIG_NAME}" \
  cluster.num_nodes=4 \
  actor.micro_batch_size=16 \
  runner.logger.log_path="${LOG_DIR}" \
  runner.logger.experiment_name="${RUN_TAG}" \
  env.train.total_num_envs="${ENV_TOTAL_NUM_ENVS}" \
  env.train.dense_reward.enabled=True \
  env.train.done_shift.enabled=True \
  runner.max_epochs=100 \
  ${EXTRA_OVERRIDES} 2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
