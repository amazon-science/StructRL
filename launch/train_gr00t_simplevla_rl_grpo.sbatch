#!/bin/bash
# SimpleVLA-RL baseline (GRPO, group size 8, binary reward, critic-free, GR00T-N1.5) on RoboCasa365.
# Needs diag/ on PYTHONPATH (registers adv_type=grpo_diag). Table 1.
#SBATCH --job-name=train-gr00t-simplevla-rl-grpo
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
CONFIG_NAME="robocasa365_grpo_gr00t_composite_seen"
RAY_PORT=29501

VENV="${PROJECT_ROOT}/.venv-rlinf"
RLINF_PY="${VENV}/bin/python"
RLINF_RAY="${VENV}/bin/ray"

mkdir -p "${REPO_PATH}/logs"

RUN_TAG="${RUN_TAG:-train_gr00t_simplevla_rl_grpo}"
ENV_TOTAL_NUM_ENVS="${ENV_TOTAL_NUM_ENVS:-256}"
# Smoke test: set MAX_STEPS=2 to self-terminate after 2 global steps (no scancel
# needed). Empty (default) = full run (runner.max_steps stays -1 from config).
MAX_STEPS="${MAX_STEPS:-}"
MAX_STEPS_OVERRIDE=""
[ -n "${MAX_STEPS}" ] && MAX_STEPS_OVERRIDE="runner.max_steps=${MAX_STEPS}"
# Resume: set RESUME_DIR to a global_step_N checkpoint dir to continue training.
RESUME_DIR="${RESUME_DIR:-}"
RESUME_OVERRIDE=""
[ -n "${RESUME_DIR}" ] && RESUME_OVERRIDE="runner.resume_dir=${RESUME_DIR}"
# Side-channel diagnostic only (sparse/GRPO does not USE this reward), but
# venv.py still computes info["dense_reward"], so keep the path valid.
export STRUCTRL_TIMING_JSON="${STRUCTRL_TIMING_JSON:-${PROJECT_ROOT}/structrl/demo_timing_composite_seen.json}"

HEAD_NODE=$(hostname)
HEAD_IP=$(hostname -I | awk '{print $1}')
echo "===== ${RUN_TAG} head=${HEAD_NODE} (${HEAD_IP}:${RAY_PORT}) at $(date) ====="
echo "config: ${CONFIG_NAME}"
echo "total_num_envs: ${ENV_TOTAL_NUM_ENVS}"
echo "timing_json: ${STRUCTRL_TIMING_JSON}"

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export EMBODIED_PATH REPO_PATH
# PROJECT_ROOT FIRST so `import structrl.*` resolves to the live source tree,
# not the stale partial copy in .venv-rlinf/site-packages.
export STRUCTRL_PROJECT_ROOT="${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/grpo_diag:${PROJECT_ROOT}:${REPO_PATH}:${PROJECT_ROOT}/robocasa:${PROJECT_ROOT}/robosuite:${PROJECT_ROOT}/Isaac-GR00T:${PYTHONPATH:-}"

# NCCL multi-node config — TCP/Socket mode (no RDMA). Same as sparse PPO.
read -r -d '' NCCL_ENV_BLOCK <<'EOF' || true
export NCCL_SOCKET_IFNAME=ens
export NCCL_NET=Socket
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET
export NCCL_SOCKET_NTHREADS=4
export NCCL_NSOCKS_PERTHREAD=4
# Do NOT export STRUCTRL_*/PYTHONPATH here: this heredoc is eval'd in srun
# child shells and the single-quoted literal round-trips into a broken string.
# venv.py + timing_reward_builder.py have hardcoded path defaults, so workers
# don't need these env vars; the main shell already exports them for the driver.
EOF
eval "${NCCL_ENV_BLOCK}"

LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H%M%S')-${RUN_TAG}"
mkdir -p "${LOG_DIR}"
export GRPO_DIAG_DUMP="${LOG_DIR}/grpo_diag_dump.jsonl"
echo "GRPO_DIAG_DUMP=${GRPO_DIAG_DUMP}"
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
  ${MAX_STEPS_OVERRIDE} \
  ${RESUME_OVERRIDE} \
  2>&1 | tee "${LOG_DIR}/run.log"

echo "===== ${RUN_TAG} done at $(date) ====="
kill "${RAY_WORKERS_PID}" 2>/dev/null || true
"${RLINF_RAY}" stop --force >/dev/null 2>&1 || true
