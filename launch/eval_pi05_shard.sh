#!/bin/bash
# Helper for eval_pi05_composite_seen.sbatch: one pi0.5 evaluation shard on one GPU.
set -u
DENOISE="${DENOISE:-}"

REPO=${OPENPI_ROOT}
SERVER_PY=$REPO/.venv/bin/python                                   # JAX/openpi (has NO robocasa)
CLIENT_PY=${PROJECT_ROOT}/.venv-rlinf-openpi-eval/bin/python  # isolated: robocasa+openpi_client
PORT=$((8000 + GPU))

export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=1
# PyTorch-ckpt servers torch.compile(max-autotune) at warmup; 8 servers sharing
# the default /tmp/torchinductor_$USER race on the benchmark timing files
# (InductorError: could not convert string to float: '').
# Isolate the cache per job+gpu. JAX-ckpt servers ignore this.
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_${USER}_${SLURM_JOB_ID:-x}_gpu${GPU}"
# mujoco headless render on GPU
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "[gpu $GPU] TASK=$TASK MODEL=$MODEL REPLAN=$REPLAN PORT=$PORT NODE=$(hostname)"

# ---- 1. start policy server (JAX). Cap GPU mem so mujoco has room on the shared card. ----
XLA_PYTHON_CLIENT_MEM_FRACTION=0.45 JAX_PLATFORMS="" \
  $SERVER_PY $REPO/scripts/serve_policy.py \
    --port $PORT \
    policy:checkpoint \
    --policy.config "$CONFIG" \
    --policy.dir "$CKPT_DIR" \
    ${DENOISE:+--policy.num-steps $DENOISE} \
  > "$LOG_DIR/server_${MODEL}_r${REPLAN}_${TASK}_gpu${GPU}.log" 2>&1 &
SERVER_PID=$!
echo "[gpu $GPU] server pid=$SERVER_PID, waiting for port $PORT ..."

# ---- 2. wait until the websocket port is accepting (max 10 min) ----
for i in $(seq 1 120); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "[gpu $GPU] ERROR: server died before ready. tail:"; tail -20 "$LOG_DIR/server_${MODEL}_r${REPLAN}_${TASK}_gpu${GPU}.log"; exit 1
  fi
  if $CLIENT_PY -c "import socket,sys; s=socket.socket(); s.settimeout(2); sys.exit(0 if s.connect_ex(('127.0.0.1',$PORT))==0 else 1)" 2>/dev/null; then
    echo "[gpu $GPU] server ready after ${i}x5s"; break
  fi
  sleep 5
done

# ---- 3. run the client for this ONE task ----
# NOTE: main.py entry is tyro.cli(eval_main) whose param is `args: Args`, so all
# flags are prefixed --args.* (matches README).
cd $REPO/examples/robocasa
$CLIENT_PY main.py \
  --args.host 127.0.0.1 --args.port $PORT \
  --args.env_name "$TASK" \
  --args.split target \
  --args.num_trials "$NUM_TRIALS" \
  --args.replan_steps "$REPLAN" \
  --args.log_dir "$LOG_DIR"
CLIENT_RC=$?
echo "[gpu $GPU] client rc=$CLIENT_RC for TASK=$TASK"

# ---- 4. tear down the server on this gpu ----
kill -9 $SERVER_PID 2>/dev/null
exit $CLIENT_RC
