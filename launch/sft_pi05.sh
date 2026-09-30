#!/bin/bash
# pi0.5 SFT on RoboCasa365 Composite-Seen (openpi fork, JAX, 2 nodes).
#SBATCH --job-name=sft-pi05
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --time=86:00:00
#SBATCH --output=logs/%x-%j.out

set -x
cd ${OPENPI_ROOT}

# JAX multi-node: srun starts 1 process per node; train.py initializes jax.distributed from SLURM_* vars
export JAX_PLATFORMS=""
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
# NIC names differ per node (e.g. enp75s0/enp72s0); the prefix "enp" matches all of them
export NCCL_SOCKET_IFNAME=enp
export

echo "=== pi05 2-node | nodes: $SLURM_JOB_NODELIST | nnodes=$SLURM_NNODES ==="

# pi0.5 full fine-tuning on the 16 RoboCasa365 Composite-Seen tasks, 2 nodes x 8 GPUs with FSDP
# pi05=True max_token_len=200 (knowledge-insulated dual head); everything else follows the official pi0 recipe
srun --ntasks-per-node=1 bash -c '
export JAX_PLATFORMS="" XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0 NCCL_SOCKET_IFNAME=enp
echo "task PROCID=$SLURM_PROCID NODE=$(hostname) CVD=$CUDA_VISIBLE_DEVICES"
${OPENPI_ROOT}/.venv/bin/python ${OPENPI_ROOT}/scripts/train.py \
  pi05_robocasa_finetune_target_composite_seen \
  --exp_name=cs16_2n_20260725 \
  --fsdp_devices=16 --batch_size=64 \
  --no-wandb_enabled --overwrite
'
echo "PI05_2N EXIT: $?"
