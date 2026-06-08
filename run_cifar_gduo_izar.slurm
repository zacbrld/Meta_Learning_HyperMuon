#!/bin/bash
#SBATCH --job-name=cifar_gduo
#SBATCH --output=/home/chetaill/muon/logs/cifar_gduo_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/cifar_gduo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/chetaill/muon

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load gcc cuda

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV:-torch_env}"

CODE_DIR="${CODE_DIR:-/home/chetaill/muon}"
LOG_DIR="${LOG_DIR:-${CODE_DIR}/logs}"
RESULTS_DIR="${RESULTS_DIR:-${CODE_DIR}/results_cifar_gduo_lr_cosine}"
DATA_ROOT="${DATA_ROOT:-${CODE_DIR}/data}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${OPTIMIZER:-}" ]]; then
  OPTIMIZERS=(adamw_gduo_lr muon_gduo_lr newton_muon_gduo_lr)
  OPTIMIZER="${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}"
else
  OPTIMIZER="${OPTIMIZER:-adamw_gduo_lr}"
fi

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
EVAL_INTERVAL="${EVAL_INTERVAL:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"
WARMUP_STEPS="${WARMUP_STEPS:-100}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
GDUO_HYPER_LR="${GDUO_HYPER_LR:-1e-3}"
GDUO_HYPERGRAD_CLIP="${GDUO_HYPERGRAD_CLIP:-1.0}"
GDUO_LR_MIN_RATIO="${GDUO_LR_MIN_RATIO:-0.05}"
GDUO_LR_MAX_RATIO="${GDUO_LR_MAX_RATIO:-5.0}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${DATA_ROOT}"
cd "${CODE_DIR}"

echo "[CIFAR-GDUO] job_id=${SLURM_JOB_ID}"
echo "[CIFAR-GDUO] array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[CIFAR-GDUO] node=${SLURMD_NODENAME:-unknown}"
echo "[CIFAR-GDUO] code_dir=${CODE_DIR}"
echo "[CIFAR-GDUO] python=$(which python)"
echo "[CIFAR-GDUO] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[CIFAR-GDUO] optimizer=${OPTIMIZER}"
echo "[CIFAR-GDUO] seed=${SEED}"
echo "[CIFAR-GDUO] epochs=${EPOCHS}"
echo "[CIFAR-GDUO] batch_size=${BATCH_SIZE}"
echo "[CIFAR-GDUO] warmup_steps=${WARMUP_STEPS}"
echo "[CIFAR-GDUO] min_lr_ratio=${MIN_LR_RATIO}"
echo "[CIFAR-GDUO] gduo_hyper_lr=${GDUO_HYPER_LR}"
echo "[CIFAR-GDUO] results_dir=${RESULTS_DIR}"

cmd=(
  python -u train_cifar_fig1.py
  --optimizer "${OPTIMIZER}"
  --seed "${SEED}"
  --epochs "${EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --eval_batch_size "${EVAL_BATCH_SIZE}"
  --eval_interval "${EVAL_INTERVAL}"
  --num_workers "${NUM_WORKERS}"
  --data_root "${DATA_ROOT}"
  --results_dir "${RESULTS_DIR}"
  --eval_split "${EVAL_SPLIT}"
  --warmup_steps "${WARMUP_STEPS}"
  --min_lr_ratio "${MIN_LR_RATIO}"
  --gduo_hyper_lr "${GDUO_HYPER_LR}"
  --gduo_hypergrad_clip "${GDUO_HYPERGRAD_CLIP}"
  --gduo_lr_min_ratio "${GDUO_LR_MIN_RATIO}"
  --gduo_lr_max_ratio "${GDUO_LR_MAX_RATIO}"
)

if [[ "${USE_VALIDATION_SPLIT:-0}" == "1" ]]; then
  cmd+=(--use_validation_split)
fi

echo "[CIFAR-GDUO] running: ${cmd[*]}"
"${cmd[@]}"

echo "[CIFAR-GDUO] done"
