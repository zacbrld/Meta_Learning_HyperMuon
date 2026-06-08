#!/bin/bash
#SBATCH --job-name=cifar_fig1
#SBATCH --output=/home/chetaill/muon/logs/cifar_fig1_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/cifar_fig1_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
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
RESULTS_DIR="${RESULTS_DIR:-${CODE_DIR}/results_cifar_fig1}"
DATA_ROOT="${DATA_ROOT:-${CODE_DIR}/data}"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && -z "${OPTIMIZER:-}" ]]; then
  OPTIMIZERS=(adamw muon newton_muon)
  OPTIMIZER="${OPTIMIZERS[$SLURM_ARRAY_TASK_ID]}"
else
  OPTIMIZER="${OPTIMIZER:-adamw}"
fi

SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4096}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1024}"
EVAL_INTERVAL="${EVAL_INTERVAL:-24}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_SPLIT="${EVAL_SPLIT:-test}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${DATA_ROOT}"
cd "${CODE_DIR}"

echo "[CIFAR-Fig1] job_id=${SLURM_JOB_ID}"
echo "[CIFAR-Fig1] array_task=${SLURM_ARRAY_TASK_ID:-none}"
echo "[CIFAR-Fig1] node=${SLURMD_NODENAME:-unknown}"
echo "[CIFAR-Fig1] code_dir=${CODE_DIR}"
echo "[CIFAR-Fig1] python=$(which python)"
echo "[CIFAR-Fig1] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[CIFAR-Fig1] optimizer=${OPTIMIZER}"
echo "[CIFAR-Fig1] seed=${SEED}"
echo "[CIFAR-Fig1] epochs=${EPOCHS}"
echo "[CIFAR-Fig1] batch_size=${BATCH_SIZE}"
echo "[CIFAR-Fig1] results_dir=${RESULTS_DIR}"
echo "[CIFAR-Fig1] data_root=${DATA_ROOT}"

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
)

if [[ "${USE_VALIDATION_SPLIT:-0}" == "1" ]]; then
  cmd+=(--use_validation_split)
fi

echo "[CIFAR-Fig1] running: ${cmd[*]}"
"${cmd[@]}"

echo "[CIFAR-Fig1] done"
