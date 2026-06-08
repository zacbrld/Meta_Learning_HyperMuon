#!/bin/bash
#SBATCH --job-name=hypermuon_train
#SBATCH --output=/home/chetaill/muon/logs/hypermuon_train_%j.out
#SBATCH --error=/home/chetaill/muon/logs/hypermuon_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
RESULTS_DIR="${RESULTS_DIR:-${CODE_DIR}/results}"

MODEL="${MODEL:-mlp}"
OPTIMIZER="${OPTIMIZER:-hypermuon_l3}"
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-128}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"
cd "${CODE_DIR}"

echo "[HyperMuon] job_id=${SLURM_JOB_ID}"
echo "[HyperMuon] node=${SLURMD_NODENAME:-unknown}"
echo "[HyperMuon] code_dir=${CODE_DIR}"
echo "[HyperMuon] python=$(which python)"
echo "[HyperMuon] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[HyperMuon] model=${MODEL}"
echo "[HyperMuon] optimizer=${OPTIMIZER}"
echo "[HyperMuon] seed=${SEED}"
echo "[HyperMuon] epochs=${EPOCHS:-default}"
echo "[HyperMuon] batch_size=${BATCH_SIZE}"
echo "[HyperMuon] results_dir=${RESULTS_DIR}"

cmd=(
  python -u train.py
  --model "${MODEL}"
  --optimizer "${OPTIMIZER}"
  --seed "${SEED}"
  --batch_size "${BATCH_SIZE}"
  --results_dir "${RESULTS_DIR}"
)

if [[ -n "${EPOCHS}" ]]; then
  cmd+=(--epochs "${EPOCHS}")
fi

echo "[HyperMuon] running: ${cmd[*]}"
"${cmd[@]}"

echo "[HyperMuon] done"
