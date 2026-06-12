#!/bin/bash
#SBATCH --job-name=gpt_layerwise
#SBATCH --output=/home/chetaill/muon/logs/gpt_layerwise_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_layerwise_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --chdir=/home/chetaill/muon/external/llm-baselines

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="/scratch/izar/chetaill/Muon/hf_cache"

module purge
module load gcc cuda

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV:-torch_env}"

CODE_DIR="${CODE_DIR:-/home/chetaill/muon/external/llm-baselines}"
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_layerwise}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) OPT="adamw-gduo" ;;
  1) OPT="muon-gduo" ;;
  2) OPT="newton-muon-gduo" ;;
  *) echo "Unknown task"; exit 1 ;;
esac

# Pre-download Fineweb-edu using Task 0 to prevent cross-node race conditions on parallel filesystems
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  echo "Task 0 is downloading and preparing the dataset..."
  python - <<PY
from data.fineweb_edu import get_fineweb_edu_data
get_fineweb_edu_data("${DATASETS_DIR}")
PY
  touch "${DATASETS_DIR}/.fineweb_edu_ready"
else
  echo "Waiting for task 0 to prepare the dataset..."
  while [ ! -f "${DATASETS_DIR}/.fineweb_edu_ready" ]; do
    sleep 60
  done
  echo "Dataset is ready!"
fi

cmd=(
  python -u src/main.py
  --run_prefix "layerwise_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${OPT}"
  --experiment_name "layerwise_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${OPT}"
  --results_base_folder "${RESULTS_DIR}"
  --model base
  --dataset finewebedu
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 50304
  --opt "${OPT}"
  --device cuda:0
  --dtype bfloat16
  --n_layer 12
  --n_head 12
  --n_embd 768
  --sequence_length 1024
  --batch_size 12
  --acc_steps 4
  --iterations 8000
  --warmup_steps 400
  --eval_interval 500
  --eval_batches 32
  --latest_ckpt_interval 500
  --log_interval 50
  --scheduler cos
  --lr 0.001
  --weight_decay 0.1
  --beta1 0.9
  --beta2 0.95
  --momentum 0.95
  --nesterov True
  --muon_ns_steps 5
  --grad_clip 1.0
  --gduo_learn_lr
  --gduo_learn_momentum
  --gduo_ema_beta 0.9
  --gduo_hypergrad_clip 0.0
  --gduo_log_interval 50
)

if [[ "${OPT}" == *"muon"* ]]; then
  cmd+=(--muon_lr_factor 0.02)
fi
if [[ "${OPT}" == *"newton-muon"* ]]; then
  cmd+=(
    --newton_muon_lr_factor 0.005
    --newton_muon_ewma_beta 0.95
    --newton_muon_ridge 0.5
    --newton_muon_refresh_interval 32
    --newton_muon_max_precond_dim 1024
    --newton_muon_block_size 512
    --newton_muon_precond_clip 3.0
    --newton_muon_precond_log_interval 200
  )
fi

echo "[LAYERWISE] running: ${cmd[*]}"
"${cmd[@]}"
echo "[LAYERWISE] done"
