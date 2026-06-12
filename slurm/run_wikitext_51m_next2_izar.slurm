#!/bin/bash
#SBATCH --job-name=gpt_next2
#SBATCH --output=/home/chetaill/muon/logs/gpt_next2_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_next2_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_next2}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

ITERATIONS=500

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="newton_strength_ridge2p0"
    OPT="newton-muon-gduo"
    RIDGE=2.0
    ;;
  1)
    TAG="muon_newton_residual_ridge1p0_alpha0p05"
    OPT="muon-newton-gate"
    RIDGE=1.0
    ;;
  *) echo "Unknown task"; exit 1 ;;
esac

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  echo "Task 0 is downloading and preparing the dataset..."
  python - <<PY
from data.wikitext import get_wikitext_data
get_wikitext_data("${DATASETS_DIR}")
PY
  touch "${DATASETS_DIR}/.wikitext_ready"
else
  echo "Waiting for task 0 to prepare the dataset..."
  while [ ! -f "${DATASETS_DIR}/.wikitext_ready" ]; do
    sleep 60
  done
  echo "Dataset is ready!"
fi

cmd=(
  python -u src/main.py
  --run_prefix "wiki_next2_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "wiki_next2_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --results_base_folder "${RESULTS_DIR}"
  --model base
  --dataset wikitext
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 50304
  --opt "${OPT}"
  --device cuda:0
  --dtype bfloat16
  --n_layer 8
  --n_head 8
  --n_embd 512
  --sequence_length 1024
  --batch_size 12
  --acc_steps 4
  --iterations "${ITERATIONS}"
  --warmup_steps 50
  --eval_interval 100
  --eval_batches 32
  --latest_ckpt_interval 100
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
  --gduo_ema_beta 0.9
  --gduo_lr_hyper_lr 3000
  --gduo_momentum_hyper_lr 300000
  --gduo_hypergrad_clip 0.0
  --gduo_log_interval 10
  --newton_muon_lr_factor 0.005
  --newton_muon_ewma_beta 0.95
  --newton_muon_ridge "${RIDGE}"
  --newton_muon_refresh_interval 32
  --newton_muon_second_moment_init 1.0
  --newton_muon_max_precond_dim 1024
  --newton_muon_block_size 512
  --newton_muon_precond_clip 3.0
  --newton_muon_precond_log_interval 50
)

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  cmd+=(
    --newton_muon_learn_precond_strength
    --newton_muon_precond_strength_init 1.0
    --newton_muon_precond_strength_min 0.0
    --newton_muon_precond_strength_max 2.0
    --newton_muon_precond_strength_hyper_lr 1000
    --newton_muon_precond_strength_hypergrad_clip 0.0
  )
else
  cmd+=(--gduo_gate_init 0.05)
fi

echo "[NEXT2] running: ${cmd[*]}"
"${cmd[@]}"
echo "[NEXT2] done"
