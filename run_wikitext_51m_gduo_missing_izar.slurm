#!/bin/bash
#SBATCH --job-name=gpt_gduo_missing
#SBATCH --output=/home/chetaill/muon/logs/gpt_gduo_missing_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_gduo_missing_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_gduo_missing}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

ITERATIONS=2000
case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="adamw_gduo_global_2000"
    OPT="adamw-gduo"
    GDUO_SCOPE="global"
    GDUO_LR_HYPER_LR=100
    GDUO_MOMENTUM_HYPER_LR=5000
    ;;
  1)
    TAG="adamw_gduo_layerwise_2000"
    OPT="adamw-gduo"
    GDUO_SCOPE="tensor"
    GDUO_LR_HYPER_LR=100
    GDUO_MOMENTUM_HYPER_LR=5000
    ;;
  2)
    TAG="newton_gduo_global_2000"
    OPT="newton-muon-gduo"
    GDUO_SCOPE="global"
    GDUO_LR_HYPER_LR=3000
    GDUO_MOMENTUM_HYPER_LR=300000
    ;;
  3)
    TAG="newton_gduo_layerwise_2000"
    OPT="newton-muon-gduo"
    GDUO_SCOPE="tensor"
    GDUO_LR_HYPER_LR=3000
    GDUO_MOMENTUM_HYPER_LR=300000
    ;;
  *) echo "Unknown task ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

if [ ! -f "${DATASETS_DIR}/.wikitext_ready" ]; then
  echo "Preparing Wikitext dataset..."
  python - <<PY
from data.wikitext import get_wikitext_data
get_wikitext_data("${DATASETS_DIR}")
PY
  touch "${DATASETS_DIR}/.wikitext_ready"
fi

cmd=(
  python -u src/main.py
  --run_prefix "wiki_gduo_missing_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "wiki_gduo_missing_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
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
  --gduo_learn_lr
  --gduo_learn_momentum
  --gduo_ema_beta 0.9
  --gduo_scope "${GDUO_SCOPE}"
  --gduo_lr_hyper_lr "${GDUO_LR_HYPER_LR}"
  --gduo_momentum_hyper_lr "${GDUO_MOMENTUM_HYPER_LR}"
  --gduo_hypergrad_clip 0.0
  --gduo_log_interval 10
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
    --newton_muon_precond_log_interval 100
  )
fi

echo "[GDUO-MISSING] running: ${cmd[*]}"
"${cmd[@]}"
echo "[GDUO-MISSING] done"
