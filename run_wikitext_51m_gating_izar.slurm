#!/bin/bash
#SBATCH --job-name=gpt_gate_wiki
#SBATCH --output=/home/chetaill/muon/logs/gpt_gate_wiki_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_gate_wiki_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-1
#SBATCH --time=06:00:00
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_gating}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"
ITERATIONS="${ITERATIONS:-500}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="adam_muon_residual_alpha0p10_gate10k"
    OPT="adam-muon-gate"
    BASE_LR=0.001
    MUON_LR_FACTOR=0.02
    NEWTON_LR_FACTOR=0.005
    GATE_INIT=0.10
    GATE_HYPER_LR=10000
    ;;
  1)
    TAG="muon_newton_strength0p10_alpha0p03_ridge1"
    OPT="muon-newton-gate"
    BASE_LR=0.001
    MUON_LR_FACTOR=0.02
    NEWTON_LR_FACTOR=0.005
    GATE_INIT=0.03
    GATE_HYPER_LR=3000
    ;;
  *) echo "Unknown task"; exit 1 ;;
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
  --run_prefix "wiki_gating_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "wiki_gating_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
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
  --lr "${BASE_LR}"
  --weight_decay 0.1
  --beta1 0.9
  --beta2 0.95
  --momentum 0.95
  --nesterov True
  --muon_ns_steps 5
  --muon_lr_factor "${MUON_LR_FACTOR}"
  --newton_muon_lr_factor "${NEWTON_LR_FACTOR}"
  --grad_clip 1.0
  --gduo_lr_hyper_lr "${GATE_HYPER_LR}"
  --gduo_gate_init "${GATE_INIT}"
  --gduo_hypergrad_clip 0.0
  --gduo_log_interval 10
)

if [[ "${OPT}" == "muon-newton-gate" ]]; then
  cmd+=(
    --newton_muon_ewma_beta 0.95
    --newton_muon_ridge 1.0
    --newton_muon_refresh_interval 32
    --newton_muon_second_moment_init 1.0
    --newton_muon_max_precond_dim 1024
    --newton_muon_block_size 512
    --newton_muon_precond_clip 3.0
    --newton_muon_precond_strength_init 0.10
    --newton_muon_precond_strength_min 0.0
    --newton_muon_precond_strength_max 1.0
    --newton_muon_precond_strength_hyper_lr 10000
    --newton_muon_precond_strength_hypergrad_clip 0.0
  )
fi

echo "[GATING] running: ${cmd[*]}"
"${cmd[@]}"
echo "[GATING] done"
