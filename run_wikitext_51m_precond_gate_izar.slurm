#!/bin/bash
#SBATCH --job-name=gpt_precond_wiki
#SBATCH --output=/home/chetaill/muon/logs/gpt_precond_wiki_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_precond_wiki_%A_%a.err
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_precond_gate}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"
ITERATIONS="${ITERATIONS:-500}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="muon_adagrad_ema_strength0p10"
    PRECOND_KIND="adagrad_ema"
    ;;
  1)
    TAG="muon_soap_lite_strength0p10"
    PRECOND_KIND="soap_lite"
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
  --run_prefix "wiki_precond_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "wiki_precond_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --results_base_folder "${RESULTS_DIR}"
  --model base
  --dataset wikitext
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 50304
  --opt muon-precond-gate
  --muon_precond_kind "${PRECOND_KIND}"
  --muon_precond_beta 0.95
  --muon_precond_eps 1e-8
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
  --muon_lr_factor 0.02
  --grad_clip 1.0
  --newton_muon_precond_strength_init 0.10
  --newton_muon_precond_strength_min 0.0
  --newton_muon_precond_strength_max 1.0
  --newton_muon_precond_strength_hyper_lr 10000
  --newton_muon_precond_strength_hypergrad_clip 0.0
  --newton_muon_max_precond_dim 1024
  --gduo_log_interval 10
)

echo "[PRECOND-GATE] running: ${cmd[*]}"
"${cmd[@]}"
echo "[PRECOND-GATE] done"
