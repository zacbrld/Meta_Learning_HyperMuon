#!/bin/bash
#SBATCH --job-name=gpt_extend3000
#SBATCH --output=/home/chetaill/muon/logs/gpt_extend3000_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_extend3000_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_extend3000}"
DATASETS_DIR="/scratch/izar/chetaill/Muon/datasets"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"
ITERATIONS=3000

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

RESUME_FROM=""
OPT=""
TAG=""
GDUO_SCOPE="tensor"
GDUO_LR_HYPER_LR=0.001
GDUO_MOMENTUM_HYPER_LR=0.001
MUON_PRECOND_KIND="adagrad_ema"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="muon_gduo_layerwise_scratch_3000"
    OPT="muon-gduo"
    GDUO_LR_HYPER_LR=1000
    GDUO_MOMENTUM_HYPER_LR=100000
    ;;
  1)
    TAG="adam_muon_gate_scratch_3000"
    OPT="adam-muon-gate"
    ;;
  2)
    TAG="muon_newton_gate_resume_3000"
    OPT="muon-newton-gate"
    RESUME_FROM="/home/chetaill/muon/results_gating/wiki_gating_3028681_1_muon_newton_strength0p10_alpha0p03_ridge1/ckpts/latest"
    ;;
  3)
    TAG="adagrad_ema_muon_scratch_3000"
    OPT="muon-precond-gate"
    MUON_PRECOND_KIND="adagrad_ema"
    ;;
  4)
    TAG="soap_lite_muon_resume_3000"
    OPT="muon-precond-gate"
    MUON_PRECOND_KIND="soap_lite"
    RESUME_FROM="/home/chetaill/muon/results_precond_gate/wiki_precond_3028637_1_muon_soap_lite_strength0p10/ckpts/latest"
    ;;
  5)
    TAG="newton_gduo_layerwise_resume_3000"
    OPT="newton-muon-gduo"
    GDUO_LR_HYPER_LR=3000
    GDUO_MOMENTUM_HYPER_LR=300000
    RESUME_FROM="/home/chetaill/muon/results_gduo_missing/wiki_gduo_missing_3028678_3_newton_gduo_layerwise_2000/ckpts/latest"
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
  --run_prefix "wiki_extend3000_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "wiki_extend3000_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
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
  --muon_lr_factor 0.02
  --grad_clip 1.0
)

if [[ -n "${RESUME_FROM}" ]]; then
  cmd+=(--resume_from "${RESUME_FROM}")
fi

case "${OPT}" in
  adam-muon-gate)
    cmd+=(
      --newton_muon_lr_factor 0.005
      --gduo_lr_hyper_lr 10000
      --gduo_gate_init 0.10
      --gduo_hypergrad_clip 0.0
      --gduo_log_interval 10
    )
    ;;
  muon-newton-gate)
    cmd+=(
      --newton_muon_lr_factor 0.005
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
      --gduo_lr_hyper_lr 3000
      --gduo_gate_init 0.03
      --gduo_hypergrad_clip 0.0
      --gduo_log_interval 10
    )
    ;;
  muon-precond-gate)
    cmd+=(
      --muon_precond_kind "${MUON_PRECOND_KIND}"
      --muon_precond_beta 0.95
      --muon_precond_eps 1e-8
      --newton_muon_precond_strength_init 0.10
      --newton_muon_precond_strength_min 0.0
      --newton_muon_precond_strength_max 1.0
      --newton_muon_precond_strength_hyper_lr 10000
      --newton_muon_precond_strength_hypergrad_clip 0.0
      --newton_muon_max_precond_dim 1024
      --gduo_log_interval 10
    )
    ;;
  muon-gduo)
    cmd+=(
      --gduo_learn_lr
      --gduo_learn_momentum
      --gduo_ema_beta 0.9
      --gduo_scope "${GDUO_SCOPE}"
      --gduo_lr_hyper_lr "${GDUO_LR_HYPER_LR}"
      --gduo_momentum_hyper_lr "${GDUO_MOMENTUM_HYPER_LR}"
      --gduo_hypergrad_clip 0.0
      --gduo_log_interval 10
    )
    ;;
  newton-muon-gduo)
    cmd+=(
      --newton_muon_lr_factor 0.005
      --newton_muon_ewma_beta 0.95
      --newton_muon_ridge 0.5
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
      --newton_muon_block_size 512
      --newton_muon_precond_clip 3.0
      --newton_muon_precond_log_interval 100
      --gduo_learn_lr
      --gduo_learn_momentum
      --gduo_ema_beta 0.9
      --gduo_scope "${GDUO_SCOPE}"
      --gduo_lr_hyper_lr "${GDUO_LR_HYPER_LR}"
      --gduo_momentum_hyper_lr "${GDUO_MOMENTUM_HYPER_LR}"
      --gduo_hypergrad_clip 0.0
      --gduo_log_interval 10
    )
    ;;
esac

echo "[EXTEND3000] task=${SLURM_ARRAY_TASK_ID} tag=${TAG}"
echo "[EXTEND3000] resume_from=${RESUME_FROM:-none}"
echo "[EXTEND3000] running: ${cmd[*]}"
"${cmd[@]}"
echo "[EXTEND3000] done"
