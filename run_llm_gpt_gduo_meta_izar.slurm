#!/bin/bash
#SBATCH --job-name=gpt_gduo_meta
#SBATCH --output=/home/chetaill/muon/logs/gpt_gduo_meta_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_gduo_meta_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-5
#SBATCH --time=2:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --chdir=/home/chetaill/muon/external/llm-baselines

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load gcc cuda

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV:-torch_env}"

CODE_DIR="${CODE_DIR:-/home/chetaill/muon/external/llm-baselines}"
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_llm_gpt_gduo_meta}"
DATASETS_DIR="${DATASETS_DIR:-${CODE_DIR}/src/data/datasets}"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

SEED="${SEED:-0}"
COMMON_GDUO_ARGS=(
  --gduo_lr_hyper_lr "${GDUO_LR_HYPER_LR:-0.001}"
  --gduo_momentum_hyper_lr "${GDUO_MOMENTUM_HYPER_LR:-0.001}"
  --gduo_hypergrad_clip "${GDUO_HYPERGRAD_CLIP:-1.0}"
  --gduo_lr_min_ratio "${GDUO_LR_MIN_RATIO:-0.25}"
  --gduo_lr_max_ratio "${GDUO_LR_MAX_RATIO:-4.0}"
  --gduo_log_interval 200
)

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    OPT="muon-gduo"
    TAG="muon_gduo_lr_seed${SEED}"
    META_ARGS=(--gduo_learn_lr)
    MATRIX_ARGS=(--muon_lr_factor 0.02)
    ;;
  1)
    OPT="muon-gduo"
    TAG="muon_gduo_mu_seed${SEED}"
    META_ARGS=(--gduo_learn_momentum)
    MATRIX_ARGS=(--muon_lr_factor 0.02)
    ;;
  2)
    OPT="muon-gduo"
    TAG="muon_gduo_lr_mu_seed${SEED}"
    META_ARGS=(--gduo_learn_lr --gduo_learn_momentum)
    MATRIX_ARGS=(--muon_lr_factor 0.02)
    ;;
  3)
    OPT="newton-muon-gduo"
    TAG="newton_muon_gduo_lr_seed${SEED}"
    META_ARGS=(--gduo_learn_lr)
    MATRIX_ARGS=(
      --newton_muon_lr_factor 0.005
      --newton_muon_ewma_beta 0.95
      --newton_muon_ridge 0.5
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
      --newton_muon_block_size 512
      --newton_muon_precond_clip 3.0
      --newton_muon_precond_log_interval 200
    )
    ;;
  4)
    OPT="newton-muon-gduo"
    TAG="newton_muon_gduo_mu_seed${SEED}"
    META_ARGS=(--gduo_learn_momentum)
    MATRIX_ARGS=(
      --newton_muon_lr_factor 0.005
      --newton_muon_ewma_beta 0.95
      --newton_muon_ridge 0.5
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
      --newton_muon_block_size 512
      --newton_muon_precond_clip 3.0
      --newton_muon_precond_log_interval 200
    )
    ;;
  5)
    OPT="newton-muon-gduo"
    TAG="newton_muon_gduo_lr_mu_seed${SEED}"
    META_ARGS=(--gduo_learn_lr --gduo_learn_momentum)
    MATRIX_ARGS=(
      --newton_muon_lr_factor 0.005
      --newton_muon_ewma_beta 0.95
      --newton_muon_ridge 0.5
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
      --newton_muon_block_size 512
      --newton_muon_precond_clip 3.0
      --newton_muon_precond_log_interval 200
    )
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${DATASETS_DIR}/wikitext/train.bin" || ! -f "${DATASETS_DIR}/wikitext/val.bin" ]]; then
  echo "[GPT-GDUO-META] Missing prepared WikiText files in ${DATASETS_DIR}/wikitext" >&2
  exit 1
fi

echo "[GPT-GDUO-META] job_id=${SLURM_JOB_ID}"
echo "[GPT-GDUO-META] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[GPT-GDUO-META] node=${SLURMD_NODENAME:-unknown}"
echo "[GPT-GDUO-META] optimizer=${OPT}"
echo "[GPT-GDUO-META] seed=${SEED}"
echo "[GPT-GDUO-META] tag=${TAG}"
echo "[GPT-GDUO-META] results_dir=${RESULTS_DIR}"

cmd=(
  python -u src/main.py
  --run_prefix "gpt_gduo_meta_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "gpt_gduo_meta_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --results_base_folder "${RESULTS_DIR}"
  --seed "${SEED}"
  --data_seed 1337
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
  --sequence_length 512
  --batch_size 24
  --acc_steps 1
  --iterations 2000
  --warmup_steps 200
  --eval_interval 200
  --eval_batches 32
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
  "${MATRIX_ARGS[@]}"
  "${META_ARGS[@]}"
  "${COMMON_GDUO_ARGS[@]}"
)

echo "[GPT-GDUO-META] running: ${cmd[*]}"
"${cmd[@]}"

echo "[GPT-GDUO-META] done"
