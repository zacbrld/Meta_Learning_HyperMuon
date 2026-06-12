#!/bin/bash
#SBATCH --job-name=llm_gpt_base
#SBATCH --output=/home/chetaill/muon/logs/llm_gpt_baseline_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/llm_gpt_baseline_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=0:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --chdir=/home/chetaill/muon/external/llm-baselines

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load gcc cuda

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV:-torch_env}"

CODE_DIR="${CODE_DIR:-/home/chetaill/muon/external/llm-baselines}"
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_llm_gpt_baselines}"
DATASETS_DIR="${DATASETS_DIR:-${CODE_DIR}/src/data/datasets}"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${DATASETS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    OPT="adamw"
    MATRIX_ARGS=()
    ;;
  1)
    OPT="muon"
    MATRIX_ARGS=(--muon_lr_factor 0.02)
    ;;
  2)
    OPT="newton-muon"
    MATRIX_ARGS=(
      --newton_muon_lr_factor 0.02
      --newton_muon_ewma_beta 0.95
      --newton_muon_ridge 0.2
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
    )
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "[LLM-GPT] job_id=${SLURM_JOB_ID}"
echo "[LLM-GPT] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[LLM-GPT] node=${SLURMD_NODENAME:-unknown}"
echo "[LLM-GPT] code_dir=${CODE_DIR}"
echo "[LLM-GPT] python=$(which python)"
echo "[LLM-GPT] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[LLM-GPT] optimizer=${OPT}"
echo "[LLM-GPT] results_dir=${RESULTS_DIR}"
echo "[LLM-GPT] datasets_dir=${DATASETS_DIR}"

(
  flock 9
  python - <<PY
from data.shakespeare import get_shakespeare_data
get_shakespeare_data("${DATASETS_DIR}")
PY
) 9>"${DATASETS_DIR}/.shakespeare.lock"

cmd=(
  python -u src/main.py
  --run_prefix "llm_gpt_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  --results_base_folder "${RESULTS_DIR}"
  --model base
  --dataset shakespeare-char
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 96
  --opt "${OPT}"
  --device cuda:0
  --dtype bfloat16
  --n_layer 4
  --n_head 4
  --n_embd 256
  --sequence_length 256
  --batch_size 32
  --acc_steps 1
  --iterations 300
  --warmup_steps 30
  --eval_interval 50
  --eval_batches 16
  --log_interval 10
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
)

echo "[LLM-GPT] running: ${cmd[*]}"
"${cmd[@]}"

echo "[LLM-GPT] done"
