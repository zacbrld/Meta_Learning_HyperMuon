#!/bin/bash
#SBATCH --job-name=gpt_wikitext
#SBATCH --output=/home/chetaill/muon/logs/gpt_wikitext_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_wikitext_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-2
#SBATCH --time=1:00:00
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_llm_gpt_wikitext}"
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
      --newton_muon_lr_factor "${NEWTON_MUON_LR_FACTOR:-0.01}"
      --newton_muon_ewma_beta "${NEWTON_MUON_EWMA_BETA:-0.95}"
      --newton_muon_ridge "${NEWTON_MUON_RIDGE:-0.2}"
      --newton_muon_refresh_interval 32
      --newton_muon_max_precond_dim 1024
      --newton_muon_block_size 512
    )
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

echo "[GPT-WikiText] job_id=${SLURM_JOB_ID}"
echo "[GPT-WikiText] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[GPT-WikiText] node=${SLURMD_NODENAME:-unknown}"
echo "[GPT-WikiText] code_dir=${CODE_DIR}"
echo "[GPT-WikiText] python=$(which python)"
echo "[GPT-WikiText] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[GPT-WikiText] optimizer=${OPT}"
echo "[GPT-WikiText] results_dir=${RESULTS_DIR}"
echo "[GPT-WikiText] datasets_dir=${DATASETS_DIR}"

python - <<'PY'
import tiktoken
print("tiktoken OK", getattr(tiktoken, "__version__", "unknown"))
PY

if [[ "${SLURM_ARRAY_TASK_ID}" == "0" ]]; then
  python - <<PY
from data.wikitext import get_wikitext_data
get_wikitext_data("${DATASETS_DIR}")
PY
else
  for _ in $(seq 1 180); do
    if [[ -f "${DATASETS_DIR}/wikitext/train.bin" && -f "${DATASETS_DIR}/wikitext/val.bin" ]]; then
      break
    fi
    echo "[GPT-WikiText] waiting for prepared WikiText files"
    sleep 10
  done
  if [[ ! -f "${DATASETS_DIR}/wikitext/train.bin" || ! -f "${DATASETS_DIR}/wikitext/val.bin" ]]; then
    echo "[GPT-WikiText] WikiText files were not prepared in time" >&2
    exit 1
  fi
fi

cmd=(
  python -u src/main.py
  --run_prefix "gpt_wikitext_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
)

echo "[GPT-WikiText] running: ${cmd[*]}"
"${cmd[@]}"

echo "[GPT-WikiText] done"
