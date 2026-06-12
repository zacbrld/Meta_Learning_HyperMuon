#!/bin/bash
#SBATCH --job-name=gpt_adam_seeds
#SBATCH --output=/home/chetaill/muon/logs/gpt_adam_seeds_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_adam_seeds_%A_%a.err
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_llm_gpt_adam_seed_confirm}"
DATASETS_DIR="${DATASETS_DIR:-${CODE_DIR}/src/data/datasets}"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

SEED="${SLURM_ARRAY_TASK_ID}"
TAG="adamw_seed${SEED}"

if [[ ! -f "${DATASETS_DIR}/wikitext/train.bin" || ! -f "${DATASETS_DIR}/wikitext/val.bin" ]]; then
  echo "[GPT-ADAM-SEEDS] Missing prepared WikiText files in ${DATASETS_DIR}/wikitext" >&2
  exit 1
fi

echo "[GPT-ADAM-SEEDS] job_id=${SLURM_JOB_ID}"
echo "[GPT-ADAM-SEEDS] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[GPT-ADAM-SEEDS] node=${SLURMD_NODENAME:-unknown}"
echo "[GPT-ADAM-SEEDS] optimizer=adamw"
echo "[GPT-ADAM-SEEDS] seed=${SEED}"
echo "[GPT-ADAM-SEEDS] tag=${TAG}"
echo "[GPT-ADAM-SEEDS] results_dir=${RESULTS_DIR}"

cmd=(
  python -u src/main.py
  --run_prefix "gpt_adam_seeds_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "gpt_adam_seeds_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --results_base_folder "${RESULTS_DIR}"
  --seed "${SEED}"
  --data_seed 1337
  --model base
  --dataset wikitext
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 50304
  --opt adamw
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
)

echo "[GPT-ADAM-SEEDS] running: ${cmd[*]}"
"${cmd[@]}"

echo "[GPT-ADAM-SEEDS] done"
