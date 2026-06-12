#!/bin/bash
#SBATCH --job-name=gpt_nm_stab
#SBATCH --output=/home/chetaill/muon/logs/gpt_nmstab_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/gpt_nmstab_%A_%a.err
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
RESULTS_DIR="${RESULTS_DIR:-/home/chetaill/muon/results_llm_newton_stability}"
DATASETS_DIR="${DATASETS_DIR:-${CODE_DIR}/src/data/datasets}"
LOG_DIR="${LOG_DIR:-/home/chetaill/muon/logs}"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"
cd "${CODE_DIR}"
export PYTHONPATH="${CODE_DIR}/src:${PYTHONPATH:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0)
    TAG="lr0p005_ridge0p5_clip3"
    NEWTON_LR="0.005"
    RIDGE="0.5"
    PRECOND_CLIP="3.0"
    ;;
  1)
    TAG="lr0p004_ridge0p5_clip3"
    NEWTON_LR="0.004"
    RIDGE="0.5"
    PRECOND_CLIP="3.0"
    ;;
  2)
    TAG="lr0p004_ridge1p0_clip3"
    NEWTON_LR="0.004"
    RIDGE="1.0"
    PRECOND_CLIP="3.0"
    ;;
  *)
    echo "Unknown array task: ${SLURM_ARRAY_TASK_ID}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${DATASETS_DIR}/wikitext/train.bin" || ! -f "${DATASETS_DIR}/wikitext/val.bin" ]]; then
  echo "[GPT-NM-STAB] Missing prepared WikiText files in ${DATASETS_DIR}/wikitext" >&2
  exit 1
fi

echo "[GPT-NM-STAB] job_id=${SLURM_JOB_ID}"
echo "[GPT-NM-STAB] array_task=${SLURM_ARRAY_TASK_ID}"
echo "[GPT-NM-STAB] node=${SLURMD_NODENAME:-unknown}"
echo "[GPT-NM-STAB] tag=${TAG}"
echo "[GPT-NM-STAB] newton_lr=${NEWTON_LR}"
echo "[GPT-NM-STAB] ridge=${RIDGE}"
echo "[GPT-NM-STAB] precond_clip=${PRECOND_CLIP}"
echo "[GPT-NM-STAB] results_dir=${RESULTS_DIR}"

cmd=(
  python -u src/main.py
  --run_prefix "gpt_nmstab_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --experiment_name "gpt_nmstab_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${TAG}"
  --results_base_folder "${RESULTS_DIR}"
  --model base
  --dataset wikitext
  --datasets_dir "${DATASETS_DIR}"
  --data_in_ram
  --vocab_size 50304
  --opt newton-muon
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
  --newton_muon_lr_factor "${NEWTON_LR}"
  --newton_muon_ewma_beta 0.95
  --newton_muon_ridge "${RIDGE}"
  --newton_muon_refresh_interval 32
  --newton_muon_max_precond_dim 1024
  --newton_muon_block_size 512
  --newton_muon_precond_clip "${PRECOND_CLIP}"
  --newton_muon_precond_log_interval 200
)

echo "[GPT-NM-STAB] running: ${cmd[*]}"
"${cmd[@]}"

echo "[GPT-NM-STAB] done"
