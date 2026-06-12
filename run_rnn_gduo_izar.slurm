#!/bin/bash
#SBATCH --job-name=rnn_gduo
#SBATCH --output=/home/chetaill/muon/logs/rnn_gduo_%A_%a.out
#SBATCH --error=/home/chetaill/muon/logs/rnn_gduo_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --chdir=/home/chetaill/muon

set -euo pipefail

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

module purge
module load gcc cuda

source ~/miniforge3/bin/activate
conda activate "${CONDA_ENV:-torch_env}"

mkdir -p /home/chetaill/muon/logs /home/chetaill/muon/results_rnn_gduo /scratch/izar/chetaill/Muon/rnn_data

case "${SLURM_ARRAY_TASK_ID}" in
  0) OPT="adam" ;;
  1) OPT="hyperadam_global" ;;
  2) OPT="hyperadam_groupwise" ;;
  3) OPT="hyperadam_layerwise" ;;
  *) echo "Unknown task"; exit 1 ;;
esac

cmd=(
  python -u train_rnn_gduo.py
  --optimizer "${OPT}"
  --dataset tinyshakespeare
  --data_dir /scratch/izar/chetaill/Muon/rnn_data
  --results_dir /home/chetaill/muon/results_rnn_gduo
  --device cuda
  --seed 0
  --steps 2000
  --eval_interval 100
  --eval_batches 20
  --batch_size 64
  --seq_len 128
  --emb_dim 128
  --hidden_dim 256
  --num_layers 1
  --lr 0.001
  --beta1 0.9
  --beta2 0.999
  --grad_clip 1.0
  --lr_hyper_lr 1.0
  --beta1_hyper_lr 10.0
  --ema_beta 0.9
  --min_lr_ratio 0.25
  --max_lr_ratio 4.0
)

echo "[RNN_GDUO] running: ${cmd[*]}"
"${cmd[@]}"
echo "[RNN_GDUO] done"
