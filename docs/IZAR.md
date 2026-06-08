# Izar Usage

The cluster copy used in the current experiments is:

```text
/home/chetaill/muon
```

Recommended sync from the local repo:

```bash
rsync -av --delete \
  --exclude '.git/' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude 'data/' \
  --exclude 'logs/' \
  --exclude 'results/' \
  --exclude 'results_*' \
  --exclude 'figures/' \
  --exclude 'izar_fetch/' \
  ./ izar:/home/chetaill/muon/
```

Fixed Newton-Muon CIFAR reproduction:

```bash
ssh izar 'cd /home/chetaill/muon && sbatch --array=0-2 run_cifar_fig1_izar.slurm'
```

GD-UO LR-only CIFAR meta-learning:

```bash
ssh izar 'cd /home/chetaill/muon && sbatch --array=0-2 run_cifar_gduo_izar.slurm'
```

Monitor:

```bash
ssh izar 'squeue -u chetaill'
```

Fetch results:

```bash
mkdir -p izar_fetch/results_cifar_fig1 izar_fetch/results_cifar_gduo_lr izar_fetch/logs
rsync -av izar:/home/chetaill/muon/results_cifar_fig1/ izar_fetch/results_cifar_fig1/
rsync -av izar:/home/chetaill/muon/results_cifar_gduo_lr/ izar_fetch/results_cifar_gduo_lr/
rsync -av 'izar:/home/chetaill/muon/logs/cifar_*' izar_fetch/logs/
```
