# Meta Learning HyperMuon

This repository studies Muon-style optimizers on CIFAR-10, with two goals:

1. reproduce the CIFAR-10 part of Figure 1 from the Newton-Muon paper;
2. test meta-learning of optimizer hyperparameters, starting with learning-rate
   adaptation.

The old proxy-based HyperMuon experiments are still kept, but the current main
path is:

```text
fixed AdamW / Muon / Newton-Muon baselines
        ->
LR-only GD-UO meta-learning
        ->
momentum and Newton-Schulz hyperparameter learning
```

Detailed experiment notes and commands are in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
Cluster commands are in [docs/IZAR.md](docs/IZAR.md).

## Repository Layout

```text
models/
  mlp.py                 legacy CIFAR MLP
  resnet.py              legacy CIFAR ResNet-20
  residual_mlp.py        32-layer residual MLP for Newton-Muon CIFAR

optimizers/
  adamw.py               legacy AdamW wrapper
  sgd.py                 legacy SGD wrapper
  muon.py                fixed Muon
  newton_muon.py         fixed Newton-Muon
  hyperadam.py           legacy proxy HyperAdam
  hypermuon.py           legacy proxy HyperMuon L1/L2/L3
  gduo_lr.py             LR-only GD-UO AdamW/Muon/Newton-Muon

train.py                 legacy proxy experiments on MLP/ResNet
plot.py                  legacy plots
run_hypermuon_izar.slurm legacy Izar launcher

train_cifar_fig1.py      CIFAR Figure 1 and GD-UO training script
plot_cifar_fig1.py       accuracy vs step/time plots
run_cifar_fig1_izar.slurm fixed AdamW/Muon/Newton-Muon array
run_cifar_gduo_izar.slurm LR-only GD-UO array
```

Generated data, logs, figures, and fetched Izar outputs are ignored by git.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch uses CUDA automatically when a GPU is available.

## Current Fixed Baseline

Reproduce the CIFAR-10 panel of Figure 1:

```bash
python train_cifar_fig1.py --optimizer adamw
python train_cifar_fig1.py --optimizer muon
python train_cifar_fig1.py --optimizer newton_muon
```

Plot:

```bash
python plot_cifar_fig1.py \
  --results_dir results_cifar_fig1 \
  --output figures/cifar_fig1_repro.png
```

## Current Meta-Learning Track

Run LR-only GD-UO variants:

```bash
python train_cifar_fig1.py --optimizer adamw_gduo_lr --min_lr_ratio 1.0
python train_cifar_fig1.py --optimizer muon_gduo_lr --min_lr_ratio 1.0
python train_cifar_fig1.py --optimizer newton_muon_gduo_lr --min_lr_ratio 1.0
```

For Muon and Newton-Muon this learns only the matrix learning rate. Momentum,
Newton-Schulz coefficients, ridge, EWMA beta, and refresh interval are fixed for
now.

## Legacy Proxy Track

The original HyperMuon experiments are still available:

```bash
python train.py --model mlp --optimizer hypermuon_l3 --seed 0
python train.py --model resnet --optimizer hypermuon_l3 --seed 0
```

This track uses the local proxy:

```text
proxy = - sum_W <grad_W, update_W(theta)>
```

It is useful as a baseline, but it is not the current main meta-learning method.
