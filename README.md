# Meta-Learning HyperMuon

This repository contains the code used for our optimizer meta-learning report.
The goal is to test whether optimizer choices can be learned during training:

- temporal dynamics: learning rate and momentum;
- geometry: how much preconditioning or curvature information should change the update direction.

The repository is intentionally small. It keeps only the training code, the
plotting code, and the compact CSV files needed to regenerate the paper figures.

## Reproduce the figures

Install the dependencies:

```bash
pip install -r requirements.txt
```

Generate every figure:

```bash
python run.py
```

or equivalently:

```bash
python generate_plots.py
```

Generate only selected figures:

```bash
python generate_plots.py --plots baselines
python generate_plots.py --plots gduo geometry layerwise
```

The generated files are written to `figures/paper_like/`:

```text
figure1_cifar_wikitext_baselines.pdf
figure2_gduo_scopes.pdf
figure3_geometry_variants.pdf
figure4_muon_layerwise_lr_momentum.pdf
```

## Repository layout

```text
run.py
    Minimal entry point for reproducing the plots.

generate_plots.py
    Selects which paper figures to regenerate.

utils/plot_utils.py
    Shared plotting code. It reads the compact CSV files in data/plot_inputs/.

data/plot_inputs/
    Small processed CSV files used by the plots. These replace the raw cluster
    logs and keep the repository lightweight.

train_cifar_fig1.py
    CIFAR-10 training script for the fixed AdamW, Muon, and Newton--Muon
    baselines, plus the CIFAR GD-UO variants.

models/
    CIFAR residual MLP model used by train_cifar_fig1.py.

optimizers/
    CIFAR optimizer implementations used by train_cifar_fig1.py.

external/llm-baselines/
    GPT/WikiText training pipeline and optimizer implementations used for the
    main language-model experiments.

figures/paper_like/
    Generated paper figures.
```

## Training

The paper uses two training pipelines.

For the CIFAR-10 reproduction:

```bash
python train_cifar_fig1.py --optimizer adamw
python train_cifar_fig1.py --optimizer muon
python train_cifar_fig1.py --optimizer newton_muon
```

For GPT/WikiText, use the language-model training entry point:

```bash
cd external/llm-baselines
python src/main.py ...
```

The exact large-scale runs were executed on a cluster, but the repository does
not include cluster launchers or raw logs. The plots are reproduced from the
processed CSV files in `data/plot_inputs/`.

## What the folders mean

`data/plot_inputs/` contains the numerical results used for the figures.
`utils/` contains reusable plotting utilities.
`models/` and `optimizers/` support the CIFAR experiments.
`external/llm-baselines/` contains the GPT/WikiText experiment code.
`figures/` contains generated outputs.

No Slurm scripts, shell sync scripts, raw Izar logs, or table exports are needed
for this minimal reproducibility package.
