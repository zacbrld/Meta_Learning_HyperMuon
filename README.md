# Meta-Learning HyperMuon

This project studies whether optimizer choices can be learned during training.
We start from Muon and Newton--Muon, then learn two kinds of optimizer behavior:

- temporal dynamics: learning-rate and momentum schedules;
- geometry: how much preconditioning or curvature information should change the update direction.

The paper figures are generated from the experiment logs stored in `izar_fetch/`.
The training jobs were run on the EPFL Izar cluster. Re-running all training from
scratch is possible with the Slurm launchers, but the submission can be
reproduced locally from the fetched logs.

## Reproduce the paper artifacts

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Regenerate the figures and result tables:

```bash
python run.py
```

The command writes:

```text
figures/paper_like/figure1_cifar_wikitext_baselines.pdf
figures/paper_like/figure2_gduo_scopes.pdf
figures/paper_like/figure3_geometry_variants.pdf
figures/paper_like/figure4_muon_layerwise_lr_momentum.pdf
tables/table_cifar_baselines.csv
tables/table_cifar_baselines.tex
tables/table_wikitext_baselines.csv
tables/table_wikitext_baselines.tex
tables/table_gduo_scopes.csv
tables/table_gduo_scopes.tex
tables/table_geometry.csv
tables/table_geometry.tex
```

To regenerate only the figures:

```bash
python run.py --skip-tables
```

To regenerate only the tables:

```bash
python run.py --skip-plots
```

## Main files

```text
run.py
    Rebuilds all paper figures and tables from fetched logs.

generate_plots.py
    Rebuilds all paper figures only.

utils/plot_utils.py
    Shared plotting, log parsing, and figure generation utilities.

plot_paper_*.py
    Thin wrappers kept for regenerating one figure at a time.

train_cifar_fig1.py
    CIFAR-10 training script used for the fixed AdamW, Muon, and Newton--Muon
    reproduction.

external/llm-baselines/
    GPT/WikiText training code and optimizer implementations.

slurm/*.slurm
    Izar launchers used to run the cluster experiments.
```

## Plotting

All figure logic is centralized in `utils/plot_utils.py`. The simplest plotting
entry point is:

```bash
python generate_plots.py
```

To regenerate only selected figures:

```bash
python generate_plots.py --plots baselines
python generate_plots.py --plots gduo geometry layerwise
```

## Data layout

```text
izar_fetch/results_cifar_fig1/
    CIFAR-10 CSV logs.

izar_fetch/llm_newton_stability/diagnostic/
    GPT WikiText baseline curves.

izar_fetch/current_meta_logs/
    Main GD-UO learning-rate and momentum runs.

izar_fetch/recent_gating_precond/
    Learned geometry and preconditioner runs.

izar_fetch/layerwise_3005859/results/
    Final learned layerwise Muon hyperparameters.
```

## Methods

`AdamW`, `Muon`, and `Newton--Muon` are fixed optimizer baselines. `GD-UO`
variants learn optimizer hyperparameters online by differentiating through the
training step. For temporal learning, we learn learning-rate and momentum either
globally or per matrix. For geometry learning, Muon remains the backbone and a
learned bucket-wise gate controls how strongly a preconditioned gradient is
injected before Muon's projection.

## Cluster runs

The Slurm files reproduce the raw training jobs on Izar. The most relevant ones
for the paper are:

```text
slurm/run_cifar_fig1_izar.slurm
slurm/run_llm_newton_stability_izar.slurm
slurm/run_wikitext_51m_layerwise_izar.slurm
slurm/run_wikitext_51m_gduo_missing_izar.slurm
slurm/run_wikitext_51m_gating_izar.slurm
slurm/run_wikitext_51m_precond_gate_izar.slurm
```

The local reproduction script does not submit cluster jobs.

## Training code organization

For the report, the cleanest story is to keep two training entry points:

```text
train_cifar_fig1.py
    Small CIFAR-10 reproduction used for the fixed optimizer baselines.

external/llm-baselines/src/main.py
    GPT/WikiText training entry point used for the main paper experiments.
```

The optimizer logic belongs in `external/llm-baselines/src/optim/` for GPT runs
and in `optimizers/` for the small CIFAR reproduction. Slurm files should stay
in `slurm/`; they are launch recipes, not training code. Fetched logs and
generated figures are separated from source code in `izar_fetch/`, `figures/`,
and `tables/`.
