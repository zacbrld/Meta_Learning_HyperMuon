# Meta-Learning HyperMuon

This is the reproducibility package for our optimizer meta-learning project.

The project asks a simple question: can an optimizer learn some of its own
design choices while the model is training? Instead of fixing every coefficient
by grid search, we learn them online with hypergradients.

We study two parts of the optimizer:

- **time dynamics**: the learning rate and momentum, globally and per layer;
- **geometry**: how much curvature or preconditioning should change the update
  direction before Muon's matrix projection.

The experiments compare AdamW, Muon, Newton--Muon, learned LR/momentum variants,
and softer geometry learners such as Adam/Muon, AdaGrad-EMA/Muon, and
SOAP-lite/Muon. The report itself is in `MetaMuon.pdf`.

## Quick Start

Install the dependencies:

```bash
pip install -r requirements.txt
```

Regenerate every figure used in the report:

```bash
python generate_plots.py
```

Replay the processed training results:

```bash
python train.py --task cifar
python train.py --task gpt
```

## Figures

`generate_plots.py` reads the compact CSV files in `data/plot_inputs/` and
writes the figures to `figures/paper_like/`.

Generate selected figures only:

```bash
python generate_plots.py --plots baselines
python generate_plots.py --plots gduo geometry layerwise
```

Generated outputs:

```text
figure1_cifar_wikitext_baselines.pdf
figure2_gduo_scopes.pdf
figure3_geometry_variants.pdf
figure4_muon_layerwise_lr_momentum.pdf
```

## Training

The repository is intentionally minimal. We do not include raw cluster logs,
Slurm launchers, or old scratch scripts. The paper figures are reproduced from
processed CSV files, while the GPT/WikiText training backend is kept runnable.

Replay processed results:

```bash
python train.py --task cifar
python train.py --task gpt
```

Launch a real GPT/WikiText run:

```bash
python train.py --task gpt --optimizer muon --execute
python train.py --task gpt --optimizer adagrad-ema --execute
```

Change training settings with flags:

```bash
python train.py --task gpt --optimizer muon --execute --iterations 2000 --device cuda
```

The GPT entry point can also be called directly:

```bash
cd external/llm-baselines
python src/main.py ...
```

## Layout

```text
MetaMuon.pdf
    Final report.

generate_plots.py
    Figure entry point.

train.py
    Result replay and GPT/WikiText training launcher.

data/plot_inputs/
    Processed numerical results used by the plots.

utils/plot_utils.py
    Shared plotting code.

external/llm-baselines/
    GPT/WikiText training code and optimizer implementations.

figures/paper_like/
    Generated figures.
```

This keeps the submission small, readable, and focused on reproducing the
results used in the report.
