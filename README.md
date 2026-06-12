# Meta-Learning HyperMuon

This is the reproducibility package for our optimizer meta-learning project.

Modern optimizers contain many hand-fixed choices: learning rates, momentum,
normalization constants, and preconditioning strengths. These choices are
usually tuned offline by grid search, then frozen for the whole training run.
Our project asks whether some of them can be learned directly during training.

We study two parts of the optimizer:

- **time dynamics**: the learning rate and momentum, globally and per layer;
- **geometry**: how much curvature or preconditioning should change the update
  direction before Muon's matrix projection.

We compare fixed optimizers against meta-learned variants on CIFAR-10 and
GPT/WikiText. For the language-model experiments, we test global meta-learning
and layerwise meta-learning for AdamW, Muon, and Newton--Muon. The strongest
time-dynamics result is Muon with layerwise learned learning rate and momentum:
on GPT/WikiText, it improves the fixed Muon validation loss from **4.096** to
**3.477** after 2000 iterations. We then test learned geometry corrections on
top of Muon. The best soft preconditioner in our sweep, AdaGrad-EMA/Muon,
reaches **3.402** validation loss and **0.4005** validation accuracy.

In short: layerwise meta-learning makes Muon substantially stronger, and soft
learned geometry works better than injecting a hard Newton correction everywhere.
The report itself is in `MetaMuon.pdf`.

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
For the language-model experiments, we used and adapted
[epfml/llm-baselines](https://github.com/epfml/llm-baselines) as the GPT
training codebase.

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
    GPT/WikiText training code adapted from epfml/llm-baselines.

figures/paper_like/
    Generated figures.
```
