# Meta-Learning HyperMuon

**Learning the Dynamics and Geometry of Matrix Optimizers via Hypergradients**
Zacharie Bourlard, Thomas Chetaille, Marius Lhôte — EPFL, *Optimization for
Machine Learning* mini-project, Spring 2026.

Final report: [`Report_OML.pdf`](Report_OML.pdf).

This repository studies whether the **hypergradient** framework of
[Chandra et al., 2022 (GD-UO)](https://proceedings.neurips.cc/paper_files/paper/2022/hash/36ce475705c1dc6c50a5956cedff3d01-Abstract-Conference.html)
can make Muon-style matrix optimizers adapt *online* along two axes:

1. **Dynamics** — per-layer learning rate and momentum (how far / how much
   inertia per step);
2. **Geometry** — *how much* input-side curvature to inject around Muon's
   orthogonalized update, via a learned residual gate.

We evaluate on two settings: **CIFAR-10** with a 32-layer residual MLP, and a
**51M-parameter GPT** trained on **WikiText**.

---

## Quick start

```bash
pip install -r requirements.txt                          # core: CIFAR + plotting
pip install -r external/llm-baselines/requirements.txt   # extra: GPT / WikiText path
python run.py --list                   # see every stage -> report deliverable

# CIFAR-10 baselines (runs on CPU or a single GPU):
python run.py cifar-baselines          # full (100 epochs)
python run.py cifar-baselines --quick  # 2-epoch smoke test

# Inspect the GPU (WikiText) pipeline without launching it:
python run.py wikitext-dynamics --dry-run

# Rebuild the paper figures from available CSVs:
python run.py figures
```

`run.py` is the single reproducibility entry point. It does not re-implement
anything — it calls the existing training scripts and plotting scripts with the
exact hyper-parameters used in the report. Every command is printed before it
runs; `--dry-run` prints them without executing.

---

## Pipeline: stage → report deliverable

| `run.py` stage       | Reproduces                                              | Compute     |
|----------------------|---------------------------------------------------------|-------------|
| `cifar-baselines`    | Table 1 (CIFAR columns) + Fig 1 (top)                   | CPU / 1 GPU |
| `cifar-meta`         | CIFAR GD-UO LR sanity runs (local appendix)             | CPU / 1 GPU |
| `wikitext-baselines` | Table 1 (WikiText columns) + Fig 1 (bottom)             | 1 GPU       |
| `wikitext-dynamics`  | Fig 3 + Fig 4 (LR + momentum, `global` & layerwise)     | 1 GPU       |
| `wikitext-geometry`  | Table 2 (Adam/Muon, Muon/Newton, AdaGrad-EMA, SOAP-lite)| 1 GPU       |
| `figures`            | All paper figures (from CSVs)                           | CPU         |
| `all`                | Every stage above, in order                             | 1 GPU       |

The WikiText runs (8 transformer blocks, `n_embd=512`, sequence length 1024,
2000 iterations) require a CUDA GPU and were executed on the EPFL **Izar**
cluster. The `run_wikitext_*_izar.slurm` launchers are the authoritative
cluster configs; `run.py` mirrors them so the same experiments can be launched
on any single GPU.

---

## Methods and hyper-parameters

### Optimizers

- **AdamW** — coordinate-wise adaptive baseline.
- **Muon** ([Jordan et al., 2024](https://kellerjordan.github.io/posts/muon/);
  [Liu et al., 2025](https://arxiv.org/abs/2502.16982)) — Nesterov momentum
  followed by 5 Newton–Schulz iterations that orthogonalize the update
  (`(a,b,c) = (3.4445, −4.7750, 2.0315)`), equalizing the update's singular
  values.
- **Newton–Muon** ([Du & Su, 2026](https://arxiv.org/abs/2604.01472)) —
  right-preconditions the gradient by the inverse input-activation covariance
  `(K + λI)^{-1}` (EWMA-maintained via forward hooks) before the Muon step.

### GD-UO meta-learning (dynamics)

After a step `w_t = w_{t-1} − α ∇L_t`, the next loss is differentiated w.r.t.
`α` itself. We keep `α` (reparametrized as `exp(·)` for positivity) and
momentum `µ` (as `σ(·)` for the `(0,1)` range) attached to the graph, detach
the weight gradients (to avoid second-order terms) and previous weights, and
let `backward()` deposit the hypergradient. Two granularities:

- **global** (`--gduo_scope global`): one LR/momentum shared by all layers;
- **layerwise** (`--gduo_scope tensor`, default): each parameter bucket
  (q/k/v projections, attention output, MLP fc, MLP proj) learns its own
  LR scale and momentum.

Meta hyper-learning-rates (from the layerwise launcher), `ema_beta=0.9`,
hypergrad clip `0`:

| Optimizer          | `lr_hyper_lr` | `momentum_hyper_lr` |
|--------------------|---------------|---------------------|
| `adamw-gduo`       | 100           | 5 000               |
| `muon-gduo`        | 1 000         | 100 000             |
| `newton-muon-gduo` | 3 000         | 300 000             |

### Geometry gates (curvature)

On top of the Muon layerwise LR+momentum reference, a learned gate
`g_geo = g + s_bucket·(P(g) − g)` interpolates between the raw and a
geometry-aware gradient before Muon's projection (`s_bucket=0` ⇒ plain Muon).
Four choices of `P` (Table 2), all with gate init `0.05–0.10`:

| Variant            | `--opt` (+ flags)                                              |
|--------------------|----------------------------------------------------------------|
| Adam/Muon          | `adam-muon-gate`                                               |
| Muon/Newton        | `muon-newton-gate` + `--newton_muon_precond_strength_*`        |
| AdaGrad-EMA/Muon   | `muon-precond-gate --muon_precond_kind adagrad_ema`           |
| SOAP-lite/Muon     | `muon-precond-gate --muon_precond_kind soap_lite`            |

### CIFAR-10 baseline hyper-parameters

ResMLP-32 (width 512, ≈10M params), 100 epochs, batch size 4096, random
crop + horizontal flip, 100-step warmup + cosine decay. AdamW LR `8e-4`;
Muon matrix LR `0.16`, `µ=0.8`; Newton-Muon matrix LR `0.16`, `µ=0.75`,
`γ=0.05`, `β_K=0.95`, refresh interval 16. See `train_cifar_fig1.py --help`
for the full list and defaults.

---

## Headline results (from `Report_OML.pdf`)

| Optimizer    | CIFAR-10 final acc. | WikiText val loss (2000 it) |
|--------------|---------------------|-----------------------------|
| AdamW        | 60.51%              | 4.472                       |
| Muon         | 66.24%              | 4.096                       |
| Newton–Muon  | **67.33%**          | **4.050**                   |

- **Dynamics:** Muon-GDUO **layerwise** LR+momentum reaches val loss **3.468**
  (best) at 2000 iterations — the clearest gain over fixed baselines.
- **Geometry:** soft gates beat hard Newton corrections;
  **AdaGrad-EMA/Muon** wins the sweep (val loss **3.402**), SOAP-lite close
  behind; uniform Muon/Newton is the least stable.

---

## Repository layout

```text
run.py                     single reproducibility entry point (this file's stages)
train_cifar_fig1.py        CIFAR-10 ResMLP-32 trainer (baselines + GD-UO LR)
models/residual_mlp.py     32-layer residual MLP for CIFAR
optimizers/                fixed Muon / Newton-Muon + GD-UO LR mix-ins (CIFAR)
plot_paper_*.py            figure generators for the report
run_*_izar.slurm           Izar (Slurm) launchers — authoritative cluster configs
external/llm-baselines/    GPT-51M / WikiText codebase (see citation below)
  src/optim/gduo_meta.py     GD-UO LR + momentum (global / layerwise)
  src/optim/gating.py        residual geometry gate
  src/optim/muon*.py         Muon / Newton-Muon / precond gates
docs/EXPERIMENTS.md        detailed experiment notes
docs/IZAR.md               cluster sync / fetch commands
```

Generated artifacts (`data/`, `results_*/`, `figures/`, `izar_fetch/`, `*.csv`,
checkpoints) are git-ignored — regenerate them with `run.py`.

---

## External libraries and citations

- **LLM training codebase:** [`epfml/llm-baselines`](https://github.com/epfml/llm-baselines),
  vendored under `external/llm-baselines/` (see its `LICENSE`); our optimizer
  mix-ins live in `external/llm-baselines/src/optim/`.
- **PyTorch / torchvision** for models, autograd and CIFAR-10 data.
- Papers: Chandra et al. (GD-UO, NeurIPS 2022); Jordan et al. (Muon, 2024);
  Liu et al. (Muon is scalable, 2025); Du & Su (Newton-Muon, 2026). Full
  references in `Report_OML.pdf`.

## Reproducibility notes

- Default seed `0`; CIFAR is deterministic on a fixed device.
- `--quick` reduces CIFAR to 2 epochs and WikiText to 200 iterations for a fast
  end-to-end smoke test.
- The `figures` stage regenerates a figure only when its input CSV is present
  locally (CIFAR CSVs from a local run, or cluster CSVs fetched into
  `izar_fetch/`); missing inputs are skipped with a warning.
