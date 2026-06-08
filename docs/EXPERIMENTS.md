# Experiments

This repo currently has three experiment tracks. The first one is kept as a
legacy baseline; the second and third are the current direction.

## Track A: legacy proxy HyperMuon

Status: kept for reference, not the main path for now.

Files:

```text
train.py
plot.py
run_hypermuon_izar.slurm
optimizers/hyperadam.py
optimizers/hypermuon.py
models/mlp.py
models/resnet.py
```

Idea:

```text
proxy = - sum_W <grad_W, update_W(theta)>
proxy.backward()
```

This asks whether the update produced by current hyperparameters is locally
aligned with the current gradient. It is cheap and pragmatic, but it is not the
one-step GD-UO signal.

Available optimizers:

```text
sgd
adamw
hyperadam
muon
hypermuon_l1   # learns lr
hypermuon_l2   # learns lr + momentum
hypermuon_l3   # learns lr + momentum + Newton-Schulz a,b,c
```

Example:

```bash
python train.py --model resnet --optimizer hypermuon_l3 --seed 0
```

## Track B: Newton-Muon Figure 1 CIFAR reproduction

Status: current fixed-baseline reproduction.

Files:

```text
train_cifar_fig1.py
plot_cifar_fig1.py
run_cifar_fig1_izar.slurm
models/residual_mlp.py
optimizers/muon.py
optimizers/newton_muon.py
```

Protocol:

```text
dataset: CIFAR-10
model: 32-layer residual MLP, width 512
train: 100 epochs, batch size 4096
schedule: 100-step warmup + cosine decay to 0.1x
eval: test accuracy every 24 steps
```

Fixed baselines:

```text
adamw
muon
newton_muon
```

Run locally:

```bash
python train_cifar_fig1.py --optimizer adamw
python train_cifar_fig1.py --optimizer muon
python train_cifar_fig1.py --optimizer newton_muon
```

Run on Izar:

```bash
sbatch --array=0-2 run_cifar_fig1_izar.slurm
```

Plot:

```bash
python plot_cifar_fig1.py \
  --results_dir results_cifar_fig1 \
  --output figures/cifar_fig1_repro.png
```

Latest recovered CIFAR fixed-baseline result:

```text
Newton-Muon  final 67.33%, best 67.49%
Muon         final 66.24%, best 66.39%
AdamW        final 60.51%, best 60.98%
```

## Track C: GD-UO learning-rate meta-learning

Status: current meta-learning path.

Files:

```text
train_cifar_fig1.py
run_cifar_gduo_izar.slurm
optimizers/gduo_lr.py
```

Idea:

```text
d loss_t / d lr_{t-1} = - <grad_t, update_direction_{t-1}>
```

This is the LR-only one-step signal from `Gradient Descent: The Ultimate
Optimizer`. It uses the update direction that was actually applied on the
previous step, unlike the proxy track.

Available optimizers:

```text
adamw_gduo_lr
muon_gduo_lr
newton_muon_gduo_lr
```

For Muon and Newton-Muon, only the matrix learning rate is learned. The
geometric hyperparameters stay fixed:

```text
momentum
Newton-Schulz a,b,c
Newton-Muon ridge
Newton-Muon EWMA beta
Newton-Muon refresh_interval
```

Run locally:

```bash
python train_cifar_fig1.py --optimizer adamw_gduo_lr --min_lr_ratio 1.0
python train_cifar_fig1.py --optimizer muon_gduo_lr --min_lr_ratio 1.0
python train_cifar_fig1.py --optimizer newton_muon_gduo_lr --min_lr_ratio 1.0
```

Run on Izar:

```bash
sbatch --array=0-2 run_cifar_gduo_izar.slurm
```

The GD-UO Slurm script uses warmup, then keeps the scheduler scale constant
with `--min_lr_ratio 1.0`, so the learned LR is easier to interpret.

## Next steps

Recommended order:

1. Diagnose the GD-UO LR-only runs against the fixed Figure 1 baselines.
2. If LR-only is stable, add GD-UO momentum for Muon and Newton-Muon.
3. Add GD-UO or proxy learning for Newton-Schulz `a,b,c`.
4. Treat Newton-Muon `ridge` and `ewma_beta` as later experiments; they touch
   covariance inverse logic and are more fragile.
5. Sweep `refresh_interval` by grid search, not gradient, because it is
   discrete.
