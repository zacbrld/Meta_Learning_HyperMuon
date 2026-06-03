"""
Main training script for HyperMuon experiments.

Usage:
  python train.py --model mlp --optimizer hypermuon_l3 --seed 0
"""

import argparse
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MLP, ResNet20
from optimizers import (
    SGDOptimizer, AdamWOptimizer, MuonOptimizer,
    HyperAdamOptimizer, HyperMuonOptimizer,
)
from utils import get_cifar10_loaders, CSVLogger


# ---------------------------------------------------------------------------
# Seed helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    """Returns (loss, accuracy)."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += y.size(0)
    model.train()
    return total_loss / n, correct / n


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(name: str, model: nn.Module):
    """Return (optimizer, is_hyper) where is_hyper signals proxy-based step."""
    name = name.lower()

    if name == "sgd":
        return SGDOptimizer(model, lr=0.1, momentum=0.9), False

    if name == "adamw":
        return AdamWOptimizer(model, lr=1e-3, weight_decay=0.01), False

    if name == "hyperadam":
        return HyperAdamOptimizer(model, lr_init=1e-3, kappa=1e-5), True

    if name == "muon":
        return MuonOptimizer(
            model,
            lr=1e-3, momentum=0.95, weight_decay=0.1,
            ns_a=3.4445, ns_b=-4.7750, ns_c=2.0315,
        ), False

    if name == "hypermuon_l1":
        return HyperMuonOptimizer(model, level=1), True

    if name == "hypermuon_l2":
        return HyperMuonOptimizer(model, level=2), True

    if name == "hypermuon_l3":
        return HyperMuonOptimizer(model, level=3), True

    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    augment = (args.model == "resnet")
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size,
        seed=args.seed,
        augment_train=augment,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.model == "mlp":
        model = MLP().to(device)
    elif args.model == "resnet":
        model = ResNet20().to(device)
    else:
        raise ValueError(f"Unknown model: {args.model!r}")

    # ── Optimizer ─────────────────────────────────────────────────────────
    optimizer, is_hyper = build_optimizer(args.optimizer, model)

    # Cosine annealing scheduler for non-hyper optimizers (matches GD-UO)
    scheduler = None
    if not is_hyper:
        if args.optimizer == "sgd":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer._optim, T_max=args.epochs
            )
        elif args.optimizer == "adamw":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer._optim, T_max=args.epochs
            )
        # Muon: no scheduler (lr is fixed in spec)

    criterion = nn.CrossEntropyLoss()

    # ── Logger ────────────────────────────────────────────────────────────
    os.makedirs(args.results_dir, exist_ok=True)
    log_path = os.path.join(
        args.results_dir,
        f"{args.model}_{args.optimizer}_seed{args.seed}.csv",
    )
    logger = CSVLogger(log_path)

    # ── Training ──────────────────────────────────────────────────────────
    global_step = 0
    n_batches = len(train_loader)
    print(f"Training {args.model} | {args.optimizer} | seed {args.seed} | {args.epochs} epochs | {n_batches} steps/epoch")
    print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_rms_vals = []
        epoch_loss_sum = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step(model)

            metrics = optimizer.get_metrics()
            epoch_rms_vals.append(metrics.get("update_rms", float("nan")))
            epoch_loss_sum += loss.item()

            logger.log_step(global_step, epoch, loss.item(), metrics)
            global_step += 1

            # Progress bar: overwrite same line every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                avg_loss = epoch_loss_sum / (batch_idx + 1)
                pct = (batch_idx + 1) / n_batches
                bar_len = 20
                filled = int(bar_len * pct)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r  Epoch {epoch:3d}/{args.epochs}  [{bar}] "
                    f"{batch_idx+1:4d}/{n_batches}  "
                    f"loss={avg_loss:.4f}  lr={metrics['lr']:.2e}",
                    end="", flush=True,
                )

        # ── End of epoch: validation ───────────────────────────────────────
        val_loss, val_acc = evaluate(model, val_loader, device)
        metrics = optimizer.get_metrics()

        valid_rms = [v for v in epoch_rms_vals if not math.isnan(v)]
        if valid_rms:
            metrics["update_rms"] = sum(valid_rms) / len(valid_rms)

        logger.log_epoch(global_step, epoch, val_loss, val_acc, metrics)
        logger.flush()

        if scheduler is not None:
            scheduler.step()
            optimizer._lr = optimizer._optim.param_groups[0]["lr"]

        # End-of-epoch summary (overwrite the progress bar line)
        avg_train_loss = epoch_loss_sum / n_batches
        print(
            f"\r  Epoch {epoch:3d}/{args.epochs}  "
            f"train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}  "
            f"lr={metrics['lr']:.2e}"
        )

    # ── Final test evaluation ──────────────────────────────────────────────
    test_loss, test_acc = evaluate(model, test_loader, device)
    metrics = optimizer.get_metrics()
    logger.log_epoch(global_step, args.epochs, float("nan"), float("nan"),
                     metrics, test_accuracy=test_acc)
    logger.close()

    print(f"\nFinal test accuracy: {test_acc:.4f}")
    print(f"Results saved to {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="HyperMuon training")
    p.add_argument("--model", choices=["mlp", "resnet"], default="mlp")
    p.add_argument(
        "--optimizer",
        choices=["sgd", "adamw", "hyperadam", "muon",
                 "hypermuon_l1", "hypermuon_l2", "hypermuon_l3"],
        default="muon",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=None,
                   help="defaults: 100 for mlp, 200 for resnet")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--results_dir", type=str, default="results/")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.epochs is None:
        args.epochs = 100 if args.model == "mlp" else 200
    train(args)
