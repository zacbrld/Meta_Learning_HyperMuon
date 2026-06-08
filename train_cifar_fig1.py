"""
CIFAR-10 reproduction script for Figure 1 of the Newton-Muon paper.

Default setup:
  - residual MLP with 32 hidden layers, width 512
  - 100 epochs, batch size 4096
  - random crop + horizontal flip
  - warmup + cosine learning-rate schedule
  - AdamW, Muon, or Newton-Muon with the paper's final CIFAR hyperparameters
"""

import argparse
import csv
import math
import os
import random
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from models import ResidualMLP32
from optimizers import (
    GDUOAdamWOptimizer,
    GDUOMuonOptimizer,
    GDUONewtonMuonOptimizer,
    MuonOptimizer,
    NewtonMuonOptimizer,
)


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

LOG_COLUMNS = [
    "optimizer",
    "run_name",
    "seed",
    "matrix_lr_arg",
    "adamw_lr_arg",
    "step",
    "epoch",
    "train_time_sec",
    "train_loss",
    "eval_split",
    "eval_loss",
    "eval_accuracy",
    "lr",
    "base_lr",
    "lr_scale",
    "mu",
    "a",
    "b",
    "c",
    "hypgrad_lr",
    "hypgrad_lr_unclipped",
    "hypgrad_mu",
    "hypgrad_mu_unclipped",
    "hypgrad_ridge",
    "hypgrad_ridge_unclipped",
    "gduo_alignment",
    "update_rms",
    "ewma_beta",
    "ridge",
    "refresh_interval",
    "precond_refreshes",
]


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_transforms():
    from torchvision import transforms

    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, eval_transform


def build_loaders(args):
    from torchvision import datasets

    train_transform, eval_transform = build_transforms()

    full_train_aug = datasets.CIFAR10(
        args.data_root, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        args.data_root, train=False, download=True, transform=eval_transform
    )

    val_set = None
    train_set = full_train_aug
    if args.use_validation_split:
        full_train_eval = datasets.CIFAR10(
            args.data_root, train=True, download=True, transform=eval_transform
        )
        generator = torch.Generator().manual_seed(args.seed)
        indices = torch.randperm(len(full_train_aug), generator=generator).tolist()
        train_indices = indices[:45000]
        val_indices = indices[45000:]
        train_set = Subset(full_train_aug, train_indices)
        val_set = Subset(full_train_eval, val_indices)

    pin_memory = torch.cuda.is_available()
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(
            val_set,
            batch_size=args.eval_batch_size,
            shuffle=False,
            **loader_kwargs,
        )

    return train_loader, val_loader, test_loader


def default_or(value: Optional[float], default: float) -> float:
    return default if value is None else value


def build_optimizer(args, model: ResidualMLP32):
    optimizer_name = args.optimizer.lower()
    matrix_param_names = model.matrix_param_names()

    if optimizer_name == "adamw":
        lr = default_or(args.adamw_lr, 8e-4)
        wd = default_or(args.adamw_wd, 1e-2)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999)
        )
        for group in optimizer.param_groups:
            group["base_lr"] = group["lr"]
        return optimizer

    if optimizer_name == "adamw_gduo_lr":
        return GDUOAdamWOptimizer(
            model,
            lr_init=default_or(args.adamw_lr, 8e-4),
            weight_decay=default_or(args.adamw_wd, 1e-2),
            betas=(0.9, 0.999),
            hyper_lr=args.gduo_hyper_lr,
            hypergrad_clip=args.gduo_hypergrad_clip,
            lr_min_ratio=args.gduo_lr_min_ratio,
            lr_max_ratio=args.gduo_lr_max_ratio,
        )

    if optimizer_name == "muon":
        return MuonOptimizer(
            model,
            lr=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.8),
            weight_decay=default_or(args.matrix_wd, 1e-3),
            adamw_lr=default_or(args.adamw_lr, 1.6e-3),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            matrix_param_names=matrix_param_names,
        )

    if optimizer_name == "muon_gduo_lr":
        return GDUOMuonOptimizer(
            model,
            matrix_param_names=matrix_param_names,
            lr_init=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.8),
            weight_decay=default_or(args.matrix_wd, 1e-3),
            adamw_lr=default_or(args.adamw_lr, 1.6e-3),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            hyper_lr=args.gduo_hyper_lr,
            hypergrad_clip=args.gduo_hypergrad_clip,
            lr_min_ratio=args.gduo_lr_min_ratio,
            lr_max_ratio=args.gduo_lr_max_ratio,
        )

    if optimizer_name in {"muon_gduo_mu", "muon_gduo_lr_mu"}:
        return GDUOMuonOptimizer(
            model,
            matrix_param_names=matrix_param_names,
            lr_init=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.8),
            learn_lr=optimizer_name == "muon_gduo_lr_mu",
            learn_momentum=True,
            weight_decay=default_or(args.matrix_wd, 1e-3),
            adamw_lr=default_or(args.adamw_lr, 1.6e-3),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            hyper_lr=args.gduo_hyper_lr,
            hypergrad_clip=args.gduo_hypergrad_clip,
            momentum_hyper_lr=args.gduo_mu_hyper_lr,
            momentum_hypergrad_clip=args.gduo_mu_hypergrad_clip,
            lr_min_ratio=args.gduo_lr_min_ratio,
            lr_max_ratio=args.gduo_lr_max_ratio,
        )

    if optimizer_name == "newton_muon":
        return NewtonMuonOptimizer(
            model,
            matrix_param_names=matrix_param_names,
            lr=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.75),
            weight_decay=default_or(args.matrix_wd, 3e-4),
            adamw_lr=default_or(args.adamw_lr, 8e-4),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            ewma_beta=args.ewma_beta,
            ridge=args.ridge,
            refresh_interval=args.refresh_interval,
        )

    if optimizer_name == "newton_muon_gduo_lr":
        return GDUONewtonMuonOptimizer(
            model,
            matrix_param_names=matrix_param_names,
            lr_init=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.75),
            weight_decay=default_or(args.matrix_wd, 3e-4),
            adamw_lr=default_or(args.adamw_lr, 8e-4),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            ewma_beta=args.ewma_beta,
            ridge=args.ridge,
            refresh_interval=args.refresh_interval,
            hyper_lr=args.gduo_hyper_lr,
            hypergrad_clip=args.gduo_hypergrad_clip,
            lr_min_ratio=args.gduo_lr_min_ratio,
            lr_max_ratio=args.gduo_lr_max_ratio,
        )

    if optimizer_name in {
        "newton_muon_gduo_mu",
        "newton_muon_gduo_ridge",
        "newton_muon_gduo_ridge_mu",
        "newton_muon_gduo_lr_ridge_mu",
    }:
        return GDUONewtonMuonOptimizer(
            model,
            matrix_param_names=matrix_param_names,
            lr_init=default_or(args.matrix_lr, 0.16),
            momentum=default_or(args.matrix_momentum, 0.75),
            learn_lr=optimizer_name == "newton_muon_gduo_lr_ridge_mu",
            learn_momentum=optimizer_name
            in {
                "newton_muon_gduo_mu",
                "newton_muon_gduo_ridge_mu",
                "newton_muon_gduo_lr_ridge_mu",
            },
            learn_ridge=optimizer_name
            in {
                "newton_muon_gduo_ridge",
                "newton_muon_gduo_ridge_mu",
                "newton_muon_gduo_lr_ridge_mu",
            },
            weight_decay=default_or(args.matrix_wd, 3e-4),
            adamw_lr=default_or(args.adamw_lr, 8e-4),
            adamw_wd=default_or(args.adamw_wd, 1e-2),
            adamw_betas=(0.9, 0.999),
            ewma_beta=args.ewma_beta,
            ridge=args.ridge,
            refresh_interval=args.refresh_interval,
            hyper_lr=args.gduo_hyper_lr,
            hypergrad_clip=args.gduo_hypergrad_clip,
            momentum_hyper_lr=args.gduo_mu_hyper_lr,
            momentum_hypergrad_clip=args.gduo_mu_hypergrad_clip,
            ridge_hyper_lr=args.gduo_ridge_hyper_lr,
            ridge_hypergrad_clip=args.gduo_ridge_hypergrad_clip,
            lr_min_ratio=args.gduo_lr_min_ratio,
            lr_max_ratio=args.gduo_lr_max_ratio,
            ridge_min_ratio=args.gduo_ridge_min_ratio,
            ridge_max_ratio=args.gduo_ridge_max_ratio,
        )

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def lr_scale_for_step(step_index: int, total_steps: int, warmup_steps: int, min_ratio: float):
    if warmup_steps > 0 and step_index < warmup_steps:
        return (step_index + 1) / warmup_steps

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_index + 1 - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def set_optimizer_lr_scale(optimizer, scale: float):
    if hasattr(optimizer, "set_lr_scale"):
        optimizer.set_lr_scale(scale)
        return

    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * scale


def optimizer_metrics(optimizer) -> dict:
    if hasattr(optimizer, "get_metrics"):
        return optimizer.get_metrics()

    return {
        "lr": optimizer.param_groups[0]["lr"],
        "mu": float("nan"),
        "a": float("nan"),
        "b": float("nan"),
        "c": float("nan"),
        "update_rms": float("nan"),
        "ewma_beta": float("nan"),
        "ridge": float("nan"),
        "refresh_interval": float("nan"),
        "precond_refreshes": float("nan"),
    }


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0.0
    correct = 0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += y.numel()
    model.train()
    return total_loss / n, correct / n


def open_logger(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w", newline="")
    writer = csv.DictWriter(handle, fieldnames=LOG_COLUMNS)
    writer.writeheader()
    return handle, writer


def log_eval_row(
    writer,
    args,
    step: int,
    epoch: float,
    train_time_sec: float,
    train_loss: float,
    eval_split: str,
    eval_loss: float,
    eval_accuracy: float,
    metrics: dict,
):
    row = {column: float("nan") for column in LOG_COLUMNS}
    row.update(
        {
            "optimizer": args.optimizer,
            "run_name": args.run_name or "",
            "seed": args.seed,
            "matrix_lr_arg": args.matrix_lr if args.matrix_lr is not None else float("nan"),
            "adamw_lr_arg": args.adamw_lr if args.adamw_lr is not None else float("nan"),
            "step": step,
            "epoch": epoch,
            "train_time_sec": train_time_sec,
            "train_loss": train_loss,
            "eval_split": eval_split,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
        }
    )
    for key in (
        "lr",
        "base_lr",
        "lr_scale",
        "mu",
        "a",
        "b",
        "c",
        "hypgrad_lr",
        "hypgrad_lr_unclipped",
        "hypgrad_mu",
        "hypgrad_mu_unclipped",
        "hypgrad_ridge",
        "hypgrad_ridge_unclipped",
        "gduo_alignment",
        "update_rms",
        "ewma_beta",
        "ridge",
        "refresh_interval",
        "precond_refreshes",
    ):
        if key in metrics:
            row[key] = metrics[key]
    writer.writerow(row)


def train(args):
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, test_loader = build_loaders(args)

    if args.eval_split == "val":
        if val_loader is None:
            raise ValueError("--eval_split val requires --use_validation_split")
        eval_loader = val_loader
    else:
        eval_loader = test_loader

    model = ResidualMLP32(width=args.width, depth=args.depth).to(device)
    optimizer = build_optimizer(args, model)
    criterion = nn.CrossEntropyLoss()

    total_steps = args.epochs * len(train_loader)
    os.makedirs(args.results_dir, exist_ok=True)
    run_name = args.run_name or f"cifar10_resmlp{args.depth}_{args.optimizer}_seed{args.seed}"
    run_name = run_name.replace("/", "_")
    log_path = os.path.join(
        args.results_dir,
        f"{run_name}.csv",
    )
    log_handle, writer = open_logger(log_path)

    print(f"[CIFAR-Fig1] device={device}")
    print(f"[CIFAR-Fig1] optimizer={args.optimizer}")
    print(f"[CIFAR-Fig1] epochs={args.epochs} batch_size={args.batch_size}")
    print(f"[CIFAR-Fig1] steps_per_epoch={len(train_loader)} total_steps={total_steps}")
    print(f"[CIFAR-Fig1] eval_split={args.eval_split} eval_interval={args.eval_interval}")
    print(f"[CIFAR-Fig1] log_path={log_path}")

    global_step = 0
    train_time_sec = 0.0
    last_loss = float("nan")

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_idx, (x, y) in enumerate(train_loader):
                scale = lr_scale_for_step(
                    global_step,
                    total_steps,
                    args.warmup_steps,
                    args.min_lr_ratio,
                )
                set_optimizer_lr_scale(optimizer, scale)

                if device.type == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                if hasattr(optimizer, "get_metrics"):
                    optimizer.step(model)
                else:
                    optimizer.step()

                if device.type == "cuda":
                    torch.cuda.synchronize()
                train_time_sec += time.perf_counter() - start

                last_loss = loss.item()
                global_step += 1

                should_eval = global_step % args.eval_interval == 0
                is_final_step = global_step == total_steps
                if should_eval or is_final_step:
                    eval_loss, eval_acc = evaluate(model, eval_loader, device)
                    metrics = optimizer_metrics(optimizer)
                    epoch_float = epoch - 1 + (batch_idx + 1) / len(train_loader)
                    log_eval_row(
                        writer,
                        args,
                        global_step,
                        epoch_float,
                        train_time_sec,
                        last_loss,
                        args.eval_split,
                        eval_loss,
                        eval_acc,
                        metrics,
                    )
                    log_handle.flush()
                    print(
                        "[CIFAR-Fig1] "
                        f"step={global_step:04d}/{total_steps} "
                        f"epoch={epoch_float:.2f} "
                        f"train_loss={last_loss:.4f} "
                        f"{args.eval_split}_acc={eval_acc:.4f} "
                        f"train_time={train_time_sec:.1f}s "
                        f"lr={metrics['lr']:.3e} "
                        f"base_lr={metrics.get('base_lr', float('nan')):.3e}"
                    )
    finally:
        log_handle.close()

    print(f"[CIFAR-Fig1] done: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce CIFAR-10 Figure 1 curves.")
    parser.add_argument(
        "--optimizer",
        choices=[
            "adamw",
            "muon",
            "newton_muon",
            "adamw_gduo_lr",
            "muon_gduo_lr",
            "newton_muon_gduo_lr",
            "muon_gduo_mu",
            "muon_gduo_lr_mu",
            "newton_muon_gduo_mu",
            "newton_muon_gduo_ridge",
            "newton_muon_gduo_ridge_mu",
            "newton_muon_gduo_lr_ridge_mu",
        ],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--eval_batch_size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--results_dir", type=str, default="results_cifar_fig1")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_interval", type=int, default=24)
    parser.add_argument("--eval_split", choices=["test", "val"], default="test")
    parser.add_argument("--use_validation_split", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--adamw_lr", type=float, default=None)
    parser.add_argument("--adamw_wd", type=float, default=None)
    parser.add_argument("--matrix_lr", type=float, default=None)
    parser.add_argument("--matrix_wd", type=float, default=None)
    parser.add_argument("--matrix_momentum", type=float, default=None)
    parser.add_argument("--ewma_beta", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--refresh_interval", type=int, default=16)
    parser.add_argument("--gduo_hyper_lr", type=float, default=1e-3)
    parser.add_argument("--gduo_hypergrad_clip", type=float, default=1.0)
    parser.add_argument("--gduo_mu_hyper_lr", type=float, default=None)
    parser.add_argument("--gduo_mu_hypergrad_clip", type=float, default=None)
    parser.add_argument("--gduo_ridge_hyper_lr", type=float, default=None)
    parser.add_argument("--gduo_ridge_hypergrad_clip", type=float, default=None)
    parser.add_argument("--gduo_lr_min_ratio", type=float, default=0.05)
    parser.add_argument("--gduo_lr_max_ratio", type=float, default=5.0)
    parser.add_argument("--gduo_ridge_min_ratio", type=float, default=0.05)
    parser.add_argument("--gduo_ridge_max_ratio", type=float, default=20.0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
