"""
plot.py — Read all CSVs from results/ and generate the 6 figures from the spec.

Run: python plot.py [--results_dir results/] [--output_dir figures/]
"""

import argparse
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OPTIMIZERS = ["sgd", "adamw", "hyperadam", "muon", "hypermuon_l1", "hypermuon_l2", "hypermuon_l3"]
MODELS = ["mlp", "resnet"]
SEEDS = [0, 1, 2]

# Muon reference values
MUON_REF = {"a": 3.4445, "b": -4.7750, "c": 2.0315, "lr": 1e-3, "mu": 0.95}

COLORS = {
    "sgd": "#1f77b4",
    "adamw": "#ff7f0e",
    "hyperadam": "#2ca02c",
    "muon": "#d62728",
    "hypermuon_l1": "#9467bd",
    "hypermuon_l2": "#8c564b",
    "hypermuon_l3": "#e377c2",
}
LABELS = {
    "sgd": "SGD",
    "adamw": "AdamW",
    "hyperadam": "HyperAdam",
    "muon": "Muon",
    "hypermuon_l1": "HyperMuon-L1",
    "hypermuon_l2": "HyperMuon-L2",
    "hypermuon_l3": "HyperMuon-L3",
}


def load_runs(results_dir: str) -> dict:
    """
    Returns a nested dict: runs[model][optimizer][seed] = pd.DataFrame
    """
    runs = defaultdict(lambda: defaultdict(dict))
    pattern = re.compile(r"^(\w+)_(\w+)_seed(\d+)\.csv$")
    for fname in os.listdir(results_dir):
        m = pattern.match(fname)
        if m:
            model, opt, seed = m.group(1), m.group(2), int(m.group(3))
            df = pd.read_csv(os.path.join(results_dir, fname))
            runs[model][opt][seed] = df
    return runs


def mean_std(series_list):
    """Given a list of 1D arrays of possibly different lengths, interpolate to common grid."""
    # Use the shortest length as the reference
    min_len = min(len(s) for s in series_list)
    trimmed = np.array([np.array(s[:min_len], dtype=float) for s in series_list])
    return trimmed.mean(0), trimmed.std(0)


def epoch_series(df: pd.DataFrame, col: str):
    """Extract per-epoch values from a DataFrame (rows where val_loss is not NaN)."""
    epoch_df = df[df["val_loss"].notna() | df["test_accuracy"].notna()].copy()
    return epoch_df["epoch"].values, epoch_df[col].values


def step_series(df: pd.DataFrame, col: str):
    """Extract per-step values."""
    mask = df[col].notna()
    return df.loc[mask, "step"].values, df.loc[mask, col].values


# ---------------------------------------------------------------------------
# Figure 1 — Val Loss vs Steps
# ---------------------------------------------------------------------------

def fig1_val_loss_vs_steps(runs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 1 — Validation Loss vs Global Steps", fontsize=13)

    for ax, model in zip(axes, MODELS):
        for opt in OPTIMIZERS:
            if opt not in runs.get(model, {}):
                continue
            seed_data = runs[model][opt]
            all_losses = []
            for seed, df in seed_data.items():
                steps, vals = step_series(df, "val_loss")
                all_losses.append((steps, vals))
            if not all_losses:
                continue
            # Use common step grid
            min_len = min(len(v) for _, v in all_losses)
            losses = np.array([v[:min_len] for _, v in all_losses], dtype=float)
            steps_ref = all_losses[0][0][:min_len]
            mu_l = losses.mean(0)
            std_l = losses.std(0)
            color = COLORS.get(opt, None)
            ax.plot(steps_ref, mu_l, label=LABELS[opt], color=color)
            ax.fill_between(steps_ref, mu_l - std_l, mu_l + std_l, alpha=0.15, color=color)

        ax.set_title(f"Model: {model.upper()}")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Validation loss")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig1_val_loss_vs_steps.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Val Accuracy vs Epochs
# ---------------------------------------------------------------------------

def fig2_val_acc_vs_epochs(runs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 2 — Validation Accuracy vs Epochs", fontsize=13)

    for ax, model in zip(axes, MODELS):
        for opt in OPTIMIZERS:
            if opt not in runs.get(model, {}):
                continue
            seed_data = runs[model][opt]
            all_accs = []
            for seed, df in seed_data.items():
                epochs, vals = epoch_series(df, "val_accuracy")
                valid = ~np.isnan(vals.astype(float))
                all_accs.append((epochs[valid], vals[valid].astype(float)))
            if not all_accs:
                continue
            min_len = min(len(v) for _, v in all_accs)
            accs = np.array([v[:min_len] for _, v in all_accs], dtype=float)
            epochs_ref = all_accs[0][0][:min_len]
            mu_a = accs.mean(0)
            std_a = accs.std(0)
            color = COLORS.get(opt, None)
            ax.plot(epochs_ref, mu_a, label=LABELS[opt], color=color)
            ax.fill_between(epochs_ref, mu_a - std_a, mu_a + std_a, alpha=0.15, color=color)

        ax.set_title(f"Model: {model.upper()}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation accuracy")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig2_val_acc_vs_epochs.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Bar chart final test accuracy
# ---------------------------------------------------------------------------

def fig3_test_acc_bar(runs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Figure 3 — Final Test Accuracy (mean ± std, 3 seeds)", fontsize=13)

    for ax, model in zip(axes, MODELS):
        means, stds, labels, colors = [], [], [], []
        for opt in OPTIMIZERS:
            if opt not in runs.get(model, {}):
                continue
            test_accs = []
            for seed, df in runs[model][opt].items():
                # Last non-NaN test accuracy
                val = df["test_accuracy"].dropna()
                if len(val) > 0:
                    test_accs.append(float(val.iloc[-1]))
            if test_accs:
                means.append(np.mean(test_accs))
                stds.append(np.std(test_accs))
                labels.append(LABELS[opt])
                colors.append(COLORS.get(opt, "gray"))

        x = np.arange(len(means))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(f"Model: {model.upper()}")
        ax.set_ylabel("Test accuracy")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3)

        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig3_test_acc_bar.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 4 — HyperMuon-L3 hyperparameter trajectories
# ---------------------------------------------------------------------------

def fig4_hypermuon_trajectories(runs, output_dir):
    hp_cols = ["lr", "mu", "a", "b", "c"]
    hp_ref = {k: MUON_REF[k] for k in hp_cols}

    fig, axes = plt.subplots(len(hp_cols), 2, figsize=(14, 16))
    fig.suptitle("Figure 4 — HyperMuon-L3 Hyperparameter Trajectories", fontsize=13)

    for col_idx, model in enumerate(MODELS):
        opt = "hypermuon_l3"
        if opt not in runs.get(model, {}):
            for row in range(len(hp_cols)):
                axes[row][col_idx].set_title(f"{model.upper()} — no data")
            continue

        for row, hp in enumerate(hp_cols):
            ax = axes[row][col_idx]
            all_series = []
            for seed, df in runs[model][opt].items():
                # Per-epoch values
                epoch_df = df[df["val_loss"].notna()].copy()
                vals = epoch_df[hp].values.astype(float)
                epochs = epoch_df["epoch"].values
                valid = ~np.isnan(vals)
                if valid.sum() > 0:
                    all_series.append((epochs[valid], vals[valid]))

            if all_series:
                min_len = min(len(v) for _, v in all_series)
                arr = np.array([v[:min_len] for _, v in all_series], dtype=float)
                epochs_ref = all_series[0][0][:min_len]
                mu_h = arr.mean(0)
                std_h = arr.std(0)
                ax.plot(epochs_ref, mu_h, color=COLORS["hypermuon_l3"], label="learned")
                ax.fill_between(epochs_ref, mu_h - std_h, mu_h + std_h,
                                alpha=0.2, color=COLORS["hypermuon_l3"])

            # Reference line
            ax.axhline(hp_ref[hp], linestyle="--", color="black", alpha=0.5,
                       label=f"Muon ref = {hp_ref[hp]:.4f}")
            ax.set_ylabel(hp)
            ax.set_title(f"{model.upper()} — {hp}(t)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            if row == len(hp_cols) - 1:
                ax.set_xlabel("Epoch")

    plt.tight_layout()
    out = os.path.join(output_dir, "fig4_hypermuon_trajectories.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 5 — Hypergradient norms (HyperMuon-L3)
# ---------------------------------------------------------------------------

def fig5_hypgrad_norms(runs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 5 — Hypergradient Norms (HyperMuon-L3)", fontsize=13)
    hg_cols = {"hypgrad_lr": "η", "hypgrad_mu": "µ", "hypgrad_abc": "a,b,c"}

    for ax, model in zip(axes, MODELS):
        opt = "hypermuon_l3"
        if opt not in runs.get(model, {}):
            ax.set_title(f"{model.upper()} — no data")
            continue

        for hg_col, label in hg_cols.items():
            all_series = []
            for seed, df in runs[model][opt].items():
                steps, vals = step_series(df, hg_col)
                all_series.append((steps, vals))
            if not all_series:
                continue
            min_len = min(len(v) for _, v in all_series)
            arr = np.array([v[:min_len].astype(float) for _, v in all_series])
            steps_ref = all_series[0][0][:min_len]
            mu_h = np.nanmean(arr, axis=0)
            ax.plot(steps_ref, mu_h, label=f"∥∇{label}∥")

        ax.set_title(f"Model: {model.upper()}")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Hypergradient norm")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig5_hypgrad_norms.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Figure 6 — Update RMS
# ---------------------------------------------------------------------------

def fig6_update_rms(runs, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figure 6 — Update RMS (Muon & HyperMuon)", fontsize=13)
    target_opts = ["muon", "hypermuon_l1", "hypermuon_l2", "hypermuon_l3"]

    for ax, model in zip(axes, MODELS):
        for opt in target_opts:
            if opt not in runs.get(model, {}):
                continue
            all_series = []
            for seed, df in runs[model][opt].items():
                steps, vals = step_series(df, "update_rms")
                all_series.append((steps, vals))
            if not all_series:
                continue
            min_len = min(len(v) for _, v in all_series)
            arr = np.array([v[:min_len].astype(float) for _, v in all_series])
            steps_ref = all_series[0][0][:min_len]
            mu_r = np.nanmean(arr, axis=0)
            std_r = np.nanstd(arr, axis=0)
            color = COLORS.get(opt)
            ax.plot(steps_ref, mu_r, label=LABELS[opt], color=color)
            ax.fill_between(steps_ref, mu_r - std_r, mu_r + std_r, alpha=0.15, color=color)

        ax.axhline(0.2, linestyle="--", color="black", alpha=0.6, label="target = 0.2")
        ax.set_title(f"Model: {model.upper()}")
        ax.set_xlabel("Global step")
        ax.set_ylabel("Update RMS")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "fig6_update_rms.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved {out}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", default="figures/")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = load_runs(args.results_dir)

    if not runs:
        print(f"No CSV files found in {args.results_dir!r}. Run train.py first.")
        return

    fig1_val_loss_vs_steps(runs, args.output_dir)
    fig2_val_acc_vs_epochs(runs, args.output_dir)
    fig3_test_acc_bar(runs, args.output_dir)
    fig4_hypermuon_trajectories(runs, args.output_dir)
    fig5_hypgrad_norms(runs, args.output_dir)
    fig6_update_rms(runs, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir!r}")


if __name__ == "__main__":
    main()
