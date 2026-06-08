import argparse
import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "adamw": "AdamW",
    "muon": "Muon",
    "newton_muon": "Newton-Muon",
    "adamw_gduo_lr": "AdamW GD-UO LR",
    "muon_gduo_lr": "Muon GD-UO LR",
    "newton_muon_gduo_lr": "Newton-Muon GD-UO LR",
}

COLORS = {
    "adamw": "#4C78A8",
    "muon": "#F58518",
    "newton_muon": "#54A24B",
    "adamw_gduo_lr": "#9ecae9",
    "muon_gduo_lr": "#ffbf79",
    "newton_muon_gduo_lr": "#98df8a",
}

ORDER = [
    "adamw",
    "muon",
    "newton_muon",
    "adamw_gduo_lr",
    "muon_gduo_lr",
    "newton_muon_gduo_lr",
]


def load_results(results_dir: str, eval_split: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(results_dir, "cifar10_resmlp*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CIFAR Figure 1 CSV files found in {results_dir}")

    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = os.path.basename(path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df = df[df["eval_split"] == eval_split].copy()
    df = df.dropna(subset=["eval_accuracy", "step", "train_time_sec"])
    return df


def plot_curves(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for optimizer in ORDER:
        opt_df = df[df["optimizer"] == optimizer]
        if opt_df.empty:
            continue

        for seed, seed_df in opt_df.groupby("seed"):
            seed_df = seed_df.sort_values("step")
            label = LABELS[optimizer] if seed == opt_df["seed"].min() else None
            alpha = 0.85 if opt_df["seed"].nunique() == 1 else 0.45
            axes[0].plot(
                seed_df["step"],
                seed_df["eval_accuracy"],
                color=COLORS[optimizer],
                alpha=alpha,
                linewidth=2.0,
                label=label,
            )
            axes[1].plot(
                seed_df["train_time_sec"],
                seed_df["eval_accuracy"],
                color=COLORS[optimizer],
                alpha=alpha,
                linewidth=2.0,
                label=label,
            )

    axes[0].set_title("CIFAR-10 Accuracy vs Step")
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("test accuracy")
    axes[0].grid(alpha=0.25)

    axes[1].set_title("CIFAR-10 Accuracy vs Training Time")
    axes[1].set_xlabel("training time (seconds)")
    axes[1].set_ylabel("test accuracy")
    axes[1].grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot CIFAR Figure 1 reproduction curves.")
    parser.add_argument("--results_dir", type=str, default="results_cifar_fig1")
    parser.add_argument("--output", type=str, default="figures/cifar_fig1_repro.png")
    parser.add_argument("--eval_split", choices=["test", "val"], default="test")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = load_results(args.results_dir, args.eval_split)
    plot_curves(data, args.output)
    print(f"Saved {args.output}")
