import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


COLORS = {
    "adamw": "#4C78A8",
    "muon": "#F58518",
    "newton": "#2CA02C",
    "unstable": "#8E8E8E",
    "red": "#D62728",
    "purple": "#9467BD",
    "brown": "#8C564B",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.3,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig, output_base: Path):
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_gpt_curves(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing GPT curves file: {path}")
    df = pd.read_csv(path)
    return df


def best_stable_newton_name(df: pd.DataFrame) -> str:
    stable = df[df["optimizer"].str.startswith("newton stable", na=False)]
    if stable.empty:
        raise ValueError("No stable Newton-Muon curve found in GPT curves CSV.")
    finals = (
        stable.dropna(subset=["val_loss"])
        .sort_values(["optimizer", "iter"])
        .groupby("optimizer")
        .tail(1)
        .sort_values("val_loss")
    )
    return str(finals.iloc[0]["optimizer"])


def plot_gpt_main(df: pd.DataFrame, output_base: Path):
    best_newton = best_stable_newton_name(df)
    series = [
        ("adamw", "AdamW", COLORS["adamw"], "-"),
        ("muon", "Muon", COLORS["muon"], "-"),
        (best_newton, "Newton-Muon", COLORS["newton"], "-"),
    ]

    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    for name, label, color, linestyle in series:
        sub = df[(df["optimizer"] == name) & df["val_loss"].notna()].sort_values("iter")
        if sub.empty:
            continue
        ax.plot(sub["iter"], sub["val_loss"], label=label, color=color, linestyle=linestyle)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation loss")
    ax.set_title("GPT on WikiText")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=3.95)
    save_figure(fig, output_base)


def plot_gpt_stability(df: pd.DataFrame, output_base: Path):
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    specs = [
        ("adamw", "AdamW", COLORS["adamw"], "-"),
        ("muon", "Muon", COLORS["muon"], "-"),
        ("newton-muon", "Newton-Muon old, NaN", COLORS["unstable"], "--"),
        ("newton stable lr0p005_ridge0p5_clip3", "Newton-Muon lr=.005 ridge=.5", COLORS["newton"], "-"),
        ("newton stable lr0p004_ridge1p0_clip3", "Newton-Muon lr=.004 ridge=1", COLORS["purple"], "-"),
        ("newton stable lr0p004_ridge0p5_clip3", "Newton-Muon lr=.004 ridge=.5", COLORS["red"], "-"),
    ]
    for name, label, color, linestyle in specs:
        sub = df[(df["optimizer"] == name) & df["val_loss"].notna()].sort_values("iter")
        if sub.empty:
            continue
        ax.plot(sub["iter"], sub["val_loss"], label=label, color=color, linestyle=linestyle)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation loss")
    ax.set_title("GPT Newton-Muon stability sweep")
    ax.legend(frameon=False, ncol=1)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=3.95)
    save_figure(fig, output_base)


def plot_gpt_perplexity(df: pd.DataFrame, output_base: Path):
    best_newton = best_stable_newton_name(df)
    series = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        (best_newton, "Newton-Muon", COLORS["newton"]),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    for name, label, color in series:
        sub = df[(df["optimizer"] == name) & df["val_pp"].notna()].sort_values("iter")
        if sub.empty:
            continue
        # Drop the iteration-0 point to keep the scale readable.
        sub = sub[sub["iter"] > 0]
        ax.plot(sub["iter"], sub["val_pp"], label=label, color=color)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation perplexity")
    ax.set_title("GPT on WikiText")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    save_figure(fig, output_base)


def load_cifar_curves(results_dir: Path) -> pd.DataFrame:
    paths = sorted(results_dir.glob("cifar10_resmlp32_*_seed0.csv"))
    if not paths:
        raise FileNotFoundError(f"Missing CIFAR result CSVs in {results_dir}")
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        df["source"] = path.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["eval_split"] == "test"].copy()
    df = df.dropna(subset=["eval_accuracy", "eval_loss", "epoch", "step"])
    return df


def plot_cifar_main(df: pd.DataFrame, output_base: Path):
    specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        ("newton_muon", "Newton-Muon", COLORS["newton"]),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.15))
    for opt, label, color in specs:
        sub = df[df["optimizer"] == opt].sort_values("epoch")
        if sub.empty:
            continue
        ax.plot(sub["epoch"], 100.0 * sub["eval_accuracy"], label=label, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("CIFAR-10 ResMLP-32")
    ax.legend(frameon=False)
    ax.set_xlim(left=0)
    save_figure(fig, output_base)


def plot_combined(cifar_df: pd.DataFrame, gpt_df: pd.DataFrame, output_base: Path):
    best_newton = best_stable_newton_name(gpt_df)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.25))

    cifar_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        ("newton_muon", "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in cifar_specs:
        sub = cifar_df[cifar_df["optimizer"] == opt].sort_values("epoch")
        if sub.empty:
            continue
        axes[0].plot(sub["epoch"], 100.0 * sub["eval_accuracy"], label=label, color=color)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test accuracy (%)")
    axes[0].set_title("(a) CIFAR-10 ResMLP-32")
    axes[0].set_xlim(left=0)

    gpt_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        (best_newton, "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in gpt_specs:
        sub = gpt_df[(gpt_df["optimizer"] == opt) & gpt_df["val_loss"].notna()].sort_values("iter")
        if sub.empty:
            continue
        axes[1].plot(sub["iter"], sub["val_loss"], label=label, color=color)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("(b) GPT WikiText")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=3.95)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save_figure(fig, output_base)


def plot_paper_figure1_layout(cifar_df: pd.DataFrame, gpt_df: pd.DataFrame, output_base: Path):
    best_newton = best_stable_newton_name(gpt_df)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.0))

    gpt_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        (best_newton, "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in gpt_specs:
        sub = gpt_df[(gpt_df["optimizer"] == opt) & gpt_df["val_loss"].notna()].sort_values("iter")
        if sub.empty:
            continue
        axes[0, 0].plot(sub["iter"], sub["val_loss"], label=label, color=color)

        timed = gpt_df[gpt_df["optimizer"] == opt].sort_values("iter").copy()
        timed["iter_dt"] = timed["iter_dt"].ffill().fillna(0.0)
        timed["wall_time_sec"] = timed["iter"].diff().fillna(0.0) * timed["iter_dt"]
        timed["wall_time_sec"] = timed["wall_time_sec"].cumsum()
        timed = timed[timed["val_loss"].notna()]
        axes[0, 1].plot(timed["wall_time_sec"], timed["val_loss"], label=label, color=color)

    axes[0, 0].set_title("GPT WikiText")
    axes[0, 0].set_xlabel("Training step")
    axes[0, 0].set_ylabel("Validation loss")
    axes[0, 0].set_xlim(left=0)
    axes[0, 0].set_ylim(bottom=3.95)

    axes[0, 1].set_title("GPT WikiText")
    axes[0, 1].set_xlabel("Wall-clock time (s)")
    axes[0, 1].set_ylabel("Validation loss")
    axes[0, 1].set_xlim(left=0)
    axes[0, 1].set_ylim(bottom=3.95)

    cifar_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        ("newton_muon", "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in cifar_specs:
        sub = cifar_df[cifar_df["optimizer"] == opt].sort_values("step")
        if sub.empty:
            continue
        acc = 100.0 * sub["eval_accuracy"]
        axes[1, 0].plot(sub["step"], acc, label=label, color=color)
        axes[1, 1].plot(sub["train_time_sec"], acc, label=label, color=color)

    axes[1, 0].set_title("CIFAR-10 ResMLP-32")
    axes[1, 0].set_xlabel("Training step")
    axes[1, 0].set_ylabel("Test accuracy (%)")
    axes[1, 0].set_xlim(left=0)

    axes[1, 1].set_title("CIFAR-10 ResMLP-32")
    axes[1, 1].set_xlabel("Wall-clock time (s)")
    axes[1, 1].set_ylabel("Test accuracy (%)")
    axes[1, 1].set_xlim(left=0)

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes.ravel()):
        ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, output_base)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-style curves from fetched CIFAR/GPT results.")
    parser.add_argument("--output_dir", default="figures/paper_like")
    parser.add_argument(
        "--gpt_curves",
        default="izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv",
    )
    parser.add_argument("--cifar_dir", default="izar_fetch/results_cifar_fig1")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    output_dir = Path(args.output_dir)
    gpt_df = load_gpt_curves(Path(args.gpt_curves))
    cifar_df = load_cifar_curves(Path(args.cifar_dir))

    plot_gpt_main(gpt_df, output_dir / "gpt_wikitext_val_loss_paper")
    plot_gpt_perplexity(gpt_df, output_dir / "gpt_wikitext_val_perplexity_paper")
    plot_gpt_stability(gpt_df, output_dir / "gpt_wikitext_newton_stability_sweep")
    plot_cifar_main(cifar_df, output_dir / "cifar10_resmlp32_accuracy_paper")
    plot_combined(cifar_df, gpt_df, output_dir / "figure1_cifar_gpt_paper")
    plot_paper_figure1_layout(cifar_df, gpt_df, output_dir / "figure1_four_panel_paper")

    print(f"Saved paper-style figures to {output_dir}")


if __name__ == "__main__":
    main()
