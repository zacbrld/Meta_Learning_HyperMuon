import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


COLORS = {
    "adamw": "#3B6EA8",
    "muon": "#D97904",
    "newton": "#2F8F5B",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.5,
            "axes.labelsize": 8.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.55,
            "ytick.major.width": 0.55,
            "xtick.minor.width": 0.45,
            "ytick.minor.width": 0.45,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.7,
            "ytick.minor.size": 1.7,
            "axes.grid": True,
            "grid.color": "#B8B8B8",
            "grid.alpha": 0.20,
            "grid.linewidth": 0.45,
            "lines.linewidth": 1.05,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def load_cifar(cifar_dir: Path) -> pd.DataFrame:
    paths = sorted(cifar_dir.glob("cifar10_resmlp32_*_seed0.csv"))
    if not paths:
        raise FileNotFoundError(f"No CIFAR CSVs found in {cifar_dir}")
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["source"] = path.name
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df["eval_split"] == "test"].copy()
    return df.dropna(subset=["epoch", "eval_accuracy"])


def load_gpt(gpt_curves: Path) -> pd.DataFrame:
    if not gpt_curves.exists():
        raise FileNotFoundError(f"Missing GPT curves CSV: {gpt_curves}")
    return pd.read_csv(gpt_curves)


def best_stable_newton(gpt_df: pd.DataFrame) -> str:
    stable = gpt_df[gpt_df["optimizer"].str.startswith("newton stable", na=False)]
    if stable.empty:
        raise ValueError("No stable Newton-Muon curve found.")
    finals = (
        stable.dropna(subset=["val_loss"])
        .sort_values(["optimizer", "iter"])
        .groupby("optimizer", as_index=False)
        .tail(1)
        .sort_values("val_loss")
    )
    return str(finals.iloc[0]["optimizer"])


def format_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))


def plot(cifar_df: pd.DataFrame, gpt_df: pd.DataFrame, output: Path):
    newton_name = best_stable_newton(gpt_df)
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.45))

    cifar_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        ("newton_muon", "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in cifar_specs:
        sub = cifar_df[cifar_df["optimizer"] == opt].sort_values("epoch")
        if sub.empty:
            continue
        axes[0].plot(sub["step"], 100.0 * sub["eval_accuracy"], label=label, color=color)

    axes[0].set_xlabel(r"$\mathrm{Training\ step}$")
    axes[0].set_ylabel(r"$\mathrm{Test\ accuracy}\;(\%)$")
    axes[0].set_xlim(0, 1300)
    axes[0].set_ylim(20, 70)
    format_axes(axes[0])

    gpt_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        (newton_name, "Newton-Muon", COLORS["newton"]),
    ]
    for opt, label, color in gpt_specs:
        sub = gpt_df[(gpt_df["optimizer"] == opt) & gpt_df["val_loss"].notna()].sort_values("iter")
        if sub.empty:
            continue
        sub = sub[sub["iter"] > 0]
        axes[1].plot(sub["iter"], sub["val_loss"], label=label, color=color)

    axes[1].set_xlabel(r"$\mathrm{Training\ step}$")
    axes[1].set_ylabel(r"$\mathrm{Validation\ loss}$")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(3.9, 6.3)
    format_axes(axes[1])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=2.2,
        columnspacing=1.5,
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.19, top=0.84, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper baseline figure: CIFAR-10 accuracy and GPT WikiText loss.")
    parser.add_argument("--cifar_dir", default="izar_fetch/results_cifar_fig1")
    parser.add_argument(
        "--gpt_curves",
        default="izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv",
    )
    parser.add_argument("--output", default="figures/paper_like/figure1_cifar_wikitext_baselines")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    plot(load_cifar(Path(args.cifar_dir)), load_gpt(Path(args.gpt_curves)), Path(args.output))
    print(f"Saved {Path(args.output).with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
