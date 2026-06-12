import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


GDUO_RE = re.compile(
    r"\[GDUO-Muon\] step=(\d+) "
    r"lr_scale_avg=([0-9.eE+-]+) \(min=([0-9.eE+-]+), max=([0-9.eE+-]+)\) "
    r"lr_bound_frac=([0-9.eE+-]+) "
    r"momentum_avg=([0-9.eE+-]+) \(min=([0-9.eE+-]+), max=([0-9.eE+-]+)\)"
)

COMPONENTS = ["attn_qkv", "attn_proj", "mlp_fc", "mlp_proj", "pos_emb"]
LABELS = {
    "attn_qkv": "attn-qkv",
    "attn_proj": "attn-proj",
    "mlp_fc": "mlp-fc",
    "mlp_proj": "mlp-proj",
    "pos_emb": "pos-emb",
}
COLORS = {
    "attn_qkv": "#3B6EA8",
    "attn_proj": "#D97904",
    "mlp_fc": "#2F8F5B",
    "mlp_proj": "#8B5FBF",
    "pos_emb": "#6E6E6E",
}
MARKERS = {
    "attn_qkv": "o",
    "attn_proj": "s",
    "mlp_fc": "^",
    "mlp_proj": "D",
    "pos_emb": "x",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 7.6,
            "axes.labelsize": 7.9,
            "legend.fontsize": 6.4,
            "xtick.labelsize": 7.1,
            "ytick.labelsize": 7.1,
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
            "lines.linewidth": 0.9,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def parse_aggregate_log(path: Path) -> pd.DataFrame:
    rows = []
    text = path.read_text(errors="replace")
    for match in GDUO_RE.finditer(text):
        rows.append(
            {
                "step": int(match.group(1)),
                "lr_avg": float(match.group(2)),
                "lr_min": float(match.group(3)),
                "lr_max": float(match.group(4)),
                "lr_bound_frac": float(match.group(5)),
                "momentum_avg": float(match.group(6)),
                "momentum_min": float(match.group(7)),
                "momentum_max": float(match.group(8)),
            }
        )
    if not rows:
        raise ValueError(f"No GDUO-Muon lines found in {path}")
    return pd.DataFrame(rows)


def prepare_final(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["use_muon"].astype(bool)].copy()
    df = df[df["component"].isin(COMPONENTS)].copy()
    df["layer_plot"] = df["layer"].astype(float)
    df.loc[df["component"] == "pos_emb", "layer_plot"] = -0.55
    return df


def style_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def plot(final: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05), sharex=True)
    ax_lr_final, ax_mu_final = axes

    handles = []
    labels = []
    for component in [c for c in COMPONENTS if c != "pos_emb"]:
        sub = final[final["component"] == component].sort_values("layer_plot")
        if sub.empty:
            continue
        h = ax_lr_final.plot(
            sub["layer_plot"],
            sub["lr_scale"],
            color=COLORS[component],
            marker=MARKERS[component],
            markersize=3.2,
            linewidth=0.8,
            label=LABELS[component],
        )[0]
        ax_mu_final.plot(
            sub["layer_plot"],
            sub["momentum"],
            color=COLORS[component],
            marker=MARKERS[component],
            markersize=3.2,
            linewidth=0.8,
        )
        handles.append(h)
        labels.append(LABELS[component])

    for ax in (ax_lr_final, ax_mu_final):
        ax.set_xlim(-0.9, 7.35)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        style_axis(ax)

    ax_lr_final.axhline(1.0, color="#888888", linewidth=0.55, linestyle=":")
    ax_lr_final.set_ylabel(r"$\mathrm{Final\ LR\ scale}$")
    ax_lr_final.set_ylim(0.20, 1.05)

    ax_mu_final.axhline(0.95, color="#888888", linewidth=0.55, linestyle=":")
    ax_mu_final.set_ylabel(r"$\mathrm{Final\ momentum}$")
    ax_mu_final.set_ylim(0.94, 0.976)
    fig.supxlabel(r"$\mathrm{Transformer\ block}$", y=0.008)

    pos_emb = final[final["component"] == "pos_emb"]
    if not pos_emb.empty:
        pos = pos_emb.iloc[0]
        fig.text(
            0.745,
            0.962,
            f"pos-emb: {pos['lr_scale']:.2f}/{pos['momentum']:.2f}",
            ha="center",
            va="center",
            fontsize=6.4,
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "#8A8A8A",
                "linewidth": 0.45,
            },
        )

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        handlelength=1.4,
        columnspacing=0.75,
        bbox_to_anchor=(0.34, 1.01),
    )
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.125, top=0.86, hspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot final Muon layerwise GD-UO LR/momentum values.")
    parser.add_argument(
        "--final_csv",
        default="izar_fetch/layerwise_3005859/results/gduo_layerwise_final.csv",
    )
    parser.add_argument(
        "--output",
        default="figures/paper_like/figure4_muon_layerwise_lr_momentum",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    final = prepare_final(Path(args.final_csv))
    output = Path(args.output)
    plot(final, output)
    final.to_csv(output.with_suffix(".final.csv"), index=False)
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {output.with_suffix('.final.csv')}")


if __name__ == "__main__":
    main()
