import logging
from pathlib import Path
from typing import Union

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOT_DATA = ROOT / "data" / "plot_inputs"
PathLike = Union[str, Path]

COLORS = {
    "adamw": "#3B6EA8",
    "muon": "#D97904",
    "newton": "#2F8F5B",
    "soap": "#8B5FBF",
    "black": "#111111",
}

LAYER_COMPONENTS = ["attn_qkv", "attn_proj", "mlp_fc", "mlp_proj", "pos_emb"]
LAYER_LABELS = {
    "attn_qkv": "attn-qkv",
    "attn_proj": "attn-proj",
    "mlp_fc": "mlp-fc",
    "mlp_proj": "mlp-proj",
    "pos_emb": "pos-emb",
}
LAYER_COLORS = {
    "attn_qkv": "#3B6EA8",
    "attn_proj": "#D97904",
    "mlp_fc": "#2F8F5B",
    "mlp_proj": "#8B5FBF",
    "pos_emb": "#6E6E6E",
}
LAYER_MARKERS = {
    "attn_qkv": "o",
    "attn_proj": "s",
    "mlp_fc": "^",
    "mlp_proj": "D",
    "pos_emb": "x",
}


def set_paper_style(font_size=7.8, label_size=8.2, legend_size=7.0, line_width=0.95):
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": font_size,
            "axes.labelsize": label_size,
            "legend.fontsize": legend_size,
            "xtick.labelsize": font_size - 0.5,
            "ytick.labelsize": font_size - 0.5,
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
            "lines.linewidth": line_width,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def paper_axis(ax, x_bins=5, y_bins=5, integer_x=False):
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=integer_x))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_bins))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def save_figure(fig, output):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def make_baseline_figure(
    cifar_csv: PathLike = PLOT_DATA / "cifar_baselines.csv",
    wikitext_csv: PathLike = PLOT_DATA / "wikitext_baselines.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure1_cifar_wikitext_baselines",
):
    set_paper_style(font_size=7.6, label_size=7.9, legend_size=7.0, line_width=0.95)
    cifar = pd.read_csv(cifar_csv)
    wikitext = pd.read_csv(wikitext_csv)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05))

    cifar_specs = [
        ("adamw", "AdamW", COLORS["adamw"]),
        ("muon", "Muon", COLORS["muon"]),
        ("newton_muon", "Newton-Muon", COLORS["newton"]),
    ]
    for optimizer, label, color in cifar_specs:
        sub = cifar[cifar["optimizer"] == optimizer].sort_values("step")
        axes[0].plot(sub["step"], 100.0 * sub["eval_accuracy"], label=label, color=color)

    axes[0].set_ylabel(r"$\mathrm{Test\ accuracy}\;(\%)$")
    axes[0].set_xlim(0, 1300)
    axes[0].set_ylim(20, 70)
    paper_axis(axes[0])

    gpt_specs = [
        ("AdamW", COLORS["adamw"]),
        ("Muon", COLORS["muon"]),
        ("Newton-Muon", COLORS["newton"]),
    ]
    for optimizer, color in gpt_specs:
        sub = wikitext[wikitext["optimizer"] == optimizer].sort_values("iter")
        axes[1].plot(sub["iter"], sub["val_loss"], label=optimizer, color=color)

    axes[1].set_ylabel(r"$\mathrm{Validation\ loss}$")
    axes[1].set_xlim(0, 2000)
    axes[1].set_ylim(3.9, 6.3)
    paper_axis(axes[1])
    fig.supxlabel(r"$\mathrm{Training\ step}$", y=0.008)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=0.85,
        bbox_to_anchor=(0.5, 1.025),
    )
    fig.subplots_adjust(left=0.20, right=0.99, bottom=0.125, top=0.91, hspace=0.12)
    save_figure(fig, output)
    return Path(output).with_suffix(".pdf")


def make_gduo_scope_figure(
    curves_csv: PathLike = PLOT_DATA / "gduo_scopes.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure2_gduo_scopes",
):
    set_paper_style(font_size=8.2, label_size=8.5, legend_size=7.3, line_width=1.0)
    curves = pd.read_csv(curves_csv)
    fig = plt.figure(figsize=(6.3, 4.35))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.36)
    axes = [
        fig.add_subplot(grid[0, 0:2]),
        fig.add_subplot(grid[0, 2:4]),
        fig.add_subplot(grid[1, 1:3]),
    ]
    panels = [
        ("adam", r"$\mathrm{AdamW}$"),
        ("muon", r"$\mathrm{Muon}$"),
        ("newton", r"$\mathrm{Newton{-}Muon}$"),
    ]
    variants = [
        ("baseline", "Baseline", "-", COLORS["adamw"]),
        ("global", "LR+momentum global", "--", COLORS["muon"]),
        ("layerwise", "LR+momentum layerwise", "-.", COLORS["newton"]),
    ]

    for ax, (optimizer, panel_label) in zip(axes, panels):
        panel = curves[curves["optimizer"] == optimizer]
        for variant, label, linestyle, color in variants:
            sub = panel[(panel["variant"] == variant) & (panel["label"] == label)].sort_values("iter")
            ax.plot(sub["iter"], sub["val_loss"], color=color, linestyle=linestyle, label=label)
        ax.text(0.97, 0.94, panel_label, transform=ax.transAxes, ha="right", va="top")
        ax.set_xlim(0, 2000)
        ax.set_ylim(3.35, 6.35)
        paper_axis(ax)

    fig.supxlabel(r"$\mathrm{Training\ step}$", y=0.055)
    fig.supylabel(r"$\mathrm{Validation\ loss}$", x=0.045)
    handles, labels = [], []
    for ax in axes:
        found_handles, found_labels = ax.get_legend_handles_labels()
        for handle, label in zip(found_handles, found_labels):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        handlelength=2.0,
        columnspacing=1.25,
        bbox_to_anchor=(0.5, 0.985),
    )
    fig.subplots_adjust(left=0.11, right=0.99, bottom=0.125, top=0.91)
    save_figure(fig, output)
    return Path(output).with_suffix(".pdf")


def make_geometry_variants_figure(
    curves_csv: PathLike = PLOT_DATA / "geometry_variants.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure3_geometry_variants",
):
    set_paper_style(font_size=8.0, label_size=8.5, legend_size=7.0, line_width=0.95)
    curves = pd.read_csv(curves_csv)
    colors = {
        "Muon LR+momentum layerwise": COLORS["black"],
        "Adam/Muon gate": COLORS["adamw"],
        "Muon/Newton-Muon gate": COLORS["muon"],
        "AdaGrad-EMA/Muon": COLORS["newton"],
        "SOAP-lite/Muon": COLORS["soap"],
    }
    order = list(colors)
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for label in order:
        sub = curves[curves["label"] == label].sort_values("iter")
        linewidth = 1.15 if label == "Muon LR+momentum layerwise" else 0.9
        ax.plot(sub["iter"], sub["val_loss"], color=colors[label], linewidth=linewidth, label=label)
    ax.set_xlim(0, 2000)
    ax.set_ylim(3.35, 5.95)
    ax.set_xlabel(r"$\mathrm{Training\ step}$")
    ax.set_ylabel(r"$\mathrm{Validation\ loss}$")
    paper_axis(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.15)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.185, top=0.985)
    save_figure(fig, output)
    return Path(output).with_suffix(".pdf")


def make_muon_layerwise_figure(
    final_csv: PathLike = PLOT_DATA / "muon_layerwise_final.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure4_muon_layerwise_lr_momentum",
):
    set_paper_style(font_size=7.6, label_size=7.9, legend_size=6.4, line_width=0.9)
    final = pd.read_csv(final_csv)
    final = final[final["component"].isin(LAYER_COMPONENTS)].copy()
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05), sharex=True)
    ax_lr, ax_momentum = axes
    handles = []
    labels = []

    for component in [component for component in LAYER_COMPONENTS if component != "pos_emb"]:
        sub = final[final["component"] == component].sort_values("layer_plot")
        handle = ax_lr.plot(
            sub["layer_plot"],
            sub["lr_scale"],
            color=LAYER_COLORS[component],
            marker=LAYER_MARKERS[component],
            markersize=3.2,
            linewidth=0.8,
            label=LAYER_LABELS[component],
        )[0]
        ax_momentum.plot(
            sub["layer_plot"],
            sub["momentum"],
            color=LAYER_COLORS[component],
            marker=LAYER_MARKERS[component],
            markersize=3.2,
            linewidth=0.8,
        )
        handles.append(handle)
        labels.append(LAYER_LABELS[component])

    for ax in (ax_lr, ax_momentum):
        ax.set_xlim(-0.9, 7.35)
        ax.set_xticks([0, 1, 2, 3, 4, 5, 6, 7])
        paper_axis(ax, integer_x=True)

    ax_lr.axhline(1.0, color="#888888", linewidth=0.55, linestyle=":")
    ax_lr.set_ylabel(r"$\mathrm{Final\ LR\ scale}$")
    ax_lr.set_ylim(0.20, 1.05)
    ax_momentum.axhline(0.95, color="#888888", linewidth=0.55, linestyle=":")
    ax_momentum.set_ylabel(r"$\mathrm{Final\ momentum}$")
    ax_momentum.set_ylim(0.94, 0.976)
    fig.supxlabel(r"$\mathrm{Transformer\ block}$", y=0.008)

    pos = final[final["component"] == "pos_emb"]
    if not pos.empty:
        pos = pos.iloc[0]
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
    save_figure(fig, output)
    return Path(output).with_suffix(".pdf")


def make_all_paper_plots():
    return [
        make_baseline_figure(),
        make_gduo_scope_figure(),
        make_geometry_variants_figure(),
        make_muon_layerwise_figure(),
    ]
