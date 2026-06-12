import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


COLORS = {
    "baseline": "#3B6EA8",
    "global": "#D97904",
    "layerwise": "#2F8F5B",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.2,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.3,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
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
            "lines.linewidth": 1.0,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def format_axes(ax):
    ax.set_xlim(0, 2000)
    ax.set_ylim(3.35, 6.35)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def parse_eval_log(path: Path, optimizer: str, variant: str, label: str) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    rows = []
    for match in re.finditer(
        r">Eval: Iter=(\d+) \(([0-9.]+) epochs\) val_loss=([0-9.]+) "
        r"val_pp=([0-9.]+) val_acc=([0-9.eE+-]+)",
        text,
    ):
        rows.append(
            {
                "optimizer": optimizer,
                "variant": variant,
                "label": label,
                "iter": int(match.group(1)),
                "val_loss": float(match.group(3)),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["iter"] > 0].copy()
    return df


def best_stable_newton_name(df: pd.DataFrame) -> str:
    stable = df[df["optimizer"].str.startswith("newton stable", na=False)]
    finals = (
        stable.dropna(subset=["val_loss"])
        .sort_values(["optimizer", "iter"])
        .groupby("optimizer")
        .tail(1)
        .sort_values("val_loss")
    )
    if finals.empty:
        raise ValueError("No stable Newton-Muon baseline found.")
    return str(finals.iloc[0]["optimizer"])


def load_baselines(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    newton = best_stable_newton_name(raw)
    specs = [
        ("adamw", "adam", "Baseline"),
        ("muon", "muon", "Baseline"),
        (newton, "newton", "Baseline"),
    ]
    frames = []
    for opt, optimizer, label in specs:
        sub = raw[(raw["optimizer"] == opt) & raw["val_loss"].notna() & (raw["iter"] > 0)].copy()
        sub = sub[["iter", "val_loss"]]
        sub["optimizer"] = optimizer
        sub["variant"] = "baseline"
        sub["label"] = label
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def load_all(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_baselines(Path(args.baseline_curves))]
    frames.extend(
        [
            parse_eval_log(
                Path("izar_fetch/current_meta_logs/gpt_layer_wiki_3005857_0.out"),
                "adam",
                "layerwise",
                "LR+momentum layerwise",
            ),
            parse_eval_log(
                Path("izar_fetch/current_meta_logs/gpt_layer_wiki_3005857_1.out"),
                "muon",
                "layerwise",
                "LR+momentum layerwise",
            ),
            parse_eval_log(
                Path("izar_fetch/current_meta_logs/gpt_layer_wiki_3005857_2.out"),
                "newton",
                "layerwise",
                "LR+momentum layerwise",
            ),
            parse_eval_log(
                Path("izar_fetch/current_meta_logs/gpt_meta_follow_3005881_2.out"),
                "muon",
                "global",
                "LR+momentum global",
            ),
            parse_eval_log(
                Path("izar_fetch/logs/gpt_gduo_meta_2988587_5.out"),
                "newton",
                "global",
                "LR+momentum global",
            ),
        ]
    )

    # No true AdamW global LR+momentum run is available. We keep the plotting data
    # faithful and report this in the coverage table instead of adding a proxy.
    df = pd.concat([x for x in frames if not x.empty], ignore_index=True)
    df = df[df["iter"] <= 2000].copy()

    expected = pd.DataFrame(
        [
            ("adam", "Baseline", "baseline"),
            ("adam", "LR+momentum global", "global"),
            ("adam", "LR+momentum layerwise", "layerwise"),
            ("muon", "Baseline", "baseline"),
            ("muon", "LR+momentum global", "global"),
            ("muon", "LR+momentum layerwise", "layerwise"),
            ("newton", "Baseline", "baseline"),
            ("newton", "LR+momentum global", "global"),
            ("newton", "LR+momentum layerwise", "layerwise"),
        ],
        columns=["optimizer", "label", "variant"],
    )
    observed = (
        df.groupby(["optimizer", "label", "variant"], as_index=False)["iter"]
        .max()
        .rename(columns={"iter": "max_iter"})
    )
    coverage = expected.merge(observed, how="left", on=["optimizer", "label", "variant"])
    coverage["max_iter"] = coverage["max_iter"].fillna(0).astype(int)
    coverage["has_2000"] = coverage["max_iter"] >= 2000
    return df, coverage


def plot(df: pd.DataFrame, output: Path):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), sharey=True)
    panels = [
        ("adam", r"$\mathrm{AdamW}$"),
        ("muon", r"$\mathrm{Muon}$"),
        ("newton", r"$\mathrm{Newton{-}Muon}$"),
    ]
    order = [
        ("baseline", "Baseline", "-"),
        ("global", "LR+momentum global", "--"),
        ("layerwise", "LR+momentum layerwise", "-."),
    ]

    for ax, (optimizer, panel_label) in zip(axes, panels):
        sub_panel = df[df["optimizer"] == optimizer]
        for variant, label, linestyle in order:
            sub = sub_panel[(sub_panel["variant"] == variant) & (sub_panel["label"] == label)]
            if sub.empty:
                continue
            sub = sub.sort_values("iter")
            ax.plot(sub["iter"], sub["val_loss"], color=COLORS[variant], linestyle=linestyle, label=label)
        ax.text(0.03, 0.94, panel_label, transform=ax.transAxes, ha="left", va="top")
        ax.set_xlabel(r"$\mathrm{Training\ step}$")
        format_axes(ax)

    axes[0].set_ylabel(r"$\mathrm{Validation\ loss}$")
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
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
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.20, top=0.82, wspace=0.12)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper figure: baselines vs LR+momentum GD-UO scopes.")
    parser.add_argument(
        "--baseline_curves",
        default="izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv",
    )
    parser.add_argument("--output", default="figures/paper_like/figure2_gduo_scopes")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    df, coverage = load_all(args)
    output = Path(args.output)
    plot(df, output)
    coverage_path = output.with_suffix(".coverage.csv")
    coverage.to_csv(coverage_path, index=False)
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {coverage_path}")
    missing = coverage[~coverage["has_2000"]]
    if not missing.empty:
        print("Curves without 2000 steps:")
        for row in missing.itertuples(index=False):
            print(f"  {row.optimizer}: {row.label} max_iter={row.max_iter}")


if __name__ == "__main__":
    main()
