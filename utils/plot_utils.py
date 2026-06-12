import logging
import re
from pathlib import Path
from typing import Optional, Union

import matplotlib

matplotlib.use("Agg")
logging.getLogger("fontTools").setLevel(logging.ERROR)
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PathLike = Union[Path, str]

EVAL_RE = re.compile(
    r">Eval: Iter=(\d+) \(([0-9.]+) epochs\) val_loss=([0-9.]+) "
    r"val_pp=([0-9.]+) val_acc=([0-9.eE+-]+)"
)

GDUO_RE = re.compile(
    r"\[GDUO-Muon\] step=(\d+) "
    r"lr_scale_avg=([0-9.eE+-]+) \(min=([0-9.eE+-]+), max=([0-9.eE+-]+)\) "
    r"lr_bound_frac=([0-9.eE+-]+) "
    r"momentum_avg=([0-9.eE+-]+) \(min=([0-9.eE+-]+), max=([0-9.eE+-]+)\)"
)

BASELINE_COLORS = {
    "adamw": "#3B6EA8",
    "muon": "#D97904",
    "newton": "#2F8F5B",
}

SCOPE_COLORS = {
    "baseline": "#3B6EA8",
    "global": "#D97904",
    "layerwise": "#2F8F5B",
}

GEOMETRY_COLORS = {
    "Muon LR+momentum layerwise": "#111111",
    "Adam/Muon gate": "#3B6EA8",
    "Muon/Newton-Muon gate": "#D97904",
    "AdaGrad-EMA/Muon": "#2F8F5B",
    "SOAP-lite/Muon": "#8B5FBF",
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


def set_paper_style(
    font_size: float = 7.8,
    label_size: float = 8.2,
    legend_size: float = 7.0,
    line_width: float = 0.95,
) -> None:
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


def paper_axis(ax, x_bins: int = 5, y_bins: int = 5, integer_x: bool = False) -> None:
    ax.xaxis.set_major_locator(MaxNLocator(nbins=x_bins, integer=integer_x))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=y_bins))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)


def save_figure(fig, output: Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def read_text(path: Path) -> str:
    return Path(path).read_text(errors="replace")


def parse_eval_log(path: Path, label: str = "", optimizer: str = "", variant: str = "") -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    rows = []
    for match in EVAL_RE.finditer(read_text(path)):
        rows.append(
            {
                "iter": int(match.group(1)),
                "val_loss": float(match.group(3)),
                "val_ppl": float(match.group(4)),
                "val_acc": float(match.group(5)),
                "label": label,
                "optimizer": optimizer,
                "variant": variant,
                "source": str(path),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["iter"] > 0].copy()
    return df


def load_cifar_test_curves(cifar_dir: Path) -> pd.DataFrame:
    paths = sorted(Path(cifar_dir).glob("cifar10_resmlp32_*_seed0.csv"))
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


def load_gpt_curves(gpt_curves: Path) -> pd.DataFrame:
    if not Path(gpt_curves).exists():
        raise FileNotFoundError(f"Missing GPT curves CSV: {gpt_curves}")
    return pd.read_csv(gpt_curves)


def best_stable_newton_name(gpt_df: pd.DataFrame) -> str:
    stable = gpt_df[gpt_df["optimizer"].str.startswith("newton stable", na=False)]
    finals = (
        stable.dropna(subset=["val_loss"])
        .sort_values(["optimizer", "iter"])
        .groupby("optimizer", as_index=False)
        .tail(1)
        .sort_values("val_loss")
    )
    if finals.empty:
        raise ValueError("No stable Newton-Muon curve found.")
    return str(finals.iloc[0]["optimizer"])


def log_metadata(path: Path) -> dict:
    text = read_text(path)
    opt = re.search(r"'opt': '([^']+)'", text)
    kind = re.search(r"'muon_precond_kind': '([^']+)'", text)
    configured_iter = re.search(r"'iterations': (\d+)", text)
    return {
        "opt": opt.group(1) if opt else "",
        "kind": kind.group(1) if kind else "",
        "configured_iter": int(configured_iter.group(1)) if configured_iter else 0,
    }


def choose_best_candidate(paths: list[Path], opt: str, kind: str = "") -> Optional[Path]:
    candidates = []
    for path in paths:
        meta = log_metadata(path)
        if meta["opt"] != opt:
            continue
        if kind and meta["kind"] != kind:
            continue
        df = parse_eval_log(path, "_candidate")
        if df.empty:
            continue
        max_iter = int(df["iter"].max())
        final_loss = float(df.sort_values("iter").iloc[-1]["val_loss"])
        candidates.append((max_iter, meta["configured_iter"], -final_loss, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1]


def make_baseline_figure(
    cifar_dir: PathLike = ROOT / "izar_fetch" / "results_cifar_fig1",
    gpt_curves: PathLike = ROOT
    / "izar_fetch"
    / "llm_newton_stability"
    / "diagnostic"
    / "gpt_wikitext_with_stable_newton_curves.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure1_cifar_wikitext_baselines",
) -> Path:
    set_paper_style(font_size=7.6, label_size=7.9, legend_size=7.0, line_width=0.95)
    cifar_df = load_cifar_test_curves(Path(cifar_dir))
    gpt_df = load_gpt_curves(Path(gpt_curves))
    newton_name = best_stable_newton_name(gpt_df)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05))

    cifar_specs = [
        ("adamw", "AdamW", BASELINE_COLORS["adamw"]),
        ("muon", "Muon", BASELINE_COLORS["muon"]),
        ("newton_muon", "Newton-Muon", BASELINE_COLORS["newton"]),
    ]
    for opt, label, color in cifar_specs:
        sub = cifar_df[cifar_df["optimizer"] == opt].sort_values("epoch")
        if not sub.empty:
            axes[0].plot(sub["step"], 100.0 * sub["eval_accuracy"], label=label, color=color)

    axes[0].set_ylabel(r"$\mathrm{Test\ accuracy}\;(\%)$")
    axes[0].set_xlim(0, 1300)
    axes[0].set_ylim(20, 70)
    paper_axis(axes[0])

    gpt_specs = [
        ("adamw", "AdamW", BASELINE_COLORS["adamw"]),
        ("muon", "Muon", BASELINE_COLORS["muon"]),
        (newton_name, "Newton-Muon", BASELINE_COLORS["newton"]),
    ]
    for opt, label, color in gpt_specs:
        sub = gpt_df[(gpt_df["optimizer"] == opt) & gpt_df["val_loss"].notna()].sort_values("iter")
        sub = sub[sub["iter"] > 0]
        if not sub.empty:
            axes[1].plot(sub["iter"], sub["val_loss"], label=label, color=color)

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
    output = Path(output)
    save_figure(fig, output)
    return output.with_suffix(".pdf")


def load_scope_baselines(path: Path) -> pd.DataFrame:
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


def load_gduo_scope_curves(
    baseline_curves: PathLike = ROOT
    / "izar_fetch"
    / "llm_newton_stability"
    / "diagnostic"
    / "gpt_wikitext_with_stable_newton_curves.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [load_scope_baselines(Path(baseline_curves))]
    frames.extend(
        [
            parse_eval_log(
                ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_0.out",
                "LR+momentum global",
                "adam",
                "global",
            ),
            parse_eval_log(
                ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_1.out",
                "LR+momentum layerwise",
                "adam",
                "layerwise",
            ),
            parse_eval_log(
                ROOT / "izar_fetch" / "current_meta_logs" / "gpt_layer_wiki_3005857_1.out",
                "LR+momentum layerwise",
                "muon",
                "layerwise",
            ),
            parse_eval_log(
                ROOT / "izar_fetch" / "current_meta_logs" / "gpt_meta_follow_3005881_2.out",
                "LR+momentum global",
                "muon",
                "global",
            ),
            parse_eval_log(
                ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_2.out",
                "LR+momentum global",
                "newton",
                "global",
            ),
            parse_eval_log(
                ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_3.out",
                "LR+momentum layerwise",
                "newton",
                "layerwise",
            ),
        ]
    )

    df = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
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


def make_gduo_scope_figure(
    baseline_curves: PathLike = ROOT
    / "izar_fetch"
    / "llm_newton_stability"
    / "diagnostic"
    / "gpt_wikitext_with_stable_newton_curves.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure2_gduo_scopes",
) -> tuple[Path, pd.DataFrame]:
    set_paper_style(font_size=8.2, label_size=8.5, legend_size=7.3, line_width=1.0)
    df, coverage = load_gduo_scope_curves(baseline_curves)
    fig = plt.figure(figsize=(6.3, 4.35))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.0], hspace=0.38, wspace=0.36)
    axes = [
        fig.add_subplot(gs[0, 0:2]),
        fig.add_subplot(gs[0, 2:4]),
        fig.add_subplot(gs[1, 1:3]),
    ]
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
            if not sub.empty:
                ax.plot(
                    sub.sort_values("iter")["iter"],
                    sub.sort_values("iter")["val_loss"],
                    color=SCOPE_COLORS[variant],
                    linestyle=linestyle,
                    label=label,
                )
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
    output = Path(output)
    save_figure(fig, output)
    coverage.to_csv(output.with_suffix(".coverage.csv"), index=False)
    return output.with_suffix(".pdf"), coverage


def load_geometry_curves(
    recent_dir: PathLike = ROOT / "izar_fetch" / "recent_gating_precond",
    best_muon_layerwise: PathLike = ROOT
    / "izar_fetch"
    / "current_meta_logs"
    / "gpt_layer_wiki_3005857_1.out",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent_logs = sorted(Path(recent_dir).glob("*.out"))
    specs = [
        ("Muon LR+momentum layerwise", Path(best_muon_layerwise), True),
        ("Adam/Muon gate", choose_best_candidate(recent_logs, "adam-muon-gate"), False),
        ("Muon/Newton-Muon gate", choose_best_candidate(recent_logs, "muon-newton-gate"), False),
        ("AdaGrad-EMA/Muon", choose_best_candidate(recent_logs, "muon-precond-gate", "adagrad_ema"), False),
        ("SOAP-lite/Muon", choose_best_candidate(recent_logs, "muon-precond-gate", "soap_lite"), False),
    ]

    frames = []
    coverage_rows = []
    for label, path, reference in specs:
        if path is None:
            coverage_rows.append(
                {"label": label, "source": "", "max_iter": 0, "has_2000": False, "reference": reference}
            )
            continue
        df = parse_eval_log(path, label)
        df = df[df["iter"] <= 2000].copy()
        max_iter = int(df["iter"].max()) if not df.empty else 0
        df["has_2000"] = max_iter >= 2000
        df["reference"] = reference
        frames.append(df)
        coverage_rows.append(
            {
                "label": label,
                "source": str(path),
                "max_iter": max_iter,
                "has_2000": max_iter >= 2000,
                "reference": reference,
            }
        )
    curves = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return curves, coverage


def make_geometry_variants_figure(
    recent_dir: PathLike = ROOT / "izar_fetch" / "recent_gating_precond",
    best_muon_layerwise: PathLike = ROOT
    / "izar_fetch"
    / "current_meta_logs"
    / "gpt_layer_wiki_3005857_1.out",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure3_geometry_variants",
) -> tuple[Path, pd.DataFrame]:
    set_paper_style(font_size=8.0, label_size=8.5, legend_size=7.0, line_width=0.95)
    curves, coverage = load_geometry_curves(recent_dir, best_muon_layerwise)
    order = [
        "Muon LR+momentum layerwise",
        "Adam/Muon gate",
        "Muon/Newton-Muon gate",
        "AdaGrad-EMA/Muon",
        "SOAP-lite/Muon",
    ]
    fig, ax = plt.subplots(figsize=(3.55, 2.45))
    for label in order:
        sub = curves[curves["label"] == label].sort_values("iter")
        if sub.empty:
            continue
        row = coverage[coverage["label"] == label].iloc[0]
        linestyle = "-" if bool(row.has_2000) else "--"
        linewidth = 1.15 if label == "Muon LR+momentum layerwise" else 0.9
        alpha = 1.0 if bool(row.has_2000) or label == "Muon LR+momentum layerwise" else 0.72
        legend_label = label if bool(row.has_2000) else rf"{label} ({int(row.max_iter)})"
        ax.plot(
            sub["iter"],
            sub["val_loss"],
            color=GEOMETRY_COLORS[label],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=legend_label,
        )
    ax.set_xlim(0, 2000)
    ax.set_ylim(3.35, 5.95)
    ax.set_xlabel(r"$\mathrm{Training\ step}$")
    ax.set_ylabel(r"$\mathrm{Validation\ loss}$")
    paper_axis(ax)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.15)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.185, top=0.985)
    output = Path(output)
    save_figure(fig, output)
    coverage.to_csv(output.with_suffix(".coverage.csv"), index=False)
    return output.with_suffix(".pdf"), coverage


def parse_gduo_aggregate_log(path: Path) -> pd.DataFrame:
    rows = []
    for match in GDUO_RE.finditer(read_text(path)):
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


def prepare_layerwise_final(path: PathLike) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["use_muon"].astype(bool)].copy()
    df = df[df["component"].isin(LAYER_COMPONENTS)].copy()
    df["layer_plot"] = df["layer"].astype(float)
    df.loc[df["component"] == "pos_emb", "layer_plot"] = -0.55
    return df


def make_muon_layerwise_figure(
    final_csv: PathLike = ROOT
    / "izar_fetch"
    / "layerwise_3005859"
    / "results"
    / "gduo_layerwise_final.csv",
    output: PathLike = ROOT / "figures" / "paper_like" / "figure4_muon_layerwise_lr_momentum",
) -> tuple[Path, pd.DataFrame]:
    set_paper_style(font_size=7.6, label_size=7.9, legend_size=6.4, line_width=0.9)
    final = prepare_layerwise_final(final_csv)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.05), sharex=True)
    ax_lr, ax_momentum = axes
    handles = []
    labels = []

    for component in [component for component in LAYER_COMPONENTS if component != "pos_emb"]:
        sub = final[final["component"] == component].sort_values("layer_plot")
        if sub.empty:
            continue
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
    output = Path(output)
    save_figure(fig, output)
    final.to_csv(output.with_suffix(".final.csv"), index=False)
    return output.with_suffix(".pdf"), final


def make_all_paper_plots() -> list[Path]:
    outputs = []
    outputs.append(make_baseline_figure())
    scope_output, _ = make_gduo_scope_figure()
    geometry_output, _ = make_geometry_variants_figure()
    layerwise_output, _ = make_muon_layerwise_figure()
    outputs.extend([scope_output, geometry_output, layerwise_output])
    return outputs
