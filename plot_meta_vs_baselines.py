import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASELINE_CSV = Path(
    "izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv"
)
LOG_DIR = Path("izar_fetch/current_meta_logs")
OUT_DIR = Path("figures/meta_learning")
TOKENS_PER_ITER = 12 * 4 * 1024


COLORS = {
    "adam": "#4C78A8",
    "muon": "#F58518",
    "newton": "#2CA02C",
    "meta": "#D62728",
    "meta2": "#9467BD",
}


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "lines.linewidth": 2.2,
            "savefig.dpi": 220,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_eval_log(path: Path, name: str) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    rows = []
    for match in re.finditer(
        r">Eval: Iter=(\d+) \(([0-9.]+) epochs\) val_loss=([0-9.]+) "
        r"val_pp=([0-9.]+) val_acc=([0-9.eE+-]+)",
        text,
    ):
        rows.append(
            {
                "optimizer": name,
                "iter": int(match.group(1)),
                "epoch": float(match.group(2)),
                "val_loss": float(match.group(3)),
                "val_pp": float(match.group(4)),
                "val_acc": float(match.group(5)),
            }
        )
    return pd.DataFrame(rows)


def parse_train_log(path: Path, name: str) -> pd.DataFrame:
    text = path.read_text(errors="replace")
    rows = []
    for match in re.finditer(
        r"Train: Iter=(\d+) .*?train_loss=([0-9.]+) iter_dt=([0-9.eE+-]+)s lr=([0-9.eE+-]+)",
        text,
    ):
        rows.append(
            {
                "optimizer": name,
                "iter": int(match.group(1)),
                "train_loss": float(match.group(2)),
                "iter_dt": float(match.group(3)),
                "lr": float(match.group(4)),
            }
        )
    return pd.DataFrame(rows)


def add_time_columns(eval_df: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    eval_df = eval_df.copy()
    if train_df.empty:
        eval_df["iter_dt"] = float("nan")
        eval_df["time_sec"] = float("nan")
        return eval_df
    dt = train_df["iter_dt"].tail(min(10, len(train_df))).mean()
    eval_df["iter_dt"] = dt
    eval_df["time_sec"] = eval_df["iter"] * dt
    return eval_df


def save(fig, stem: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_style()
    baseline = pd.read_csv(BASELINE_CSV)
    baseline = baseline[baseline["val_loss"].notna()].copy()

    adam_meta = parse_eval_log(
        LOG_DIR / "gpt_meta_follow_3005874_0.out", "AdamW-GDUO"
    )
    adam_meta_train = parse_train_log(
        LOG_DIR / "gpt_meta_follow_3005874_0.out", "AdamW-GDUO"
    )
    muon_layer = parse_eval_log(
        LOG_DIR / "gpt_layer_wiki_3005857_1.out", "Muon-GDUO layerwise"
    )
    muon_layer_train = parse_train_log(
        LOG_DIR / "gpt_layer_wiki_3005857_1.out", "Muon-GDUO layerwise"
    )
    muon_global = parse_eval_log(
        LOG_DIR / "gpt_meta_follow_3005881_2.out", "Muon-GDUO global"
    )
    muon_global_train = parse_train_log(
        LOG_DIR / "gpt_meta_follow_3005881_2.out", "Muon-GDUO global"
    )
    newton_ridge05 = parse_eval_log(
        LOG_DIR / "gpt_layer_wiki_3005857_2.out", "Newton-GDUO ridge=.5"
    )
    newton_ridge05_train = parse_train_log(
        LOG_DIR / "gpt_layer_wiki_3005857_2.out", "Newton-GDUO ridge=.5"
    )
    newton_ridge1 = parse_eval_log(
        LOG_DIR / "gpt_meta_follow_3005874_1.out", "Newton-GDUO ridge=1"
    )
    newton_ridge1_train = parse_train_log(
        LOG_DIR / "gpt_meta_follow_3005874_1.out", "Newton-GDUO ridge=1"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.45))

    ax = axes[0]
    sub = baseline[baseline["optimizer"] == "adamw"].sort_values("iter")
    ax.plot(sub["iter"], sub["val_loss"], label="AdamW baseline", color=COLORS["adam"])
    ax.plot(
        adam_meta["iter"],
        adam_meta["val_loss"],
        label="AdamW-GDUO",
        color=COLORS["meta"],
    )
    ax.set_title("AdamW")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation loss")
    ax.set_xlim(0, 2000)
    ax.legend(frameon=False)

    ax = axes[1]
    sub = baseline[baseline["optimizer"] == "muon"].sort_values("iter")
    ax.plot(sub["iter"], sub["val_loss"], label="Muon baseline", color=COLORS["muon"])
    ax.plot(
        muon_layer["iter"],
        muon_layer["val_loss"],
        label="Muon-GDUO layerwise",
        color=COLORS["meta"],
    )
    ax.plot(
        muon_global["iter"],
        muon_global["val_loss"],
        label="Muon-GDUO global",
        color=COLORS["meta2"],
        linestyle="--",
    )
    ax.set_title("Muon")
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, 3000)
    ax.legend(frameon=False)

    ax = axes[2]
    for opt, label, color, style in [
        ("newton stable lr0p005_ridge0p5_clip3", "baseline lr=.005 ridge=.5", COLORS["newton"], "-"),
        ("newton stable lr0p004_ridge1p0_clip3", "baseline lr=.004 ridge=1", "#8C564B", "-"),
    ]:
        sub = baseline[
            (baseline["optimizer"] == opt)
            & (baseline["iter"] > 0)
            & (baseline["iter"] <= 500)
        ].sort_values("iter")
        ax.plot(sub["iter"], sub["val_loss"], label=label, color=color, linestyle=style, marker="o")
    ax.plot(
        newton_ridge05[newton_ridge05["iter"] > 0]["iter"],
        newton_ridge05[newton_ridge05["iter"] > 0]["val_loss"],
        label="GDUO lr=.005 ridge=.5",
        color=COLORS["meta"],
        marker="o",
    )
    ax.plot(
        newton_ridge1[newton_ridge1["iter"] > 0]["iter"],
        newton_ridge1[newton_ridge1["iter"] > 0]["val_loss"],
        label="GDUO lr=.005 ridge=1",
        color=COLORS["meta2"],
        linestyle="--",
        marker="o",
    )
    ax.set_title("Newton-Muon, zoom 100-500")
    ax.set_xlabel("Iteration")
    ax.set_xlim(100, 500)
    ax.set_ylim(4.7, 6.0)
    ax.legend(frameon=False)

    fig.suptitle("WikiText GPT-51M: baselines vs GD-UO meta-learning", y=1.03)
    save(fig, "gpt_wikitext_baselines_vs_meta")

    plot_metric_panels(
        baseline,
        adam_meta,
        muon_layer,
        muon_global,
        newton_ridge05,
        newton_ridge1,
        metric="val_acc",
        ylabel="Validation accuracy",
        stem="gpt_wikitext_accuracy_baselines_vs_meta",
    )
    plot_metric_panels(
        baseline,
        adam_meta,
        muon_layer,
        muon_global,
        newton_ridge05,
        newton_ridge1,
        metric="val_pp",
        ylabel="Validation perplexity",
        stem="gpt_wikitext_perplexity_baselines_vs_meta",
        logy=True,
    )

    timed = [
        add_time_columns(adam_meta, adam_meta_train),
        add_time_columns(muon_layer, muon_layer_train),
        add_time_columns(muon_global, muon_global_train),
        add_time_columns(newton_ridge05, newton_ridge05_train),
        add_time_columns(newton_ridge1, newton_ridge1_train),
    ]
    plot_time_loss(timed)
    plot_throughput(
        [
            ("AdamW-GDUO", adam_meta_train),
            ("Muon-GDUO layerwise", muon_layer_train),
            ("Muon-GDUO global", muon_global_train),
            ("Newton-GDUO ridge=.5", newton_ridge05_train),
            ("Newton-GDUO ridge=1", newton_ridge1_train),
        ]
    )


def plot_metric_panels(
    baseline,
    adam_meta,
    muon_layer,
    muon_global,
    newton_ridge05,
    newton_ridge1,
    metric,
    ylabel,
    stem,
    logy=False,
):
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 3.45))
    ax = axes[0]
    sub = baseline[baseline["optimizer"] == "adamw"].sort_values("iter")
    ax.plot(sub["iter"], sub[metric], label="AdamW baseline", color=COLORS["adam"])
    ax.plot(adam_meta["iter"], adam_meta[metric], label="AdamW-GDUO", color=COLORS["meta"])
    ax.set_title("AdamW")
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 2000)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[1]
    sub = baseline[baseline["optimizer"] == "muon"].sort_values("iter")
    ax.plot(sub["iter"], sub[metric], label="Muon baseline", color=COLORS["muon"])
    ax.plot(muon_layer["iter"], muon_layer[metric], label="Muon-GDUO layerwise", color=COLORS["meta"])
    ax.plot(muon_global["iter"], muon_global[metric], label="Muon-GDUO global", color=COLORS["meta2"], linestyle="--")
    ax.set_title("Muon")
    ax.set_xlabel("Iteration")
    ax.set_xlim(0, 3000)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False)

    ax = axes[2]
    for opt, label, color, style in [
        ("newton stable lr0p005_ridge0p5_clip3", "baseline lr=.005 ridge=.5", COLORS["newton"], "-"),
        ("newton stable lr0p004_ridge1p0_clip3", "baseline lr=.004 ridge=1", "#8C564B", "-"),
    ]:
        sub = baseline[
            (baseline["optimizer"] == opt)
            & (baseline["iter"] > 0)
            & (baseline["iter"] <= 500)
        ].sort_values("iter")
        ax.plot(sub["iter"], sub[metric], label=label, color=color, linestyle=style, marker="o")
    ax.plot(
        newton_ridge05[newton_ridge05["iter"] > 0]["iter"],
        newton_ridge05[newton_ridge05["iter"] > 0][metric],
        label="GDUO lr=.005 ridge=.5",
        color=COLORS["meta"],
        marker="o",
    )
    ax.plot(
        newton_ridge1[newton_ridge1["iter"] > 0]["iter"],
        newton_ridge1[newton_ridge1["iter"] > 0][metric],
        label="GDUO lr=.005 ridge=1",
        color=COLORS["meta2"],
        linestyle="--",
        marker="o",
    )
    ax.set_title("Newton-Muon, zoom 100-500")
    ax.set_xlabel("Iteration")
    ax.set_xlim(100, 500)
    if logy:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    fig.suptitle(f"WikiText GPT-51M: {ylabel}", y=1.03)
    save(fig, stem)


def plot_time_loss(frames):
    fig, ax = plt.subplots(figsize=(5.9, 3.6))
    styles = {
        "AdamW-GDUO": (COLORS["adam"], "-"),
        "Muon-GDUO layerwise": (COLORS["meta"], "-"),
        "Muon-GDUO global": (COLORS["meta2"], "--"),
        "Newton-GDUO ridge=.5": (COLORS["newton"], "-"),
        "Newton-GDUO ridge=1": ("#8C564B", "--"),
    }
    for df in frames:
        if df.empty:
            continue
        name = df["optimizer"].iloc[0]
        color, style = styles.get(name, ("black", "-"))
        ax.plot(df["time_sec"] / 60.0, df["val_loss"], label=name, color=color, linestyle=style)
    ax.set_xlabel("Approx. training time (minutes)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Meta-learning runs by wall-clock time")
    ax.legend(frameon=False)
    save(fig, "gpt_wikitext_meta_loss_vs_time")


def plot_throughput(train_frames):
    labels = []
    toks = []
    dts = []
    for name, df in train_frames:
        if df.empty:
            continue
        dt = df["iter_dt"].tail(min(10, len(df))).mean()
        labels.append(name)
        dts.append(dt)
        toks.append(TOKENS_PER_ITER / dt)
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    bars = ax.bar(range(len(labels)), toks, color="#4C78A8")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_ylabel("Approx. training tokens/sec")
    ax.set_title("Throughput from logged iter_dt")
    for bar, dt in zip(bars, dts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{dt:.2f}s/iter",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    save(fig, "gpt_wikitext_meta_throughput")


if __name__ == "__main__":
    main()
