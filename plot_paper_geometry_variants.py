import argparse
import re
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator
import pandas as pd


EVAL_RE = re.compile(
    r">Eval: Iter=(\d+) \(([0-9.]+) epochs\) val_loss=([0-9.]+) "
    r"val_pp=([0-9.]+) val_acc=([0-9.eE+-]+)"
)


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
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
            "lines.linewidth": 0.95,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )


def parse_log(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    text = path.read_text(errors="replace")
    rows = []
    for match in EVAL_RE.finditer(text):
        rows.append(
            {
                "iter": int(match.group(1)),
                "val_loss": float(match.group(3)),
                "val_acc": float(match.group(5)),
                "label": label,
                "source": str(path),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[df["iter"] > 0].copy()
    return df


def log_metadata(path: Path) -> dict:
    text = path.read_text(errors="replace")
    opt = re.search(r"'opt': '([^']+)'", text)
    kind = re.search(r"'muon_precond_kind': '([^']+)'", text)
    configured_iter = re.search(r"'iterations': (\d+)", text)
    return {
        "opt": opt.group(1) if opt else "",
        "kind": kind.group(1) if kind else "",
        "configured_iter": int(configured_iter.group(1)) if configured_iter else 0,
    }


def choose_best_candidate(paths, opt: str, kind: str = "") -> Optional[Path]:
    candidates = []
    for path in paths:
        meta = log_metadata(path)
        if meta["opt"] != opt:
            continue
        if kind and meta["kind"] != kind:
            continue
        df = parse_log(path, "_candidate")
        if df.empty:
            continue
        max_iter = int(df["iter"].max())
        final_loss = float(df.sort_values("iter").iloc[-1]["val_loss"])
        candidates.append((max_iter, meta["configured_iter"], -final_loss, path.stat().st_mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][-1]


def load_curves(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    recent_dir = Path(args.recent_dir)
    recent_logs = sorted(recent_dir.glob("*.out"))
    specs = [
        (
            "Muon LR+momentum layerwise",
            Path(args.best_muon_layerwise),
            True,
        ),
        (
            "Adam/Muon gate",
            choose_best_candidate(recent_logs, "adam-muon-gate"),
            False,
        ),
        (
            "Muon/Newton-Muon gate",
            choose_best_candidate(recent_logs, "muon-newton-gate"),
            False,
        ),
        (
            "AdaGrad-EMA/Muon",
            choose_best_candidate(recent_logs, "muon-precond-gate", "adagrad_ema"),
            False,
        ),
        (
            "SOAP-lite/Muon",
            choose_best_candidate(recent_logs, "muon-precond-gate", "soap_lite"),
            False,
        ),
    ]

    frames = []
    coverage_rows = []
    for label, path, reference in specs:
        if path is None:
            coverage_rows.append(
                {
                    "label": label,
                    "source": "",
                    "max_iter": 0,
                    "has_2000": False,
                    "reference": reference,
                }
            )
            continue
        df = parse_log(path, label)
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


def plot(curves: pd.DataFrame, coverage: pd.DataFrame, output: Path):
    colors = {
        "Muon LR+momentum layerwise": "#111111",
        "Adam/Muon gate": "#3B6EA8",
        "Muon/Newton-Muon gate": "#D97904",
        "AdaGrad-EMA/Muon": "#2F8F5B",
        "SOAP-lite/Muon": "#8B5FBF",
    }
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
            color=colors[label],
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            label=legend_label,
        )

    ax.set_xlim(0, 2000)
    ax.set_ylim(3.35, 5.95)
    ax.set_xlabel(r"$\mathrm{Training\ step}$")
    ax.set_ylabel(r"$\mathrm{Validation\ loss}$")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(which="both", top=False, right=False)
    ax.legend(frameon=False, loc="upper right", handlelength=2.0, borderaxespad=0.15)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.185, top=0.985)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.015)
    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Paper plot for geometry/preconditioner variants.")
    parser.add_argument(
        "--recent_dir",
        default="izar_fetch/recent_gating_precond",
        help="Directory containing fetched gpt_gate/gpt_precond logs.",
    )
    parser.add_argument(
        "--best_muon_layerwise",
        default="izar_fetch/current_meta_logs/gpt_layer_wiki_3005857_1.out",
    )
    parser.add_argument(
        "--output",
        default="figures/paper_like/figure3_geometry_variants",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_style()
    curves, coverage = load_curves(args)
    output = Path(args.output)
    plot(curves, coverage, output)
    coverage_path = output.with_suffix(".coverage.csv")
    coverage.to_csv(coverage_path, index=False)
    print(f"Saved {output.with_suffix('.pdf')}")
    print(f"Saved {coverage_path}")
    missing = coverage[~coverage["has_2000"]]
    if not missing.empty:
        print("Curves without 2000 steps:")
        for row in missing.itertuples(index=False):
            print(f"  {row.label}: max_iter={row.max_iter} source={row.source}")


if __name__ == "__main__":
    main()
