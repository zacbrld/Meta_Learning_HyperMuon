import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures" / "paper_like"
TABLE_DIR = ROOT / "tables"

EVAL_RE = re.compile(
    r">Eval: Iter=(\d+) .*?val_loss=([0-9.]+) "
    r"val_pp=([0-9.]+) val_acc=([0-9.eE+-]+)"
)
TRAIN_RE = re.compile(
    r"Train: Iter=(\d+).*?iter_dt=([0-9.eE+-]+)s"
    r"(?: iter_dt_avg=([0-9.eE+-]+)s)?"
)


def run_command(args):
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def read_text(path):
    return Path(path).read_text(errors="replace")


def evals_from_log(path):
    rows = []
    for step, loss, ppl, acc in EVAL_RE.findall(read_text(path)):
        rows.append(
            {
                "iter": int(step),
                "val_loss": float(loss),
                "val_ppl": float(ppl),
                "val_acc": float(acc),
            }
        )
    return pd.DataFrame(rows)


def last_eval(path, max_iter=2000):
    df = evals_from_log(path)
    if df.empty:
        raise ValueError(f"No eval rows found in {path}")
    df = df[(df["iter"] > 0) & (df["iter"] <= max_iter)]
    return df.sort_values("iter").iloc[-1].to_dict()


def seconds_per_iter(path, max_iter=2000):
    rows = []
    for step, dt, avg in TRAIN_RE.findall(read_text(path)):
        step = int(step)
        if step <= max_iter:
            rows.append((step, float(avg or dt)))
    if not rows:
        return None
    return sorted(rows)[-1][1]


def best_stable_newton(gpt_df):
    stable = gpt_df[gpt_df["optimizer"].str.startswith("newton stable", na=False)]
    finals = (
        stable.dropna(subset=["val_loss"])
        .sort_values(["optimizer", "iter"])
        .groupby("optimizer")
        .tail(1)
        .sort_values("val_loss")
    )
    if finals.empty:
        raise ValueError("No stable Newton-Muon curve found")
    return str(finals.iloc[0]["optimizer"])


def save_table(name, df, latex):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLE_DIR / f"{name}.csv", index=False)
    (TABLE_DIR / f"{name}.tex").write_text(latex.strip() + "\n")
    print(f"saved tables/{name}.csv", flush=True)
    print(f"saved tables/{name}.tex", flush=True)


def make_cifar_table():
    paths = sorted((ROOT / "izar_fetch" / "results_cifar_fig1").glob("cifar10_resmlp32_*_seed0.csv"))
    rows = []
    for path in paths:
        df = pd.read_csv(path)
        df = df[df["eval_split"] == "test"].dropna(subset=["eval_accuracy"])
        if df.empty:
            continue
        final = df.sort_values("step").iloc[-1]
        best = df.sort_values("eval_accuracy").iloc[-1]
        rows.append(
            {
                "optimizer": str(final["optimizer"]),
                "final_acc": float(final["eval_accuracy"]),
                "best_acc": float(best["eval_accuracy"]),
            }
        )
    order = {"adamw": 0, "muon": 1, "newton_muon": 2}
    df = pd.DataFrame(rows).sort_values("optimizer", key=lambda s: s.map(order))
    labels = {
        "adamw": "AdamW",
        "muon": "Muon",
        "newton_muon": "Newton--Muon",
    }
    body = "\n".join(
        f"{labels[row.optimizer]} & {100*row.final_acc:.2f}\\% & {100*row.best_acc:.2f}\\% \\\\"
        for row in df.itertuples(index=False)
    )
    latex = rf"""
\begin{{tabular}}{{lcc}}
\toprule
\textbf{{Optimizer}} & \textbf{{Final acc.}} & \textbf{{Best acc.}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    save_table("table_cifar_baselines", df, latex)


def make_wikitext_baseline_table():
    path = ROOT / "izar_fetch" / "llm_newton_stability" / "diagnostic" / "gpt_wikitext_with_stable_newton_curves.csv"
    raw = pd.read_csv(path)
    newton_name = best_stable_newton(raw)
    specs = [
        ("adamw", "AdamW"),
        ("muon", "Muon"),
        (newton_name, "Newton--Muon"),
    ]
    rows = []
    for opt, label in specs:
        sub = raw[(raw["optimizer"] == opt) & raw["val_loss"].notna() & (raw["iter"] > 0)]
        final = sub.sort_values("iter").iloc[-1]
        rows.append(
            {
                "optimizer": label,
                "source": opt,
                "iter": int(final["iter"]),
                "val_loss": float(final["val_loss"]),
                "val_acc": float(final["val_acc"]),
            }
        )
    df = pd.DataFrame(rows)
    body = "\n".join(
        f"{row.optimizer} & {row.iter:d} & {row.val_loss:.3f} & {row.val_acc:.4f} \\\\"
        for row in df.itertuples(index=False)
    )
    latex = rf"""
\begin{{tabular}}{{lccc}}
\toprule
\textbf{{Optimizer}} & \textbf{{Iter.}} & \textbf{{Val loss}} & \textbf{{Acc.}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    save_table("table_wikitext_baselines", df, latex)


def make_gduo_scope_table():
    path = ROOT / "figures" / "paper_like" / "figure2_gduo_scopes.coverage.csv"
    if path.exists():
        coverage = pd.read_csv(path)
    else:
        coverage = pd.DataFrame()
    specs = [
        ("AdamW", "Baseline", ROOT / "izar_fetch" / "llm_newton_stability" / "diagnostic" / "gpt_wikitext_with_stable_newton_curves.csv", "adamw"),
        ("AdamW", "Global", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_0.out", None),
        ("AdamW", "Layerwise", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_1.out", None),
        ("Muon", "Baseline", ROOT / "izar_fetch" / "llm_newton_stability" / "diagnostic" / "gpt_wikitext_with_stable_newton_curves.csv", "muon"),
        ("Muon", "Global", ROOT / "izar_fetch" / "current_meta_logs" / "gpt_meta_follow_3005881_2.out", None),
        ("Muon", "Layerwise", ROOT / "izar_fetch" / "current_meta_logs" / "gpt_layer_wiki_3005857_1.out", None),
        ("Newton--Muon", "Baseline", ROOT / "izar_fetch" / "llm_newton_stability" / "diagnostic" / "gpt_wikitext_with_stable_newton_curves.csv", "best_newton"),
        ("Newton--Muon", "Global", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_2.out", None),
        ("Newton--Muon", "Layerwise", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gduo_missing_3028678_3.out", None),
    ]
    raw = pd.read_csv(ROOT / "izar_fetch" / "llm_newton_stability" / "diagnostic" / "gpt_wikitext_with_stable_newton_curves.csv")
    newton_name = best_stable_newton(raw)
    rows = []
    for optimizer, setting, source, key in specs:
        if key is None:
            row = last_eval(source)
        else:
            opt = newton_name if key == "best_newton" else key
            sub = raw[(raw["optimizer"] == opt) & raw["val_loss"].notna() & (raw["iter"] > 0)]
            final = sub.sort_values("iter").iloc[-1]
            row = {"iter": int(final["iter"]), "val_loss": float(final["val_loss"]), "val_acc": float(final["val_acc"])}
        rows.append(
            {
                "optimizer": optimizer,
                "setting": setting,
                "iter": int(row["iter"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
            }
        )
    df = pd.DataFrame(rows)
    body = "\n".join(
        f"{row.optimizer} & {row.setting} & {row.val_loss:.3f} & {row.val_acc:.4f} \\\\"
        for row in df.itertuples(index=False)
    )
    latex = rf"""
\begin{{tabular}}{{llcc}}
\toprule
\textbf{{Optimizer}} & \textbf{{Setting}} & \textbf{{Val loss}} & \textbf{{Acc.}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    if not coverage.empty:
        coverage.to_csv(TABLE_DIR / "table_gduo_scope_coverage.csv", index=False)
    save_table("table_gduo_scopes", df, latex)


def make_geometry_table():
    specs = [
        ("Adam/Muon", "residual gate", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gate_wiki_3028681_0.out"),
        ("Muon/Newton", "learned Newton strength", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_gate_wiki_3028681_1.out"),
        ("AdaGrad-EMA/Muon", "diagonal EMA precond.", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_precond_wiki_3028637_0.out"),
        ("SOAP-lite/Muon", "matrix precond.", ROOT / "izar_fetch" / "recent_gating_precond" / "gpt_precond_wiki_3028637_1.out"),
        ("Muon", "LR+momentum layerwise", ROOT / "izar_fetch" / "current_meta_logs" / "gpt_layer_wiki_3005857_1.out"),
    ]
    rows = []
    for variant, setting, path in specs:
        final = last_eval(path)
        rows.append(
            {
                "variant": variant,
                "setting": setting,
                "iter": int(final["iter"]),
                "val_loss": float(final["val_loss"]),
                "val_acc": float(final["val_acc"]),
                "seconds_per_iter": seconds_per_iter(path),
            }
        )
    df = pd.DataFrame(rows)
    body = "\n".join(
        f"{row.variant} & {row.setting} & {row.val_loss:.3f} & {row.val_acc:.4f} & {row.seconds_per_iter:.2f} \\\\"
        for row in df.itertuples(index=False)
    )
    latex = rf"""
\begin{{tabular}}{{llccc}}
\toprule
\textbf{{Variant}} & \textbf{{Setting}} & \textbf{{Val loss}} & \textbf{{Acc.}} & \textbf{{s/it}} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
"""
    save_table("table_geometry", df, latex)


def make_tables():
    make_cifar_table()
    make_wikitext_baseline_table()
    make_gduo_scope_table()
    make_geometry_table()


def make_plots():
    run_command([sys.executable, "generate_plots.py"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-tables", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.skip_plots:
        make_plots()
    if not args.skip_tables:
        make_tables()
    print("done", flush=True)


if __name__ == "__main__":
    main()
