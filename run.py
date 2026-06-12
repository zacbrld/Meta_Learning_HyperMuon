#!/usr/bin/env python3
"""Reproducibility entry point for the mini-project report ``Report_OML.pdf``:

    "Meta-Learning HyperMuon: Learning the Dynamics and Geometry of Matrix
     Optimizers via Hypergradients"  (Bourlard, Chetaille, Lhote -- EPFL, OptML 2026)

This single script orchestrates every experiment and figure used in the paper.
It does not re-implement anything; it just calls the existing entry points with
the exact hyper-parameters reported in the paper:

  * CIFAR-10 (ResMLP-32) ........ ``train_cifar_fig1.py``                (local)
  * GPT-51M / WikiText .......... ``external/llm-baselines/src/main.py`` (GPU)
  * Figures ..................... ``plot_paper_*.py``                    (local)

Mapping of report deliverables to stages (run ``--list`` to print this):

  Stage                Produces                                          Compute
  -------------------- ------------------------------------------------- -------
  cifar-baselines      Table 1 (CIFAR cols) + Fig 1 (top)                CPU/1 GPU
  cifar-meta           CIFAR GD-UO LR sanity runs (appendix/local)       CPU/1 GPU
  wikitext-baselines   Table 1 (WikiText cols) + Fig 1 (bottom)          1 GPU
  wikitext-dynamics    Fig 3 + Fig 4 (LR+momentum, global & layerwise)   1 GPU
  wikitext-geometry    Table 2 (Adam/Muon, Muon/Newton, AdaGrad, SOAP)   1 GPU
  figures              Regenerates all paper figures from CSVs           CPU
  all                  Everything above, in order                        1 GPU

Typical usage
-------------
    python run.py --list                 # show the stage -> deliverable map
    python run.py cifar-baselines        # reproduce the CIFAR baselines locally
    python run.py cifar-baselines --quick   # 2-epoch smoke test
    python run.py wikitext-dynamics --dry-run   # print the exact GPU commands
    python run.py figures                # rebuild figures from available CSVs
    python run.py all --dry-run          # inspect the whole pipeline

Notes
-----
* The GPT-51M / WikiText runs require a CUDA GPU and were executed on the EPFL
  Izar cluster (see the ``run_wikitext_*_izar.slurm`` launchers, which are the
  authoritative cluster configs that this script mirrors). On a single GPU each
  2000-iteration run takes a few hours; use ``--quick`` for a short smoke test.
* Raw run outputs (``results_*/``, ``izar_fetch/``, ``figures/``) are
  git-ignored. The ``figures`` stage regenerates a figure only when its input
  CSV is present locally; missing inputs are skipped with a warning rather than
  aborting the whole stage.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
LLM = REPO / "external" / "llm-baselines"

# --------------------------------------------------------------------------- #
# CIFAR-10 (ResMLP-32) -- runs locally via train_cifar_fig1.py
# --------------------------------------------------------------------------- #
CIFAR_BASELINES = ["adamw", "muon", "newton_muon"]
CIFAR_META = ["adamw_gduo_lr", "muon_gduo_lr", "newton_muon_gduo_lr"]

# --------------------------------------------------------------------------- #
# GPT-51M / WikiText -- shared config (matches the *_izar.slurm launchers and
# Appendix A: 8 transformer blocks, n_embd 512, seq 1024, cosine schedule).
# --------------------------------------------------------------------------- #
GPT_COMMON = [
    "--model", "base",
    "--dataset", "wikitext",
    "--data_in_ram",
    "--vocab_size", "50304",
    "--device", "cuda:0",
    "--dtype", "bfloat16",
    "--n_layer", "8",
    "--n_head", "8",
    "--n_embd", "512",
    "--sequence_length", "1024",
    "--batch_size", "12",
    "--acc_steps", "4",
    "--warmup_steps", "50",
    "--eval_interval", "100",
    "--eval_batches", "32",
    "--latest_ckpt_interval", "100",
    "--log_interval", "50",
    "--scheduler", "cos",
    "--lr", "0.001",
    "--weight_decay", "0.1",
    "--beta1", "0.9",
    "--beta2", "0.95",
    "--momentum", "0.95",
    "--nesterov", "True",
    "--muon_ns_steps", "5",
    "--grad_clip", "1.0",
]

# Per-optimizer extra flags reused by baselines, dynamics and geometry stages.
MUON_FLAGS = ["--muon_lr_factor", "0.02"]
NEWTON_FLAGS = [
    "--newton_muon_lr_factor", "0.005",
    "--newton_muon_ewma_beta", "0.95",
    "--newton_muon_ridge", "0.5",
    "--newton_muon_refresh_interval", "32",
    "--newton_muon_second_moment_init", "1.0",
    "--newton_muon_max_precond_dim", "1024",
    "--newton_muon_block_size", "512",
    "--newton_muon_precond_clip", "3.0",
    "--newton_muon_precond_log_interval", "200",
]

# Hyper-learning-rates for the GD-UO meta-updates (from the layerwise launcher).
GDUO_HYPER = {
    "adamw-gduo":        ("100",  "5000"),
    "muon-gduo":         ("1000", "100000"),
    "newton-muon-gduo":  ("3000", "300000"),
}


def gpt_base(prefix: str, results_dir: str, datasets_dir: str, iters: int) -> list[str]:
    """Common ``src/main.py`` argv prefix for one WikiText run."""
    return [
        sys.executable, "-u", "src/main.py",
        "--run_prefix", prefix,
        "--experiment_name", prefix,
        "--results_base_folder", results_dir,
        "--datasets_dir", datasets_dir,
        "--iterations", str(iters),
        *GPT_COMMON,
    ]


# --------------------------------------------------------------------------- #
# Command construction per stage
# --------------------------------------------------------------------------- #
def cifar_cmds(opts: list[str], args) -> list[tuple[list[str], dict]]:
    epochs = "2" if args.quick else str(args.epochs)
    cmds = []
    for opt in opts:
        cmd = [
            sys.executable, "-u", "train_cifar_fig1.py",
            "--optimizer", opt,
            "--seed", str(args.seed),
            "--epochs", epochs,
            "--results_dir", args.cifar_results,
            "--data_root", args.data_root,
        ]
        if opt.endswith("gduo_lr"):
            cmd += ["--min_lr_ratio", "1.0"]
        if args.quick:
            cmd += ["--batch_size", "1024", "--eval_interval", "1"]
        cmds.append((cmd, {"cwd": str(REPO)}))
    return cmds


def wikitext_baseline_cmds(args) -> list[tuple[list[str], dict]]:
    iters = 200 if args.quick else args.gpt_iters
    env = {**os.environ, "PYTHONPATH": f"{LLM / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    specs = {
        "adamw":       (["--opt", "adamw"]),
        "muon":        (["--opt", "muon", *MUON_FLAGS]),
        "newton_muon": (["--opt", "newton-muon", *NEWTON_FLAGS]),
    }
    cmds = []
    for tag, extra in specs.items():
        prefix = f"wiki_baseline_{tag}"
        cmd = gpt_base(prefix, args.gpt_results, args.datasets_dir, iters) + list(extra)
        cmds.append((cmd, {"cwd": str(LLM), "env": env}))
    return cmds


def wikitext_dynamics_cmds(args) -> list[tuple[list[str], dict]]:
    """Fig 3 / Fig 4: LR + momentum, both ``global`` and ``tensor`` (layerwise)."""
    iters = 200 if args.quick else args.gpt_iters
    env = {**os.environ, "PYTHONPATH": f"{LLM / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    base_opts = {
        "adamw-gduo":       [],
        "muon-gduo":        list(MUON_FLAGS),
        "newton-muon-gduo": list(NEWTON_FLAGS),
    }
    cmds = []
    for scope in ("global", "tensor"):
        for opt, extra in base_opts.items():
            lr_hyper, mom_hyper = GDUO_HYPER[opt]
            prefix = f"wiki_dyn_{opt}_{scope}"
            cmd = gpt_base(prefix, args.gpt_results, args.datasets_dir, iters) + [
                "--opt", opt,
                "--gduo_learn_lr",
                "--gduo_learn_momentum",
                "--gduo_scope", scope,
                "--gduo_ema_beta", "0.9",
                "--gduo_hypergrad_clip", "0.0",
                "--gduo_log_interval", "10",
                "--gduo_lr_hyper_lr", lr_hyper,
                "--gduo_momentum_hyper_lr", mom_hyper,
                *extra,
            ]
            cmds.append((cmd, {"cwd": str(LLM), "env": env}))
    return cmds


def wikitext_geometry_cmds(args) -> list[tuple[list[str], dict]]:
    """Table 2: soft geometry gates on top of Muon layerwise LR+momentum."""
    iters = 200 if args.quick else args.gpt_iters
    env = {**os.environ, "PYTHONPATH": f"{LLM / 'src'}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    lr_hyper, mom_hyper = GDUO_HYPER["muon-gduo"]
    common_gate = [
        "--gduo_learn_lr", "--gduo_learn_momentum",
        "--gduo_scope", "tensor",
        "--gduo_ema_beta", "0.9",
        "--gduo_hypergrad_clip", "0.0",
        "--gduo_log_interval", "10",
        "--gduo_lr_hyper_lr", lr_hyper,
        "--gduo_momentum_hyper_lr", mom_hyper,
        *MUON_FLAGS,
    ]
    specs = {
        "adam_muon_gate":   ["--opt", "adam-muon-gate", "--gduo_gate_init", "0.10"],
        "muon_newton_gate": ["--opt", "muon-newton-gate", "--gduo_gate_init", "0.05",
                             "--newton_muon_ewma_beta", "0.95",
                             "--newton_muon_ridge", "0.5",
                             "--newton_muon_refresh_interval", "32",
                             "--newton_muon_max_precond_dim", "1024",
                             "--newton_muon_block_size", "512",
                             "--newton_muon_precond_clip", "3.0",
                             "--newton_muon_precond_strength_init", "0.10",
                             "--newton_muon_precond_strength_min", "0.0",
                             "--newton_muon_precond_strength_max", "1.0",
                             "--newton_muon_precond_strength_hyper_lr", "10000",
                             "--newton_muon_precond_strength_hypergrad_clip", "0.0"],
        "adagrad_ema_muon": ["--opt", "muon-precond-gate", "--muon_precond_kind", "adagrad_ema",
                             "--muon_precond_beta", "0.95", "--muon_precond_eps", "1e-8",
                             "--gduo_gate_init", "0.10"],
        "soap_lite_muon":   ["--opt", "muon-precond-gate", "--muon_precond_kind", "soap_lite",
                             "--muon_precond_beta", "0.95", "--muon_precond_eps", "1e-8",
                             "--gduo_gate_init", "0.10"],
    }
    cmds = []
    for tag, extra in specs.items():
        prefix = f"wiki_geom_{tag}"
        cmd = gpt_base(prefix, args.gpt_results, args.datasets_dir, iters) + common_gate + list(extra)
        cmds.append((cmd, {"cwd": str(LLM), "env": env}))
    return cmds


def figure_cmds(args) -> list[tuple[list[str], dict]]:
    """Regenerate the paper figures. Each entry is (script, args, required input).

    The CIFAR figure can be regenerated from a local ``cifar-baselines`` run; the
    WikiText/meta figures depend on cluster CSVs fetched into ``izar_fetch/``.
    """
    # Prefer locally-produced CIFAR CSVs when present, else the fetched copies.
    local_cifar = Path(args.cifar_results)
    cifar_dir = local_cifar if list(local_cifar.glob("cifar10_resmlp32_*_seed0.csv")) \
        else REPO / "izar_fetch" / "results_cifar_fig1"

    py = [sys.executable]
    specs = [
        # (description, argv, input path that must exist)
        ("Fig 1  baselines (CIFAR + WikiText)",
         py + ["plot_paper_baselines_cifar_wikitext.py", "--cifar_dir", str(cifar_dir)],
         cifar_dir),
        ("Fig 1  CIFAR-only accuracy",
         py + ["plot_paper_curves.py", "--cifar_dir", str(cifar_dir)],
         cifar_dir),
        ("Fig 3  GD-UO scopes (global vs layerwise)",
         py + ["plot_paper_gduo_three_panels.py"],
         REPO / "izar_fetch" / "llm_newton_stability" / "diagnostic"
              / "gpt_wikitext_with_stable_newton_curves.csv"),
        ("Fig 4  learned layerwise LR + momentum",
         py + ["plot_paper_muon_layerwise_meta.py"],
         REPO / "izar_fetch" / "layerwise_3005859" / "results" / "gduo_layerwise_final.csv"),
        ("Geometry variants",
         py + ["plot_paper_geometry_variants.py"],
         REPO / "izar_fetch" / "recent_gating_precond"),
        ("Meta vs baselines",
         py + ["plot_meta_vs_baselines.py"],
         REPO / "izar_fetch"),
    ]
    cmds = []
    for desc, argv, required in specs:
        cmds.append((argv, {"cwd": str(REPO), "desc": desc, "required_input": required}))
    return cmds


STAGES = {
    "cifar-baselines":   lambda a: cifar_cmds(CIFAR_BASELINES, a),
    "cifar-meta":        lambda a: cifar_cmds(CIFAR_META, a),
    "wikitext-baselines": wikitext_baseline_cmds,
    "wikitext-dynamics": wikitext_dynamics_cmds,
    "wikitext-geometry": wikitext_geometry_cmds,
    "figures":           figure_cmds,
}

ALL_ORDER = [
    "cifar-baselines",
    "wikitext-baselines",
    "wikitext-dynamics",
    "wikitext-geometry",
    "figures",
]

STAGE_DELIVERABLE = {
    "cifar-baselines":   "Table 1 (CIFAR columns) + Figure 1 (top panel)",
    "cifar-meta":        "CIFAR GD-UO LR sanity runs (local appendix)",
    "wikitext-baselines": "Table 1 (WikiText columns) + Figure 1 (bottom panel)",
    "wikitext-dynamics": "Figure 3 + Figure 4 (LR + momentum, global & layerwise)",
    "wikitext-geometry": "Table 2 (Adam/Muon, Muon/Newton, AdaGrad-EMA, SOAP-lite)",
    "figures":           "All paper figures (regenerated from CSVs)",
    "all":               "Every stage above, in order",
}


def print_map() -> None:
    print("Stage -> report deliverable\n")
    for stage in [*ALL_ORDER[:-1], "cifar-meta", "figures", "all"]:
        print(f"  {stage:<20} {STAGE_DELIVERABLE[stage]}")
    print("\nRun a stage with:  python run.py <stage> [--quick] [--dry-run]")


def run_one(cmd: list[str], opts: dict, dry: bool) -> int:
    desc = opts.pop("desc", None)
    required = opts.pop("required_input", None)
    if desc:
        print(f"\n# {desc}")
    if required is not None and not Path(required).exists():
        print(f"  [skip] missing input: {required}")
        return 0
    printable = " ".join(shlex.quote(c) for c in cmd)
    cwd = opts.get("cwd", str(REPO))
    print(f"  $ (cd {cwd} && {printable})")
    if dry:
        return 0
    return subprocess.run(cmd, **opts).returncode


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("stage", nargs="?", choices=[*STAGES, "all"],
                   help="Which part of the pipeline to run.")
    p.add_argument("--list", action="store_true",
                   help="Print the stage -> report-deliverable map and exit.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the exact commands without executing them.")
    p.add_argument("--quick", action="store_true",
                   help="Short smoke test (few epochs / iterations).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100, help="CIFAR epochs (paper: 100).")
    p.add_argument("--gpt-iters", type=int, default=2000, help="WikiText iterations (paper: 2000).")
    p.add_argument("--cifar-results", default="results_cifar_fig1")
    p.add_argument("--gpt-results", default="results_wikitext")
    p.add_argument("--data-root", default="data", help="CIFAR-10 download dir.")
    p.add_argument("--datasets-dir", default="data/wikitext", help="WikiText cache dir.")
    args = p.parse_args()

    if args.list or args.stage is None:
        print_map()
        return 0

    stages = ALL_ORDER if args.stage == "all" else [args.stage]
    rc = 0
    for stage in stages:
        print(f"\n{'=' * 72}\n== STAGE: {stage}  ->  {STAGE_DELIVERABLE[stage]}\n{'=' * 72}")
        for cmd, opts in STAGES[stage](args):
            r = run_one(cmd, dict(opts), args.dry_run)
            if r != 0:
                print(f"  [warn] command exited with code {r}")
                rc = r
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
