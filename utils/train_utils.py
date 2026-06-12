import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PLOT_DATA = ROOT / "data" / "plot_inputs"
GPT_DIR = ROOT / "llm_utils" / "llm-baselines"


def read_plot_csv(name):
    return pd.read_csv(PLOT_DATA / name)


def add_runtime_args(parser, default_output_dir):
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--eval_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--multiple_of", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--muon_lr_factor", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--output_dir", default=default_output_dir)


def gpt_command(args, opt, extra_args=None):
    command = [
        sys.executable,
        "src/main.py",
        "--dataset",
        "wikitext",
        "--model",
        "llama",
        "--n_layer",
        str(args.n_layer),
        "--n_embd",
        str(args.n_embd),
        "--n_head",
        str(args.n_head),
        "--sequence_length",
        str(args.sequence_length),
        "--batch_size",
        str(args.batch_size),
        "--iterations",
        str(args.iterations),
        "--warmup_steps",
        str(args.warmup_steps),
        "--eval_interval",
        str(args.eval_interval),
        "--eval_batches",
        str(args.eval_batches),
        "--results_base_folder",
        str(args.output_dir),
        "--device",
        args.device,
        "--multiple_of",
        str(args.multiple_of),
        "--dtype",
        args.dtype,
        "--opt",
        opt,
        "--muon_lr_factor",
        str(args.muon_lr_factor),
        "--momentum",
        str(args.momentum),
    ]
    if extra_args:
        command.extend(extra_args)
    return command


def run_commands(commands, dry_run=False):
    for command in commands:
        print(" ".join(command))
        if not dry_run:
            subprocess.run(command, cwd=GPT_DIR, check=True)


def final_rows(df, group_cols, step_col="iter"):
    rows = []
    for keys, group in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        final = group.sort_values(step_col).iloc[-1]
        rows.append((*keys, final))
    return rows
