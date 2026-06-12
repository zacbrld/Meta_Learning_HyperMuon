import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PLOT_DATA = ROOT / "data" / "plot_inputs"
GPT_DIR = ROOT / "external" / "llm-baselines"


GPT_OPTIMIZERS = {
    "adamw": "adamw",
    "muon": "muon",
    "newton-muon": "newton-muon",
    "adam-muon": "adam-muon-gate",
    "muon-newton": "muon-newton-gate",
    "adagrad-ema": "muon-precond-gate",
    "soap-lite": "muon-precond-gate",
}

GPT_REPLAY_LABELS = {
    "adamw": ["AdamW"],
    "muon": ["Muon"],
    "newton-muon": ["Newton-Muon"],
    "adam-muon": ["Adam/Muon gate"],
    "muon-newton": ["Muon/Newton-Muon gate"],
    "adagrad-ema": ["AdaGrad-EMA/Muon"],
    "soap-lite": ["SOAP-lite/Muon"],
    "layerwise": ["Muon LR+momentum layerwise"],
}


def read_csv(name):
    path = PLOT_DATA / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def replay_cifar(optimizer):
    df = read_csv("cifar_baselines.csv")
    if optimizer != "all":
        optimizer = optimizer.replace("-", "_")
        df = df[df["optimizer"] == optimizer]
    if df.empty:
        raise ValueError(f"No CIFAR result for optimizer={optimizer}")

    rows = []
    for name, group in df.groupby("optimizer"):
        group = group.sort_values("step")
        final = group.iloc[-1]
        best = group.sort_values("eval_accuracy").iloc[-1]
        rows.append((name, int(final["step"]), 100.0 * final["eval_accuracy"], 100.0 * best["eval_accuracy"]))

    print("CIFAR-10 processed training results")
    print("optimizer, final_step, final_acc, best_acc")
    for name, step, final_acc, best_acc in rows:
        print(f"{name}, {step}, {final_acc:.2f}, {best_acc:.2f}")


def replay_gpt(optimizer):
    frames = []
    baseline = read_csv("wikitext_baselines.csv")
    baseline = baseline.rename(columns={"optimizer": "label"})
    baseline["source"] = "baseline"
    frames.append(baseline[["label", "iter", "val_loss", "val_acc", "source"]])

    geometry = read_csv("geometry_variants.csv")
    geometry = geometry.rename(columns={"label": "label"})
    geometry["source"] = "geometry"
    frames.append(geometry[["label", "iter", "val_loss", "val_acc", "source"]])

    df = pd.concat(frames, ignore_index=True)
    if optimizer != "all":
        labels = GPT_REPLAY_LABELS.get(optimizer)
        if labels is None:
            raise ValueError(f"Unknown GPT/WikiText replay optimizer={optimizer}")
        df = df[df["label"].isin(labels)]
    if df.empty:
        raise ValueError(f"No GPT/WikiText result matching optimizer={optimizer}")

    print("GPT/WikiText processed training results")
    print("variant, source, final_iter, val_loss, val_acc")
    for label, group in df.groupby("label"):
        final = group.sort_values("iter").iloc[-1]
        print(f"{label}, {final['source']}, {int(final['iter'])}, {final['val_loss']:.3f}, {final['val_acc']:.4f}")


def gpt_command(args):
    opt = GPT_OPTIMIZERS[args.optimizer]
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
        "--opt",
        opt,
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
    ]
    if args.optimizer in {"muon", "newton-muon", "adam-muon", "muon-newton", "adagrad-ema", "soap-lite"}:
        command += ["--muon_lr_factor", str(args.muon_lr_factor), "--momentum", str(args.momentum)]
    if args.optimizer == "adagrad-ema":
        command += ["--muon_precond_kind", "adagrad_ema"]
    if args.optimizer == "soap-lite":
        command += ["--muon_precond_kind", "soap_lite"]
    return command


def execute_gpt(args):
    command = gpt_command(args)
    print("Running GPT/WikiText training:")
    print(" ".join(command))
    subprocess.run(command, cwd=GPT_DIR, check=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Training entry point for the submission.")
    parser.add_argument("--task", choices=["cifar", "gpt"], required=True)
    parser.add_argument("--optimizer", default="all")
    parser.add_argument("--execute", action="store_true")
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
    parser.add_argument("--muon_lr_factor", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--output_dir", default="results_train")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.task == "cifar":
        if args.execute:
            raise SystemExit("CIFAR full training is not included in the minimal submission. Use the processed CSV replay.")
        replay_cifar(args.optimizer)
    elif args.task == "gpt":
        if args.execute:
            if args.optimizer == "all":
                raise SystemExit("--execute needs a single GPT optimizer.")
            if args.optimizer not in GPT_OPTIMIZERS:
                raise SystemExit(f"Unknown GPT optimizer: {args.optimizer}")
            execute_gpt(args)
        else:
            replay_gpt(args.optimizer)


if __name__ == "__main__":
    main()
