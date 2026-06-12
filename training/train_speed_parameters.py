import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.train_utils import add_runtime_args, final_rows, gpt_command, read_plot_csv, run_commands


OPTIMIZERS = {
    "adamw": ("adam", "adamw-gduo"),
    "muon": ("muon", "muon-gduo"),
    "newton-muon": ("newton", "newton-muon-gduo"),
}

SCOPES = {
    "global": "global",
    "layerwise": "tensor",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or replay LR+momentum meta-learning.")
    parser.add_argument("--optimizer", choices=["all", *OPTIMIZERS], default="all")
    parser.add_argument("--scope", choices=["all", *SCOPES], default="all")
    parser.add_argument("--include-baseline", action="store_true")
    add_runtime_args(parser, "results_speed_parameters")
    return parser.parse_args()


def selected_optimizers(name):
    if name == "all":
        return list(OPTIMIZERS)
    return [name]


def selected_scopes(name):
    if name == "all":
        return list(SCOPES)
    return [name]


def replay(args):
    df = read_plot_csv("gduo_scopes.csv")
    opt_keys = [OPTIMIZERS[name][0] for name in selected_optimizers(args.optimizer)]
    variants = selected_scopes(args.scope)
    if args.include_baseline:
        variants = ["baseline", *variants]
    df = df[df["optimizer"].isin(opt_keys) & df["variant"].isin(variants)]
    print("GPT/WikiText LR+momentum meta-learning")
    print("optimizer, variant, final_iter, val_loss")
    for optimizer, variant, final in final_rows(df, ["optimizer", "variant"]):
        print(f"{optimizer}, {variant}, {int(final['iter'])}, {final['val_loss']:.3f}")


def execute(args):
    commands = []
    for optimizer in selected_optimizers(args.optimizer):
        _, opt = OPTIMIZERS[optimizer]
        for scope in selected_scopes(args.scope):
            commands.append(
                gpt_command(
                    args,
                    opt,
                    [
                        "--gduo_learn_lr",
                        "--gduo_learn_momentum",
                        "--gduo_scope",
                        SCOPES[scope],
                    ],
                )
            )
    run_commands(commands, dry_run=args.dry_run)


def main():
    args = parse_args()
    if args.execute:
        execute(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
