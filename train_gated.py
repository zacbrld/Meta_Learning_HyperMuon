import argparse

from utils.train_utils import add_runtime_args, final_rows, gpt_command, read_plot_csv, run_commands


VARIANTS = {
    "adam-muon": ("Adam/Muon gate", "adam-muon-gate", []),
    "muon-newton": (
        "Muon/Newton-Muon gate",
        "muon-newton-gate",
        ["--newton_muon_precond_strength_init", "0.1"],
    ),
    "adagrad-ema": (
        "AdaGrad-EMA/Muon",
        "muon-precond-gate",
        ["--muon_precond_kind", "adagrad_ema", "--newton_muon_precond_strength_init", "0.1"],
    ),
    "soap-lite": (
        "SOAP-lite/Muon",
        "muon-precond-gate",
        ["--muon_precond_kind", "soap_lite", "--newton_muon_precond_strength_init", "0.1"],
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or replay learned geometry/gating experiments.")
    parser.add_argument("--variant", choices=["all", *VARIANTS], default="all")
    parser.add_argument("--no-reference", action="store_true")
    add_runtime_args(parser, "results_gated")
    return parser.parse_args()


def selected_variants(name):
    if name == "all":
        return list(VARIANTS)
    return [name]


def replay(args):
    df = read_plot_csv("geometry_variants.csv")
    labels = [VARIANTS[name][0] for name in selected_variants(args.variant)]
    if not args.no_reference:
        labels.append("Muon LR+momentum layerwise")
    df = df[df["label"].isin(labels)]
    print("GPT/WikiText learned geometry and gating")
    print("variant, final_iter, val_loss, val_acc")
    for label, final in final_rows(df, ["label"]):
        print(f"{label}, {int(final['iter'])}, {final['val_loss']:.3f}, {final['val_acc']:.4f}")


def execute(args):
    commands = []
    for variant in selected_variants(args.variant):
        _, opt, extra = VARIANTS[variant]
        commands.append(gpt_command(args, opt, extra))
    run_commands(commands, dry_run=args.dry_run)


def main():
    args = parse_args()
    if args.execute:
        execute(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
