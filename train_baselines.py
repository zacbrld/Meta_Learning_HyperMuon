import argparse

from utils.train_utils import add_runtime_args, final_rows, gpt_command, read_plot_csv, run_commands


GPT_BASELINES = {
    "adamw": ("AdamW", "adamw"),
    "muon": ("Muon", "muon"),
    "newton-muon": ("Newton-Muon", "newton-muon"),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run or replay fixed baselines.")
    parser.add_argument("--task", choices=["gpt", "cifar", "all"], default="all")
    parser.add_argument("--optimizer", choices=["all", *GPT_BASELINES], default="all")
    add_runtime_args(parser, "results_baselines")
    return parser.parse_args()


def selected_optimizers(name):
    if name == "all":
        return list(GPT_BASELINES)
    return [name]


def replay_gpt(args):
    df = read_plot_csv("wikitext_baselines.csv")
    labels = [GPT_BASELINES[name][0] for name in selected_optimizers(args.optimizer)]
    df = df[df["optimizer"].isin(labels)]
    print("GPT/WikiText fixed baselines")
    print("optimizer, final_iter, val_loss, val_acc")
    for optimizer, final in final_rows(df, ["optimizer"]):
        print(f"{optimizer}, {int(final['iter'])}, {final['val_loss']:.3f}, {final['val_acc']:.4f}")


def replay_cifar(args):
    df = read_plot_csv("cifar_baselines.csv")
    if args.optimizer != "all":
        df = df[df["optimizer"] == args.optimizer.replace("-", "_")]
    print("CIFAR-10 fixed baselines")
    print("optimizer, final_step, final_acc, best_acc")
    for optimizer, group in df.groupby("optimizer"):
        final = group.sort_values("step").iloc[-1]
        best = group.sort_values("eval_accuracy").iloc[-1]
        print(
            f"{optimizer}, {int(final['step'])}, "
            f"{100.0 * final['eval_accuracy']:.2f}, {100.0 * best['eval_accuracy']:.2f}"
        )


def execute_gpt(args):
    commands = []
    for name in selected_optimizers(args.optimizer):
        _, opt = GPT_BASELINES[name]
        commands.append(gpt_command(args, opt))
    run_commands(commands, dry_run=args.dry_run)


def main():
    args = parse_args()
    if args.execute:
        if args.task != "gpt":
            raise SystemExit("--execute is available for GPT/WikiText only.")
        execute_gpt(args)
        return
    if args.task in {"gpt", "all"}:
        replay_gpt(args)
    if args.task in {"cifar", "all"}:
        replay_cifar(args)


if __name__ == "__main__":
    main()

