import argparse
from pathlib import Path

from utils.plot_utils import make_baseline_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the CIFAR-10 and WikiText baseline figure.")
    parser.add_argument("--cifar_dir", default="izar_fetch/results_cifar_fig1")
    parser.add_argument(
        "--gpt_curves",
        default="izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv",
    )
    parser.add_argument("--output", default="figures/paper_like/figure1_cifar_wikitext_baselines")
    return parser.parse_args()


def main():
    args = parse_args()
    output = make_baseline_figure(Path(args.cifar_dir), Path(args.gpt_curves), Path(args.output))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
