import argparse
from pathlib import Path

from utils.plot_utils import make_gduo_scope_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the GD-UO scope comparison figure.")
    parser.add_argument(
        "--baseline_curves",
        default="izar_fetch/llm_newton_stability/diagnostic/gpt_wikitext_with_stable_newton_curves.csv",
    )
    parser.add_argument("--output", default="figures/paper_like/figure2_gduo_scopes")
    return parser.parse_args()


def main():
    args = parse_args()
    output, coverage = make_gduo_scope_figure(Path(args.baseline_curves), Path(args.output))
    print(f"Saved {output}")
    print(f"Saved {Path(args.output).with_suffix('.coverage.csv')}")
    missing = coverage[~coverage["has_2000"]]
    if not missing.empty:
        print("Curves without 2000 steps:")
        for row in missing.itertuples(index=False):
            print(f"  {row.optimizer}: {row.label} max_iter={row.max_iter}")


if __name__ == "__main__":
    main()
