import argparse
from pathlib import Path

from utils.plot_utils import make_muon_layerwise_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the Muon layerwise LR/momentum figure.")
    parser.add_argument("--final_csv", default="izar_fetch/layerwise_3005859/results/gduo_layerwise_final.csv")
    parser.add_argument("--output", default="figures/paper_like/figure4_muon_layerwise_lr_momentum")
    return parser.parse_args()


def main():
    args = parse_args()
    output, _ = make_muon_layerwise_figure(Path(args.final_csv), Path(args.output))
    print(f"Saved {output}")
    print(f"Saved {Path(args.output).with_suffix('.final.csv')}")


if __name__ == "__main__":
    main()
