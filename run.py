import argparse

from generate_plots import main as generate_plots


def parse_args():
    parser = argparse.ArgumentParser(description="Regenerate the paper figures.")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["all", "baselines", "gduo", "geometry", "layerwise"],
        default=["all"],
    )
    return parser.parse_args()


def main():
    args = parse_args()
    generate_plots(args.plots)


if __name__ == "__main__":
    main()
