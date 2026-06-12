import argparse
from pathlib import Path

from utils.plot_utils import (
    make_baseline_figure,
    make_gduo_scope_figure,
    make_geometry_variants_figure,
    make_muon_layerwise_figure,
)


PLOTS = {
    "baselines": make_baseline_figure,
    "gduo": make_gduo_scope_figure,
    "geometry": make_geometry_variants_figure,
    "layerwise": make_muon_layerwise_figure,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the paper figures from fetched experiment logs.")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["all", *PLOTS.keys()],
        default=["all"],
    )
    return parser.parse_args()


def selected_plots(names):
    if "all" in names:
        return PLOTS
    return {name: PLOTS[name] for name in names}


def main(plot_names=None):
    names = plot_names if plot_names is not None else parse_args().plots
    for name, make_plot in selected_plots(names).items():
        result = make_plot()
        output = result[0] if isinstance(result, tuple) else result
        print(f"{name}: {Path(output)}")


if __name__ == "__main__":
    main()
