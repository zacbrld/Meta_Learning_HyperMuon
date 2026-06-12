import argparse
from pathlib import Path

from utils.plot_utils import make_geometry_variants_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Generate the geometry/preconditioner variant figure.")
    parser.add_argument("--recent_dir", default="izar_fetch/recent_gating_precond")
    parser.add_argument("--best_muon_layerwise", default="izar_fetch/current_meta_logs/gpt_layer_wiki_3005857_1.out")
    parser.add_argument("--output", default="figures/paper_like/figure3_geometry_variants")
    return parser.parse_args()


def main():
    args = parse_args()
    output, coverage = make_geometry_variants_figure(
        Path(args.recent_dir),
        Path(args.best_muon_layerwise),
        Path(args.output),
    )
    print(f"Saved {output}")
    print(f"Saved {Path(args.output).with_suffix('.coverage.csv')}")
    missing = coverage[~coverage["has_2000"]]
    if not missing.empty:
        print("Curves without 2000 steps:")
        for row in missing.itertuples(index=False):
            print(f"  {row.label}: max_iter={row.max_iter} source={row.source}")


if __name__ == "__main__":
    main()
