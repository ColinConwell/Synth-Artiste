"""Save a perceptual distance matrix for an image collection."""
import argparse
from pathlib import Path
import json
import numpy as np
from synthart import Similarity


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('images', nargs='+', type=Path)
    p.add_argument('--metric', choices=['dreamsim', 'lpips', 'tpips'], default='dreamsim')
    p.add_argument('--factor', default='overall')
    p.add_argument('--output', type=Path, default=Path('outputs/distances.npy'))
    args = p.parse_args()
    metric = Similarity(args.metric, factor=args.factor)
    distances = metric.pairwise(args.images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, distances)
    args.output.with_suffix('.json').write_text(json.dumps(dict(metric=args.metric,
        factor=args.factor, images=[str(p) for p in args.images]), indent=2))
    print(distances)


if __name__ == '__main__':
    main()
