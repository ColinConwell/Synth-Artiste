"""Generate candidates and select in DreamSim embedding space."""
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from synthart import ImageGenerator, Similarity, generate_similar


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference', type=Path)
    p.add_argument('--prompt', default='An ink drawing of a potted fern')
    p.add_argument('--backend', default='diffusers', choices=['diffusers', 'openai', 'gemini'])
    p.add_argument('--candidates', type=int, default=4)
    p.add_argument('--output', type=Path, default=Path('outputs/dreamsim-conditioned'))
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env.local')
    generator, metric = ImageGenerator(args.backend), Similarity('dreamsim')
    result = generate_similar(generator, args.prompt, args.reference, metric=metric,
        candidates=args.candidates, seed=42 if args.backend == 'diffusers' else None)
    args.output.mkdir(parents=True, exist_ok=True)
    for i, image in enumerate(result.candidates):
        image.save(args.output / f'candidate-{i}.png')
    result.image.save(args.output / 'selected.png')
    (args.output / 'selection.json').write_text(json.dumps(dict(generator=generator.describe(),
        prompt=args.prompt, reference=str(args.reference), metric='dreamsim',
        distances=result.distances, selected_index=result.index), indent=2))
    print(args.output / 'selected.png')


if __name__ == '__main__':
    main()
