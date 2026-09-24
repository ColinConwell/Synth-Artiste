"""Generate one image with text, an image prompt, a LoRA, or a flow map."""
import argparse
from pathlib import Path
from dotenv import load_dotenv
from synthart import ImageGenerator, FlowMapGenerator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['openai', 'gemini', 'diffusers', 'dmf'], default='openai')
    p.add_argument('--model')
    p.add_argument('--prompt', default='A botanical ink drawing of a fern in a glass vase')
    p.add_argument('--reference', type=Path)
    p.add_argument('--ip-adapter', action='store_true')
    p.add_argument('--lora')
    p.add_argument('--lora-weight')
    p.add_argument('--seed', type=int)
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--class-id', type=int, default=207)
    p.add_argument('--output', type=Path, default=Path('outputs/image.png'))
    args = p.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env.local')
    if args.backend == 'dmf':
        generator = FlowMapGenerator()
        image = generator.generate(class_id=args.class_id, seed=args.seed or 0, steps=args.steps)
    else:
        generator = ImageGenerator(args.backend, model=args.model, ip_adapter=args.ip_adapter,
                                   lora=args.lora, lora_weight=args.lora_weight)
        image = generator.generate(args.prompt, reference=args.reference, seed=args.seed, steps=args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(args.output)


if __name__ == '__main__':
    main()
