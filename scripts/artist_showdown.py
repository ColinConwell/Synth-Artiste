"""Generate matched portfolios and evaluate original-space similarity."""
import argparse
from pathlib import Path
from dotenv import load_dotenv
from synthart.experiment import make_portfolios, evaluate_portfolios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/showdown'))
    parser.add_argument('--backends', nargs='+', choices=['openai', 'gemini'], default=['openai', 'gemini'])
    parser.add_argument('--candidates', type=int, default=2)
    parser.add_argument('--metrics', nargs='+', choices=['dreamsim', 'lpips', 'tpips'], default=['dreamsim', 'lpips', 'tpips'])
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env.local')
    if not args.evaluate_only:
        make_portfolios(args.output, backends=args.backends, candidates=args.candidates)
    print(evaluate_portfolios(args.output, metrics=args.metrics))


if __name__ == '__main__':
    main()
