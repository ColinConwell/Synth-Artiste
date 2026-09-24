# Synth-Artiste

Toolkit for conditional image generation, image comparison, and "synthetic artistry".

Generate images under explicit conditions, compare their perceptual similarity, and study how two simulated artistic styles overlap. The library presents three operations: **generate**, **select by similarity**, and **compare**.

## Installation

Use Python 3.10 or newer (3.11 recommended):

```bash
python -m venv .venv & source .venv/bin/activate
pip install -e '.[vision,notebooks,test]'
```

For API generation alone, `pip install -e .` avoids PyTorch and model downloads. Store `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env.local`; scripts load that file. In Python, call `load_dotenv('.env.local')` from `python-dotenv` before provider use. Do not commit credentials.

Local models download their weights on first use. Stable Diffusion, DreamSim, and LPIPS can use CPU, Apple MPS, or CUDA. TPIPS's 8B backbone is best run on CUDA. The flow-map example uses a 256px ImageNet checkpoint and native PyTorch attention.

On `bigram`, install the driver-compatible PyTorch build before the remaining extras:

```bash
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[vision,notebooks,test]'
```

## Generate Images

```python
from dotenv import load_dotenv
from synthart import ImageGenerator

load_dotenv('.env.local') # env file populated with API keys
artist = ImageGenerator('openai')  # GPT-Image 2.5 Flare
image = artist.generate('A botanical ink drawing of a fern in a glass vase')
image.save('fern.png')
```

Use `ImageGenerator('gemini')` for Nano Banana 2. Explicit model IDs select `gpt-image-2.5-sunburst`, `gemini-3-pro-image`, or `gemini-3.1-flash-lite-image`. The provider defaults were verified on September 24, 2026; availability can change. Pass `reference='fern.png'` to edit with either provider. Hosted seeds are unsupported.

```python
artist = ImageGenerator('diffusers', ip_adapter=True)
image = artist.generate('A fern in a ceramic bowl', reference='fern.png', seed=42)
```

Without `ip_adapter=True`, a reference requests Stable Diffusion image-to-image. Load a compatible LoRA with `lora='organization/adapter'` and optionally `lora_weight='weights.safetensors'`. The conditional-generation notebook demonstrates LCM-LoRA with the matching scheduler. LoRA training is outside this inference tour.

## Select With DreamSim Embeddings

```python
from synthart import Similarity, generate_similar

artist = ImageGenerator('openai')
metric = Similarity('dreamsim')
result = generate_similar(artist, 'A botanical ink drawing', 'fern.png',
                         metric=metric, candidates=4)
result.image.save('selected.png')
```

DreamSim selects the closest candidate in its embedding space; it does not inject embeddings into the generator. The selected image, every candidate, and all scores are available on the result.

## Compare Images

```python
from synthart import Similarity

distance = Similarity('lpips').distance('fern.png', 'selected.png')
rdm = Similarity('dreamsim').pairwise(['fern.png', 'selected.png'])
style_distance = Similarity('tpips', factor='artistic style').distance('fern.png', 'selected.png')
```

Lower distances mean more similar. DreamSim and TPIPS also expose `embed(images)`. LPIPS compares spatial features; this wrapper explicitly resizes to 256×256. TPIPS changes its representation with the requested factor. Metric scales are not interchangeable.

## Sample a Flow Map

```python
from synthart import FlowMapGenerator

image = FlowMapGenerator().generate(class_id=207, steps=4, seed=42)
image.save('golden-retriever.png')
```

Decoupled MeanFlow uses a pretrained interval-velocity model from Hugging Face. It is class-conditioned, not a text-conditioned FMRG implementation. The loader downloads two pinned, hash-checked architecture files and pretrained weights. See the [literature guide](docs/literature.md) for the FMRG comparison and source licensing note.

## Notebooks

| Notebook | Workflow |
| --- | --- |
| [Conditional Generation](notebooks/conditional-generation.ipynb) | Text, native image editing, Stable Diffusion, IP-Adapter, LoRA, DreamSim selection, and flow maps. |
| [Similarity Evaluation](notebooks/similarity-evaluation.ipynb) | LPIPS, DreamSim, aspect-conditioned TPIPS, and representational similarity analysis. |
| [Synth-Artiste Showdown](notebooks/synth-artiste-showdown.ipynb) | Matched portfolios from two providers, similarity selection, original-space overlap, and MDS/t-SNE/UMAP views. |

The generation tour makes four paid API image calls; the full showdown makes 28 on a fresh run. Repeated exact requests reuse saved images. The similarity tutorial makes no paid calls. None silently substitutes mock inference for a missing model.

## Scripts and Verification

```bash
python scripts/generate_image.py --backend openai --output outputs/fern.png
python scripts/dreamsim_conditioned.py outputs/fern.png --candidates 3
python scripts/compare_images.py outputs/fern.png outputs/dreamsim-conditioned/selected.png --metric lpips
python scripts/artist_showdown.py
python -m pytest -q
python scripts/validate_notebooks.py
```

Outputs, prompts, selection scores, distance matrices, and executed notebooks are saved under `outputs/`. [Validation notes](docs/validation.md) report what was actually run. Earlier `generate_samples.py` and `plan_and_generate.py` workflows remain available.

## References

- [DreamSim](https://dreamsim-nights.github.io/): human-aligned holistic image similarity.
- [LPIPS](https://richzhang.github.io/PerceptualSimilarity/): learned patch-level perceptual similarity.
- [TPIPS](https://peterwang512.github.io/TPIPS/): text-prompted image perceptual similarity.
- [IP-Adapter](https://ip-adapter.github.io/): image and text conditioning through separate attention.
- [Flow-Map Reward Guidance](https://arxiv.org/abs/2604.27147) and [Decoupled MeanFlow](https://github.com/kyungmnlee/dmf).

The [literature guide](docs/literature.md) connects these methods to cognitive science, explains the dimensionality-reduction choices, and records the research search.
