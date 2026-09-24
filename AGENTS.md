# Synth-Artiste Agent Guide

## Purpose

This repository unifies image generation and comparison through short, intuitive workflows, designed for pedagogy and experimentation. Keep public calls in `synthart/` conceptually unified and hide model-specific setup. `ImageGenerator` handles text and image conditioning; `FlowMapGenerator` is explicitly class-conditioned. `Similarity` returns distances where lower means more similar. `generate_similar` performs best-of-N selection, not gradient or IP-Adapter conditioning.

Keep the root README focused on installation and short examples. Put literature, limitations, and detailed verification evidence in `docs/`.

## Directory Structure

```
.
├── README.md
├── pyproject.toml
├── requirements.txt
├── LICENSE
├── docs/
│   ├── literature.md
│   ├── validation.md
│   ├── tested-environment.json
│   └── notebook-validation.json
├── notebooks/
│   ├── conditional-generation.ipynb
│   ├── similarity-evaluation.ipynb
│   └── synth-artiste-showdown.ipynb
├── scripts/
│   ├── generate_image.py
│   ├── dreamsim_conditioned.py
│   ├── compare_images.py
│   ├── artist_showdown.py
│   ├── generate_samples.py
│   ├── plan_and_generate.py
│   └── validate_notebooks.py
├── synthart/
│   ├── __init__.py
│   ├── generation.py
│   ├── similarity.py
│   ├── conditioning.py
│   ├── flow_maps.py
│   ├── spaces.py
│   ├── experiment.py
│   ├── images.py
│   ├── plotting.py
│   ├── _dmf_loader.py
│   ├── artist.py
│   ├── config.py
│   ├── generator.py
│   ├── openai_image.py
│   ├── embeddings.py
│   ├── evaluator.py
│   ├── sampler.py
│   ├── utils.py
│   └── backends/
│       ├── base.py
│       └── openai_backend.py
└── testing/
    ├── test_workflows.py
    └── test_*.py
```

Gitignored runtime paths are `.env.local`, `.venv/`, `outputs/`, and `models/`. The DMF architecture cache defaults to `~/.cache/synthart/` and moves with `SYNTHART_CACHE`.

## Usage

Install with `pip install -e '.[vision,notebooks,test]'`. API-only work can use `pip install -e .`. Load `OPENAI_API_KEY` and `GEMINI_API_KEY` from `.env.local` with `load_dotenv('.env.local')` before provider calls. Scripts already load that file. Do not print, log, or commit key values.

The current public surface is exported from `synthart`:

| Call | Role |
| --- | --- |
| `ImageGenerator(backend)` | Text generation, native image edits, Stable Diffusion image-to-image, IP-Adapter, and LoRA inference. Backends are `openai`, `gemini`, and `diffusers`. |
| `generate_similar(...)` | Best-of-N selection by a `Similarity` distance to a reference. |
| `Similarity(name)` | `dreamsim`, `lpips`, or factor-conditioned `tpips`. Use `distance`, `pairwise`, and, where supported, `embed`. |
| `FlowMapGenerator()` | Class-conditioned Decoupled MeanFlow sampling. Text prompts are rejected. |
| `project` / `portfolio_overlap` | Display projections and original-space overlap summaries. |

`SyntheticArtist`, `ImageConfig`, and `generate_dataset` remain supported for the earlier OpenAI dataset workflow.

Short examples belong in `README.md`. The notebooks are the full tours: conditional generation, similarity evaluation, and the two-artist showdown. Matching commands live in `scripts/generate_image.py`, `scripts/dreamsim_conditioned.py`, `scripts/compare_images.py`, and `scripts/artist_showdown.py`. `scripts/generate_samples.py` and `scripts/plan_and_generate.py` keep the earlier sample and planning workflows. Generated images, prompts, scores, and executed notebooks go under `outputs/`.

Repeated exact image requests should reuse the sidecar cache in `synthart.experiment.cached_image`. Cache keys must stay sensitive to generation settings and reference content.

## Package Organization

`generation.py`, `similarity.py`, `conditioning.py`, `flow_maps.py`, `spaces.py`, `experiment.py`, `images.py`, and `plotting.py` implement the unified generate, select, and compare workflows. Model construction stays inside those classes so `import synthart` does not initialize PyTorch or download weights.

`_dmf_loader.py` downloads two hash-verified architecture files from a pinned upstream commit and imports them in an isolated module. It uses the native SDPA inference path.

`artist.py`, `config.py`, `generator.py`, `openai_image.py`, `embeddings.py`, `evaluator.py`, `sampler.py`, `utils.py`, and `backends/` support the earlier artist-prompt dataset pipeline used by `generate_samples.py` and `plan_and_generate.py`. Preserve those interfaces when changing the newer generators.

`testing/test_workflows.py` is the isolated pytest suite. The other `testing/test_*.py` files are legacy, manually run API debugging scripts. `docs/literature.md` holds method choices, limits, and the DMF licensing note. `docs/validation.md` records live script commands, notebook results, and the tested environment.

## Implementation Conventions

Use optional, lazy imports for model dependencies. Importing `synthart` must not initialize PyTorch or download weights. Preserve the earlier `SyntheticArtist`, `ImageConfig`, and `generate_dataset` interfaces. Do not silently switch model families, replace pretrained models with random networks, or skip a notebook section after an error. Record model IDs, prompts, factors, seeds where supported, candidate scores, and paths. Keep cache keys sensitive to generation settings and reference content.

Use title case for headings that are not complete questions or sentences. Scientific claims must state the question, operational measure, result, and limitations. Do not infer human perceptual validity from a visually separated 2D scatterplot. Explain metric selection bias and separate provider effects from artist effects.

## Setup and Verification

Install `pip install -e '.[vision,notebooks,test]'`. On BigRAM use the CUDA 12.8 PyTorch wheels compatible with its driver (the tested pair is torch 2.8.0 / torchvision 0.23.0). TPIPS requires Transformers 5 and PEFT 0.19 or newer. Keep model-version compatibility in mind when refreshing dependencies; record a tested environment in `docs/`.

Run `python -m pytest -q` for isolated logic tests, and `python scripts/validate_notebooks.py` for full fresh-kernel execution. The notebooks make real model calls; generation notebooks also make paid API calls. Reuse exact cached requests when rerunning. Record live script commands and notebook results in `docs/validation.md`. `testing/` contains legacy, manually run API debugging scripts, not the isolated pytest suite. Do not accidentally collect them in offline CI.

Use `ssh bigram` for large models. The task's isolated remote environment is `~/synthart-lab/.venv`, with checkout `~/synthart-lab/repo`. Do not modify other workloads. Keys are in ignored `.env.local`; load without printing, logging, or committing values. Never synchronize `.env*` indiscriminately. Model artifacts and generated outputs are ignored.

The DMF loader downloads two hash-verified files from a pinned upstream commit and uses its native SDPA inference path. Do not replace this with `trust_remote_code=True`, unpinned code execution, or unconditional FlashAttention installation. The upstream revision has no root license file; do not redistribute its source under our license.

## Lingering Issues

Pytest is configured with `testpaths = ["testing"]`. Only `testing/test_workflows.py` is the offline suite. Other files in that directory define manual `test_*` entry points and can be collected if a run is not narrowed.

Hosted OpenAI and Gemini image calls do not accept a reproducible seed in this wrapper. Local Diffusers and DMF generation do. The DMF checkpoint must load strictly, including its log-variance head. Transformers 5 returns CLIP features as `BaseModelOutputWithPooling`; the legacy embedder reads `pooler_output`.

TPIPS is an 8B model and was validated on CUDA. MPS was not part of the recorded validation. FMRG reward guidance, free-text flow maps, and LoRA training are outside the implemented path. ControlNet, DreamBooth, and textual inversion are discussed in the literature guide and are not tested here. The showdown portfolio is a small descriptive teaching example.

## Installed Product Identity

When creating or changing installed applications, packages, background services, permission flows, icons, or OS notifications, read and follow [Professional App and Service Identity](instructions/professional-installed-identity.md). The rule requires product-branded system identifiers, appropriate light/dark icons, native notifications, and verification of actual installed behavior. This repository is a Python research library, not an installed desktop app or service.
