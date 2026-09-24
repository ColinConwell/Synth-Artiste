# Validation Record

Validation was performed on September 24, 2026. Live generation used the supplied OpenAI and Gemini credentials without logging their values. Local model inference ran on `bigram` with an NVIDIA RTX 4090 and CUDA-compatible PyTorch. The exact environment is recorded in [outputs/tested-environment.json](../outputs/tested-environment.json).

## Automated Checks

Eight isolated regression tests passed on macOS (Python 3.11) and BigRAM (Python 3.10). They verify candidate selection, seed progression, cache invalidation, distance-matrix validation, self-neighbor exclusion, projection output, unsupported-option rejection, and the older positional `ImageConfig` interface. Importing `synthart` was also checked to leave PyTorch unloaded. `pip check` on BigRAM reported no broken requirements.

## Live Scripts

Every script in `scripts/` was exercised with real providers or pretrained weights:

| Script | Executed Configuration | Result |
| --- | --- | --- |
| `generate_image.py` | OpenAI Flare, OpenAI Sunburst, Nano Banana 2, Nano Banana 2 Lite; DMF at four steps. | Images saved successfully. |
| `dreamsim_conditioned.py` | Stable Diffusion, two candidates, real DreamSim embeddings. | Both candidates, selected image, and score manifest saved. |
| `compare_images.py` | LPIPS on the generated reference and selected image. | Finite symmetric matrix saved; off-diagonal distance was 0.6951. |
| `artist_showdown.py` | Both providers, two artists, three subjects, two candidates; DreamSim and LPIPS evaluation. | 28 generated images, 12 selected works, candidate manifests, and distance summaries saved. TPIPS is additionally exercised by the showdown notebook. |
| `generate_samples.py` | One subject per artist, low quality. | Both legacy artist images saved. |
| `plan_and_generate.py` | One planned subject per artist, low quality, CLIP enabled. | Generation, embedding-based planning, CLIP evaluation, and manifest succeeded; `metrics_error` is null. |
| `validate_notebooks.py` | Fresh kernels with all real code cells enabled. | See the notebook execution record below. |

The two earlier scripts now accept a quality setting and the sample script accepts `--num-per-artist`, allowing small live checks without a full default run. No paid provider calls occur in the isolated pytest suite.

## Notebook Execution

The final validation command is:

```bash
CUDA_VISIBLE_DEVICES=1 ~/synthart-lab/.venv/bin/python scripts/validate_notebooks.py
```

The validator executes each notebook from top to bottom, fails on any cell exception, saves the executed notebook, and closes the kernel. Existing exact image requests are cached; candidate generation in the DreamSim tutorial and metric inference still run. The first full generation run executed native edits, SD image-to-image, IP-Adapter, LCM-LoRA, DreamSim selection, and one- and four-step DMF inference.

| Notebook | Executed Code Cells | Final Cached-Run Seconds | Status |
| --- | ---: | ---: | --- |
| `conditional-generation.ipynb` | 8 | 23.1 | Passed |
| `similarity-evaluation.ipynb` | 6 | 14.3 | Passed |
| `synth-artiste-showdown.ipynb` | 5 | 41.8 | Passed |

The final pass completed all **19 code cells** with no skipped sections or cell errors.

Executed outputs and the machine-readable report are in `outputs/executed-notebooks/`. The source notebooks retain clean runnable cells. Gallery images were inspected for visible outputs; no model was replaced by a mock. The similarity tutorial uses explicitly constructed geometric stimuli, not fake metrics.

## Issues Found and Corrected

- The newest default PyTorch wheel required a newer CUDA driver. Installing torch 2.8.0 and torchvision 0.23.0 from the CUDA 12.8 index restored GPU inference.
- The DMF checkpoint includes a log-variance head. Enabling the matching architecture head allows strict weight loading; unmatched weights are not silently ignored.
- Transformers 5 returns CLIP features in `BaseModelOutputWithPooling`. The legacy embedder now extracts its projected `pooler_output`, retaining older tensor support.
- Notebook figures are explicitly rendered inline, and the validator shuts down each kernel even when execution fails.

## Scope of the Evidence

These checks establish execution and artifact correctness, not generation superiority or validated human perception. FMRG's FLUX reward-guidance system was not run; the tested flow-map implementation is class-conditioned Decoupled MeanFlow. No LoRA was trained. MPS inference was not exercised in this validation; local models were tested on CUDA. The portfolio experiment is deliberately small and descriptive, with no significance claims.
