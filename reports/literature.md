# Literature and Design Rationale

## Research Search

The literature and provider documentation were checked on **September 24, 2026** with web search, Elicit, and Consensus. Elicit was queried for “DreamSim LPIPS TPIPS human perceptual similarity”; Consensus was queried for “DreamSim Learning New Dimensions Human Visual Similarity Synthetic Data.” The DreamSim result was fetched before use. Primary project pages, papers, model cards, and official API documentation were then checked for the actual inference interfaces.

Elicit returned LPIPS (2018; 19,544 indexed citations), DreamSim (2023; 377), and TPIPS (2026; zero in that index at lookup). Consensus reported 387 for DreamSim. These counts are dated search-index observations, not comparable quality scores; TPIPS is included for its relevant new capability, not an established citation record. The [retrieved DreamSim record](https://consensus.app/papers/details/fe66b0e577825e58840ff6830d6af9d2/) provides an additional route to the paper. This was a focused implementation review, not a systematic review or an exhaustive SOTA benchmark.

## Human Visual Similarity

| Work | Research Question and Operational Measure | Relevance and Limits |
| --- | --- | --- |
| [Zhang et al., 2018: LPIPS](https://richzhang.github.io/PerceptualSimilarity/) | Do deep features predict human choices between distorted image patches? The work compares feature distances with perceptual judgments. | The default here is the learned AlexNet metric. Spatial alignment and resizing affect the score; it is not a general semantic embedding. |
| [Fu et al., 2023: DreamSim](https://dreamsim-nights.github.io/) | Can a representation capture holistic visual judgments beyond patch distortions? The model is tuned using judgments on synthetic image triplets and evaluated on additional tasks. | It motivates the shared embedding and best-of-N example. Similarity includes semantic and visual content; it is not a pure style score. |
| [Wang et al., 2026: TPIPS](https://peterwang512.github.io/TPIPS/) | Can a text-specified aspect change which images count as similar? The work evaluates aspect-conditioned predictions against human judgments. | The embedding model gives reusable image–factor embeddings. “Color,” “shape,” and “artistic style” are distinct operational questions. The release is recent and uses a large Qwen backbone. |
| [Hebart et al., 2020](https://www.nature.com/articles/s41562-020-00951-3) | Which dimensions explain human object-similarity judgments? A data-driven model estimates a multidimensional representation from behavioral choices. | This motivates explicit factors and behavioral validation. A model embedding should not be equated with the dimensions recovered from people. |
| [Bau et al., 2017: Network Dissection](https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html) | How do hidden units align with named visual concepts? Unit activations are compared with concept annotations. | Interpretability is a separate question from image similarity. The work motivates asking which attributes a representation captures rather than treating a scalar as an explanation. |

The principal maintained implementations used here are [LPIPS](https://github.com/richzhang/PerceptualSimilarity), [DreamSim](https://github.com/ssundaram21/dreamsim), and [Adobe Research TPIPS](https://github.com/adobe-research/TPIPS). The PeterWang512 repository hosts the TPIPS project page, not the Python package.

## Conditioning Mechanisms

| Mechanism | What Changes | Implementation |
| --- | --- | --- |
| Text conditioning | The semantic condition supplied to generation. | GPT-Image 2.5, Nano Banana 2, and Stable Diffusion. |
| Image-to-image | The initial noised latent and its denoising trajectory. | Diffusers pipeline conversion with explicit strength. |
| [IP-Adapter](https://ip-adapter.github.io/) | An additional learned image-attention condition. | Official `h94/IP-Adapter` weights through [Diffusers](https://huggingface.co/docs/diffusers/using-diffusers/ip_adapter). |
| [LoRA](https://arxiv.org/abs/2106.09685) | Low-rank weight updates. | Generic adapter loading and weighting; [LCM-LoRA](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) supplies a tested acceleration example with its required scheduler. |
| Best-of-N | Which generated candidate is retained. | DreamSim embedding distance to a reference; every candidate remains available. |
| Flow-map sampling | The learned transport over a finite time interval. | Pretrained Decoupled MeanFlow with an ImageNet class condition. |

[Latent diffusion](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) provides the basis for the compact Stable Diffusion baseline. Other useful controls are [ControlNet](https://github.com/lllyasviel/ControlNet) for spatial structure, [DreamBooth](https://dreambooth.github.io/) for personalized subjects, and [textual inversion](https://textual-inversion.github.io/) for learned token embeddings. They are discussed as extensions; this implementation does not claim tested support for them.

A DreamSim embedding cannot be inserted into an IP-Adapter trained for CLIP embeddings without learning a compatible mapping. Our example instead uses the embedding as a selection condition. Similarly, LoRA inference is not LoRA training, and a distilled few-step generator is not automatically a flow map.

## Flow Maps and the Accessible Alternative

[How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance](https://arxiv.org/abs/2604.27147) uses flow maps in a training-free reward-guidance framework. Its [official implementation](https://github.com/jrrhuang/fmrg) builds on a two-timestep FLUX model and flow-map LoRAs. The documented 512px reward-guidance configuration requires roughly 48 GB VRAM with gradient checkpointing. Two 24 GB GPUs do not automatically form one 48 GB device.

[Decoupled MeanFlow](https://arxiv.org/abs/2510.24474) offers a smaller way to study the core finite-interval map. We use its [official Hugging Face checkpoint](https://huggingface.co/kyungmnlee/DMF) and [upstream architecture](https://github.com/kyungmnlee/dmf), with native PyTorch SDPA. The library loads the 256px model and samples in one or several steps. This is genuinely a flow-map technique but **does not implement FMRG, free-text conditioning, or reward gradients**. Those distinctions are deliberate and visible in the API and notebooks.

The wrapper downloads only two architecture files at pinned revision `a6db0f6135e18f4a5dad6f38df3947ed61960742`, verifies SHA-256 hashes, and replaces unused FlashAttention training imports. It preserves the SDPA architecture and strictly loads the checkpoint, including its uncertainty head. The upstream revision has no root license file; its source is not redistributed in this repository. Review upstream terms before reuse beyond research. `SYNTHART_CACHE` changes the architecture cache location.

## Representational Spaces

[Representational similarity analysis](https://www.frontiersin.org/journals/systems-neuroscience/articles/10.3389/neuro.06.004.2008/full) compares representations through their pairwise dissimilarities. The tutorials use Spearman agreement between upper-triangular RDM entries as a descriptive statistic. They do not treat the dependent pairs as independent observations for significance tests.

| Projection | Objective | Interpretation |
| --- | --- | --- |
| [Metric MDS](https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling) | Fit pairwise distances with low-dimensional Euclidean distances. | A global view; distortion remains and is measured. |
| [t-SNE](https://jmlr.org/papers/v9/vandermaaten08a.html) | Match neighborhood probabilities. | Local neighborhoods are emphasized; island spacing and area do not measure portfolio overlap. |
| [UMAP](https://arxiv.org/abs/1802.03426) | Fit a fuzzy neighborhood graph at a selected scale. | Useful for neighborhood and multiscale exploration; global distance preservation is not guaranteed. |

Every projection uses the same metric RDM and labels. Original-space within/between means and cross-artist nearest-neighbor fractions are the overlap measures. The showdown also reports provider-stratified and matched-subject comparisons. These remain descriptive because twelve selected images are insufficient for broad claims about artists or cognition.

## Provider Models

[Official OpenAI documentation](https://developers.openai.com/api/docs/guides/image-generation) identifies `gpt-image-2.5-flare` for routine generation and `gpt-image-2.5-sunburst` for editing precision. Both IDs appeared in the account's model listing during verification. The Images API supports generation and reference-image edits.

[Official Gemini documentation](https://ai.google.dev/gemini-api/docs/image-generation) identifies Nano Banana 2 (`gemini-3.1-flash-image`), Nano Banana Pro (`gemini-3-pro-image`), and the newer cost-focused Nano Banana 2 Lite (`gemini-3.1-flash-lite-image`). The default is the full Nano Banana 2 because the workflow includes reference consistency; the Lite ID is selectable. Provider branding, recommendation, and release recency are different questions. No provider is declared a universal winner by this repository.
