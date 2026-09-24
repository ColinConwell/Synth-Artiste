"""Similarity-conditioned selection works with every image generator."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Selection:
    image: object
    candidates: list
    distances: list[float]
    index: int


def generate_similar(generator, prompt, reference, *, metric=None, candidates=4, seed=None, **kwargs):
    """Generate N candidates and select the smallest reference distance.

    This is best-of-N conditioning, not gradient guidance or an embedding adapter.
    The reference is used for selection only. Pass native image conditioning by
    constructing a callable separately when both mechanisms are desired.
    """
    if candidates < 1:
        raise ValueError("candidates must be positive")
    if metric is None:
        from .similarity import Similarity
        metric = Similarity("dreamsim")
    images = [generator.generate(prompt, seed=None if seed is None else seed + i, **kwargs)
              for i in range(candidates)]
    if metric.name in {"dreamsim", "tpips"}:
        x = metric.embed([reference, *images])
        distances = np.clip(1 - x[1:] @ x[0], 0, 2).tolist()
    else:
        distances = [metric.distance(reference, image) for image in images]
    if not np.isfinite(distances).all():
        raise ValueError("Metric produced non-finite distances")
    index = int(np.argmin(distances))
    return Selection(images[index], images, distances, index)
