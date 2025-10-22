from __future__ import annotations

import numpy as np
from typing import Iterable, List, Sequence


def farthest_first_indices(embeddings: np.ndarray, k: int) -> List[int]:
    """Select k indices maximizing spread via greedy farthest-first traversal.

    embeddings: (N, D) array
    returns: list of k unique indices
    """
    N = embeddings.shape[0]
    if k >= N:
        return list(range(N))
    # Normalize for cosine geometry
    X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    # Start with the point of max norm (already unit) – effectively arbitrary
    first = 0
    selected = [first]
    # Maintain min distance to selected set for each point
    dists = 1 - (X @ X[first].T)
    for _ in range(1, k):
        # pick farthest from current selected set
        idx = int(np.argmax(dists))
        selected.append(idx)
        # update distances via min over selected
        dists = np.minimum(dists, 1 - (X @ X[idx].T))
    return selected


def select_prompts_by_embeddings(candidates: Sequence[str], embeddings: np.ndarray, k: int) -> List[str]:
    idxs = farthest_first_indices(embeddings, k)
    return [candidates[i] for i in idxs]


