from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class RunRecord:
    backend: str
    artists: List[Dict]
    prompts: List[str]
    size: str
    quality: str
    output_dir: str


def pairwise_cosine_dist(X: np.ndarray) -> np.ndarray:
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    sims = X @ X.T
    dists = 1 - sims
    return dists


def diversity_stats(embeddings: np.ndarray) -> Dict[str, float]:
    if embeddings.shape[0] < 2:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    D = pairwise_cosine_dist(embeddings)
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]
    return {
        "mean": float(np.mean(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def write_manifest(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


