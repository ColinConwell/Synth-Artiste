"""Compare portfolios in the original distance space; project only for display."""
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def validate_distances(distances):
    d = np.asarray(distances, dtype=float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 3:
        raise ValueError("Expected a square distance matrix with at least three images")
    if not np.isfinite(d).all() or (d < -1e-6).any() or not np.allclose(d, d.T, atol=1e-5):
        raise ValueError("Distances must be finite, nonnegative, and symmetric")
    if not np.allclose(np.diag(d), 0, atol=1e-5):
        raise ValueError("Distance diagonal must be zero")
    return np.maximum(d, 0)


def project(distances, method="mds", seed=42):
    d = validate_distances(distances)
    if method == "mds":
        from sklearn.manifold import MDS
        xy = MDS(n_components=2, dissimilarity="precomputed", random_state=seed, n_init=4).fit_transform(d)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        xy = TSNE(n_components=2, metric="precomputed", init="random", random_state=seed,
                  perplexity=min(5, (len(d) - 1) / 3), learning_rate="auto").fit_transform(d)
    elif method == "umap":
        from umap import UMAP
        xy = UMAP(n_components=2, metric="precomputed", init="random", random_state=seed,
                  n_neighbors=min(5, len(d) - 1)).fit_transform(d)
    else:
        raise ValueError("Choose mds (global), tsne (local), or umap (neighborhood/multiscale)")
    upper = np.triu_indices(len(d), 1)
    rho = spearmanr(d[upper], squareform(pdist(xy))[upper]).statistic
    return xy, {"distance_rank_correlation": float(rho)}


def portfolio_overlap(distances, labels):
    d = validate_distances(distances)
    labels = np.asarray(labels)
    if len(labels) != len(d) or len(np.unique(labels)) != 2:
        raise ValueError("Provide one label per image, from exactly two portfolios")
    i, j = np.triu_indices(len(d), 1)
    same = labels[i] == labels[j]
    if not same.any():
        raise ValueError("Need repeated images per portfolio")
    neighbors = d.copy()
    np.fill_diagonal(neighbors, np.inf)
    return {"within_mean": float(d[i[same], j[same]].mean()),
            "between_mean": float(d[i[~same], j[~same]].mean()),
            "cross_artist_nearest_neighbor_fraction": float(np.mean(labels[neighbors.argmin(1)] != labels))}
