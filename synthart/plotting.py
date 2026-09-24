"""Small display helpers shared by scripts and notebooks."""
import math
import matplotlib.pyplot as plt
from .images import as_image


def gallery(images, titles=None, columns=4):
    images = list(images)
    if not images:
        raise ValueError("Gallery needs at least one image")
    fig, axes = plt.subplots(math.ceil(len(images) / columns), columns,
                             figsize=(3 * columns, 3 * math.ceil(len(images) / columns)), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < len(images):
            ax.imshow(as_image(images[i]))
            if titles:
                ax.set_title(titles[i], fontsize=9, wrap=True)
    fig.tight_layout()
    return fig


def plot_spaces(distances, labels, methods=("mds", "tsne", "umap")):
    from .spaces import project
    import numpy as np
    labels = np.asarray(labels)
    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 4), squeeze=False)
    diagnostics = {}
    for ax, method in zip(axes.flat, methods):
        xy, diagnostics[method] = project(distances, method)
        for label in np.unique(labels):
            mask = labels == label
            ax.scatter(*xy[mask].T, label=label)
        for i, point in enumerate(xy):
            ax.annotate(str(i), point, fontsize=8)
        ax.set_title(method.upper())
        ax.legend()
    fig.tight_layout()
    return fig, diagnostics
