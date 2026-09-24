"""One distance interface, with each metric's published preprocessing preserved."""
import numpy as np
from .images import as_image, device_name


class Similarity:
    """Lower distances mean more similar. Only TPIPS accepts a semantic factor.

    DreamSim and TPIPS expose reusable embeddings; LPIPS compares spatial features.
    Models download on construction and are reused across calls.
    """
    def __init__(self, name="dreamsim", *, device=None, factor="overall"):
        if name not in {"dreamsim", "lpips", "tpips"}:
            raise ValueError("Choose dreamsim, lpips, or tpips")
        if name != "tpips" and factor != "overall":
            raise ValueError("Only TPIPS supports an aspect/factor")
        self.name, self.factor = name, factor
        self.device = device_name(device)
        if name == "dreamsim":
            from dreamsim import dreamsim
            self.model, self.preprocess = dreamsim(pretrained=True, device=self.device)
        elif name == "lpips":
            import lpips
            self.model = lpips.LPIPS(net="alex").to(self.device).eval()
        else:
            import tpips
            self.model = tpips.load_model("embedding", device=self.device)

    def embed(self, images):
        import torch
        items = list(images)
        if not items:
            raise ValueError("Provide at least one image")
        if self.name == "lpips":
            raise ValueError("LPIPS has spatial features, not a global embedding; use pairwise")
        with torch.inference_mode():
            vectors = []
            for item in items:
                image = as_image(item)
                if self.name == "dreamsim":
                    vector = self.model.embed(self.preprocess(image).to(self.device))
                else:
                    vector = self.model.embed(image, factor=self.factor)
                vectors.append(vector.detach().float().cpu().numpy().reshape(1, -1))
        x = np.concatenate(vectors)
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)

    def distance(self, a, b):
        if self.name != "lpips":
            x = self.embed([a, b])
            return float(np.clip(1 - x[0] @ x[1], 0, 2))
        import torch
        # Fix geometry explicitly for spatial comparisons; document the 256px warp.
        def tensor(image):
            x = np.asarray(as_image(image).resize((256, 256)), dtype=np.float32).copy()
            return torch.from_numpy(x / 127.5 - 1).permute(2, 0, 1)[None].to(self.device)
        with torch.inference_mode():
            return float(self.model(tensor(a), tensor(b)).item())

    def pairwise(self, images):
        items = list(images)
        if not items:
            raise ValueError("Provide at least one image")
        if self.name != "lpips":
            x = self.embed(items)
            distances = np.clip(1 - x @ x.T, 0, 2)
        else:
            distances = np.zeros((len(items), len(items)))
            for i in range(len(items)):
                for j in range(i):
                    distances[i, j] = distances[j, i] = self.distance(items[i], items[j])
        np.fill_diagonal(distances, 0)
        return distances
