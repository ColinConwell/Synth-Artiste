"""Reproducible artifacts for the two-artist teaching experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .artist import SyntheticArtist
from .generation import ImageGenerator
from .images import as_image, release_memory
from .similarity import Similarity
from .spaces import portfolio_overlap

ARTISTS = [
    SyntheticArtist("Geometric Nocturne", "Use flat geometric silhouettes, indigo negative space, and small amber accents."),
    SyntheticArtist("Botanical Reverie", "Use fine organic ink contours, muted sage washes, and delicate paper texture."),
]
SUBJECTS = ["a glass vase with three flowers", "a chair beside a window", "a small house among trees"]


def cached_image(generator, prompt, output, **kwargs):
    """Reuse only exact requests, recording model and content hashes next to images."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = {"generator": generator.describe(), "prompt": prompt, "options": {}}
    for key, value in kwargs.items():
        if key == "reference" and value is not None:
            image = as_image(value)
            value = hashlib.sha256(image.tobytes() + str(image.size).encode()).hexdigest()
        request["options"][key] = value
    sidecar = output.with_suffix('.json')
    if output.exists() and sidecar.exists() and json.loads(sidecar.read_text()) == request:
        return as_image(output)
    image = generator.generate(prompt, **kwargs)
    image.save(output)
    sidecar.write_text(json.dumps(request, indent=2))
    return image


def make_portfolios(output, *, backends=("openai", "gemini"), subjects=None, candidates=2):
    """Matched subjects and providers, with DreamSim best-of-N selection per anchor.

    All candidates and baseline choices are retained. API calls have no fixed seed.
    A tiny demonstration is not evidence about human artists or human judgments.
    """
    if candidates < 1:
        raise ValueError("candidates must be positive")
    subjects = SUBJECTS if subjects is None else list(subjects)
    if len(subjects) < 2:
        raise ValueError("Use at least two subjects per artist")
    output = Path(output)
    metric = Similarity('dreamsim')
    records = []
    for backend in backends:
        generator = ImageGenerator(backend)
        for artist_id, artist in enumerate(ARTISTS):
            anchor = cached_image(generator, artist.build_image_prompt('an arrangement of two ceramic vessels'),
                output / backend / f'artist-{artist_id}' / 'anchor.png')
            target = metric.embed([anchor])[0]
            for subject_id, subject in enumerate(subjects):
                folder = output / backend / f'artist-{artist_id}' / f'subject-{subject_id}'
                prompt = artist.build_image_prompt(subject)
                paths = [folder / f'candidate-{k}.png' for k in range(candidates)]
                images = [cached_image(generator, prompt, path) for path in paths]
                distances = np.clip(1 - metric.embed(images) @ target, 0, 2)
                selected = int(distances.argmin())
                records.append(dict(artist=artist.name, backend=backend, model=generator.model_id,
                    subject=subject, prompt=prompt, path=str(paths[selected].relative_to(output)),
                    baseline=str(paths[0].relative_to(output)), selected=selected,
                    distances=distances.tolist(), candidates=[str(p.relative_to(output)) for p in paths]))
    manifest = dict(conditioning='DreamSim best-of-N; anchor excluded from evaluation',
                    candidates=candidates, records=records)
    (output / 'portfolio.json').write_text(json.dumps(manifest, indent=2))
    return manifest


def evaluate_portfolios(output, metrics=("dreamsim", "lpips", "tpips"), factor="artistic style"):
    output = Path(output)
    records = json.loads((output / 'portfolio.json').read_text())['records']
    images = [output / r['path'] for r in records]
    labels = [r['artist'] for r in records]
    report = {}
    for name in metrics:
        metric = Similarity(name, factor=factor if name == 'tpips' else 'overall')
        distances = metric.pairwise(images)
        np.save(output / f'{name}-distances.npy', distances)
        # Report within each provider as well as pooled; pooling can mask provider effects.
        report[name] = {'pooled': portfolio_overlap(distances, labels), 'by_provider': {}}
        for backend in sorted({r['backend'] for r in records}):
            idx = [i for i, r in enumerate(records) if r['backend'] == backend]
            report[name]['by_provider'][backend] = portfolio_overlap(distances[np.ix_(idx, idx)],
                                                                    [labels[i] for i in idx])
        del metric
        release_memory()
    (output / 'evaluation.json').write_text(json.dumps(report, indent=2))
    return report
