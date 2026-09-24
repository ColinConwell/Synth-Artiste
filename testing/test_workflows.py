import json
import numpy as np
import pytest
from PIL import Image
from synthart import generate_similar, Similarity, ImageGenerator
from synthart.spaces import project, portfolio_overlap, validate_distances
from synthart.experiment import cached_image


class CandidateGenerator:
    def __init__(self):
        self.seeds = []
    def generate(self, prompt, seed=None):
        self.seeds.append(seed)
        return Image.new('RGB', (8, 8), (seed, 0, 0))
    def describe(self):
        return {'backend': 'test'}


class RedDistance:
    name = 'test'
    def distance(self, a, b):
        return abs(a.getpixel((0, 0))[0] - b.getpixel((0, 0))[0])


def test_selection_uses_reference_and_distinct_seeds():
    generator = CandidateGenerator()
    selected = generate_similar(generator, 'red', Image.new('RGB', (8, 8), (12,0,0)),
                                metric=RedDistance(), candidates=4, seed=10)
    assert generator.seeds == [10,11,12,13]
    assert selected.index == 2
    assert selected.distances == [2,1,0,1]
    with pytest.raises(ValueError):
        generate_similar(generator, 'x', None, metric=RedDistance(), candidates=0)


def test_cache_invalidates_changed_request(tmp_path):
    generator = CandidateGenerator()
    path = tmp_path / 'image.png'
    cached_image(generator, 'x', path, seed=10)
    cached_image(generator, 'x', path, seed=10)
    assert generator.seeds == [10]
    cached_image(generator, 'x', path, seed=11)
    assert generator.seeds == [10,11]
    assert json.loads(path.with_suffix('.json').read_text())['options']['seed'] == 11


def test_overlap_excludes_self_neighbors():
    d = np.array([[0,1,4,5],[1,0,5,4],[4,5,0,1],[5,4,1,0]])
    result = portfolio_overlap(d, ['a','a','b','b'])
    assert result == {'within_mean':1., 'between_mean':4.5, 'cross_artist_nearest_neighbor_fraction':0.}
    xy, diagnostics = project(d, 'mds')
    assert xy.shape == (4,2)
    assert np.isfinite(xy).all()
    assert diagnostics['distance_rank_correlation'] > .5


@pytest.mark.parametrize('matrix', [np.ones((3,2)), np.eye(3), [[0,1,2],[3,0,2],[2,2,0]]])
def test_bad_rdms_rejected(matrix):
    with pytest.raises(ValueError):
        validate_distances(matrix)


def test_unsupported_options_fail_before_download():
    with pytest.raises(ValueError):
        Similarity('lpips', factor='color')
    with pytest.raises(ValueError):
        ImageGenerator('gemini', ip_adapter=True)


def test_original_config_positional_interface():
    from synthart import ImageConfig
    assert ImageConfig('512x512').size == '512x512'
    assert ImageConfig().model == 'gpt-image-2.5-flare'
