import numpy as np

from sigfeat.feature.common import Index
from sigfeat.feature.common import WindowedSignal
from sigfeat.source.array import ArraySource
from sigfeat.feature.common import centroid
from sigfeat.feature.common import flatness
from sigfeat.feature.common import flux
from sigfeat.feature.common import rolloff
from sigfeat.feature.common import crest_factor
from sigfeat.feature.common import zero_crossing_count
from sigfeat.feature.common import moments


def test_index():
    idx = Index()
    res = idx.process(((1, 2, 3), 100), {})
    assert res == 100


def test_windowed_signal():
    block = np.ones(10)
    sc = ArraySource(block, samplerate=1)
    wsf = WindowedSignal(window='hanning', size=10)
    wsf.on_start(sc)
    res = wsf.process((block, 0), {})
    assert np.allclose(res,  wsf.w)
    block = np.ones((10, 2))
    sc = ArraySource(block, samplerate=1, blocksize=10)
    wsf = WindowedSignal(window='blackman')
    wsf.on_start(sc)
    res = wsf.process((block, 0), {})
    assert np.allclose(res,  wsf.w)
    sc = ArraySource(block, samplerate=1, blocksize=10)
    wsf = WindowedSignal(window='rect')
    wsf.on_start(sc)
    res = wsf.process((block, 0), {})
    assert np.allclose(res,  wsf.w)


def test_centroid():
    x = np.zeros((9, 2)) + 1e-20
    x[3, 0] = 1.0
    x[4, 1] = 2.0
    i = (np.arange(len(x)) * np.ones_like(x).T).T
    res = centroid(i, x, axis=0)
    assert list(res) == [3, 4]


def test_flatness():
    res = flatness([1e-20, 1, 1e-20], axis=0)
    assert res <= 1e-10
    res = flatness(np.ones((3, 2)), axis=0)
    assert list(res) == [1.0, 1.0]


def test_flux():
    res = flux(np.zeros(3)+1e-10, np.ones(3), axis=0)
    assert res == 0.0
    res = flux(np.ones(3), np.ones(3), axis=0)
    assert res == 0.0
    res = flux(np.array([0, 1]), np.array([1, 0]), axis=0)
    assert res == 1.0


def test_roloff():
    res = rolloff(np.array([1, 1, 1, 0]), 1, 0.75)
    assert res == 0.25


def test_crest_factor():
    x = np.array([1, 1, -1, -1])
    res = crest_factor(x)
    assert res == 1.0
    x = np.array([0, 1, 2, 1, 0, -1, -2, -1, 0])
    res = crest_factor(x)
    desired = (3**0.5)
    assert np.abs(res-desired) < 1e-5


def test_zero_crossing_count():
    x = [1, -1, 10, -10, -0.1, 0.1]
    res = zero_crossing_count(x)
    assert res == 4.0


def test_moments():
    x = np.random.randn(10000)
    m, v, s, k = moments(x)
    assert abs(m-np.mean(x)) < 0.01
    assert abs(v-np.var(x)) < 0.01
    assert abs(s) < 0.2
    assert abs(3-k) < 0.5


if __name__ == '__main__':
    import pytest
    pytest.main()  # pragma: no coverage
