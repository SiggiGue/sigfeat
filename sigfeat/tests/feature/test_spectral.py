import pytest

import numpy as np

from sigfeat.feature.spectral import SpectralCentroid
from sigfeat.feature.spectral import SpectralFlatness
from sigfeat.feature.spectral import SpectralFlux
from sigfeat.feature.spectral import SpectralCrestFactor
from sigfeat.feature.spectral import SpectralRolloff

from sigfeat.source.array import ArraySource
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink


def test_spectral_features():
    features = [
        SpectralCentroid(),
        SpectralFlatness(),
        SpectralFlux(),
        SpectralCrestFactor(),
        SpectralRolloff(),
    ]
    etr = Extractor(*features)
    x = np.sin(1000*2*np.pi*np.linspace(0, 1, 44100))
    src = ArraySource(
        x,
        samplerate=44100,
        blocksize=2048,
        overlap=1024)
    snk = etr.extract(src, DefaultDictSink())
    res = snk['results']
    assert abs(1-np.median(res['SpectralCentroid'])/1000) < 1e-3
    assert abs(np.median(res['SpectralFlatness'])) < 1e-3
    # TODO


def test_spectral_features_no_window_branch():
    from sigfeat.feature.spectral import AbsRfft
    features = [
        AbsRfft(window=False),
        SpectralCentroid(),
        SpectralFlatness(),
        SpectralFlux(),
        SpectralCrestFactor(),
        SpectralRolloff(),
    ]
    etr = Extractor(*features)
    x = np.sin(1000*2*np.pi*np.linspace(0, 1, 44100))
    src = ArraySource(
        np.tile(x, (2, 1)).T,
        samplerate=44100,
        blocksize=2048,
        overlap=1024)
    etr.extract(src, DefaultDictSink())


if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
