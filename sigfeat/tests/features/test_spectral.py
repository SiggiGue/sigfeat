import pytest

import numpy as np

from sigfeat.features.spectral import SpectralCentroid
from sigfeat.features.spectral import SpectralFlatness
from sigfeat.features.spectral import SpectralFlux
from sigfeat.features.spectral import SpectralCrestFactor
from sigfeat.features.spectral import SpectralRolloff

from sigfeat.source import ArraySource
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

if __name__ == '__main__':
    pytest.main()
