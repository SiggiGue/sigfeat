import pytest

import numpy as np

from sigfeat.features.temporal import CrestFactor
from sigfeat.features.temporal import ZeroCrossingRate
from sigfeat.features.temporal import StatMoments
from sigfeat.features.temporal import CentroidAbsSignal
from sigfeat.features.temporal import FlatnessAbsSignal
from sigfeat.features.temporal import RootMeanSquare
from sigfeat.features.temporal import Peak
from sigfeat.features.temporal import Kurtosis
from sigfeat.features.temporal import Skewness
from sigfeat.features.temporal import StandardDeviation

from sigfeat.source import ArraySource
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink


def test_temporal_features():
    features = [
        CrestFactor(),
        ZeroCrossingRate(),
        StatMoments(),
        CentroidAbsSignal(),
        FlatnessAbsSignal(),
        RootMeanSquare(),
        Peak(),
        Kurtosis(),
        Skewness(),
        StandardDeviation(),
    ]
    ex = Extractor(*features)
    x = np.sin(1000*2*np.pi*np.linspace(0, 1, 44100))
    src = ArraySource(x, samplerate=44100, blocksize=4096)
    snk = ex.extract(src, DefaultDictSink())
    res = snk['results']
    assert abs(np.median(res['CrestFactor'])-2**0.5) < 1e-3
    assert abs(1-np.median(res['ZeroCrossingRate'])/2000.0) < 1e-2
    assert abs(2**-0.5-np.median(res['RootMeanSquare'])) < 1e-3
    assert abs(1-np.median(res['Peak'])) < 1e-4


if __name__ == '__main__':
    pytest.main()
