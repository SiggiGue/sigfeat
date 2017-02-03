import pytest
import numpy as np
from sigfeat.source.array import ArraySource
from sigfeat.preprocess import Preprocess
from sigfeat.preprocess.mix import MeanMix
from sigfeat.preprocess.mix import SumMix


def test_abstract_preprocess():
    with pytest.raises(TypeError):
        Preprocess()

    class P(Preprocess):
        def process(self, data):
            return data
    src = ArraySource(
        np.ones((10, 2)),
        samplerate=10)
    pp = P(src)
    list(pp)


def test_mean_mix():
    src = ArraySource(
        np.ones((10, 2)),
        channels=2,
        samplerate=10,
        blocksize=1)
    pp = MeanMix(src)
    for data in pp:
        assert data[0] == 1.0


def test_sum_mix():
    src = ArraySource(
        np.ones((10, 2)),
        samplerate=10,
        blocksize=1)
    pp = SumMix(src)
    for data in pp:
        assert data[0] == 2.0


if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
