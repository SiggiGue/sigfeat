import numpy as np

from sigfeat.feature.delta import Delta
from sigfeat.feature import Feature
from sigfeat.source.array import ArraySource
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink


def test_delta():
    class A(Feature):
        def process(self, data, res):
            return float(data[0])
    ex = Extractor(
        Delta(A()),
        Delta(A(), order=2)
    )
    x = np.arange(0.0, 10.0, 2)
    src = ArraySource(x, samplerate=1, blocksize=1, channels=1)
    snk = ex.extract(src, DefaultDictSink())
    res = snk['results']
    assert np.allclose(np.array(res['A']).flatten(), x)
    assert np.mean(np.array(res['dA']).flatten()[1:]) == 2.0
    assert np.mean(np.array(res['ddA']).flatten()[2:]) == 0.0
    assert res['dA'][0] is None
    assert res['ddA'][0] is None
