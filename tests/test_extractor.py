import pytest

from sigfeat.base import Feature
from sigfeat.extractor import Extractor
from sigfeat.source.array import ArraySource
from sigfeat.sink import DefaultDictSink


class A(Feature):
    _started = False

    def on_start(self, *args):
        self._started = True

    def process(self, data, fdata):
        return int(data[0])


def test_extractor_with_sink():
    ex = Extractor(A(), A(name='hidden_a').hide())
    sc = ArraySource(
        list(range(10)),
        blocksize=1,
        overlap=0,
        samplerate=1)
    sk = DefaultDictSink()
    ex.extract(sc, sk)
    assert list(sk['results']['A']) == list(range(10))
    assert any(sk['hiddenfeatures'])
    assert any(sk['features'])


def test_extractor_without_sink():
    ex = Extractor(A())
    sc = ArraySource(
        list(range(10)),
        blocksize=1,
        overlap=0,
        samplerate=1)
    for i, res in enumerate(ex.extract(sc)):
        assert i == res['A']

    ex.reset()
    assert ex.featureset['A']._started is False


if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
