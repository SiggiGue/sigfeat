import pytest
import numpy as np

from sigfeat.source import ArraySource
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink

from sigfeat.features.mfcc import MelSpectrum
from sigfeat.features.mfcc import LogMelSpectrum
from sigfeat.features.mfcc import MFCC


def mkas():
    return ArraySource(
        np.random.randn(44100),
        samplerate=44100,
        blocksize=2048,
        overlap=1024
        )


def test_mel_spectrum():
    src = mkas()
    ext = Extractor(MelSpectrum().hide(False))
    snk = ext.extract(src, DefaultDictSink())


def test_log_mel_spectrum():
    src = mkas()
    ext = Extractor(LogMelSpectrum().hide(False))
    snk = ext.extract(src, DefaultDictSink())


def test_mfcc():
    src = mkas()
    ext = Extractor(MFCC())
    snk = ext.extract(src, DefaultDictSink())


if __name__ == '__main__':
    pytest.main()
