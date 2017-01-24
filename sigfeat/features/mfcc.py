import numpy as np
from scipy.fftpack import dct

from pyfilterbank.melbank import compute_melmat

from ..feature import Feature
from ..feature import HiddenFeature
from ..parameter import Parameter

from .spectral import AbsRfft


class MelSpectrum(HiddenFeature):
    numbands = Parameter(128)
    fmin = Parameter(0)
    fmax = Parameter(None)

    def requires(self):
        yield AbsRfft

    def on_start(self, source, featureset, sink):
        fftmax = np.max(featureset['AbsRfft'].frequencies)
        nfft = featureset['AbsRfft'].nfft
        if not self.fmax or self.fmax > fftmax:
            self.fmax = fftmax
        self.melmat, (self.melfreqs, self.fftfreqs) = compute_melmat(
            self.numbands,
            self.fmin,
            self.fmax,
            nfft,
            source.samplerate
        )  # TODO scaling by bandwidth not done until now

    def process(self, data, resd):
        return self.melmat @ resd['AbsRfft']


class LogMelSpectrum(HiddenFeature):
    def requires(self):
        yield MelSpectrum

    def process(self, data, resd):
        return 20*np.log10(resd['MelSpectrum'])


class MFCC(Feature):
    numbins = Parameter(20)

    def requires(self):
        yield LogMelSpectrum

    def process(self, data, resd):
        return dct(resd['LogMelSpectrum'], type=3, n=self.numbins)
