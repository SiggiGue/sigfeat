import numpy as np
import scipy.stats
import scipy.signal

from scipy.signal import get_window
from scipy.fftpack import rfft, rfftfreq

from .feature import Feature
from .feature import HiddenFeature
from .parameter import Parameter


class WindowedSignal(HiddenFeature):
    """WindowedSignal Feature provides a windowed block from source.

    Parameters
    ----------
    """
    window = Parameter(default='hann')
    size = Parameter()
    periodic = Parameter(default=True)

    def on_start(self, source, *args, **kwargs):
        if not self.size:
            self.size = source.blocksize

        win = get_window(
            window=self.window,
            Nx=self.size,
            fftbins=self.periodic)
        self.channels = source.channels

        if self.channels > 1:
            win = np.tile(win, (self.channels, 1)).T
        self.w = win

    def process(self, data, result):
        if self.window == 'rect':
            return data[0]
        return self.w * data[0]


# Spectral Features
class SpectrumRfft(HiddenFeature):
    nfft = Parameter()
    axis = Parameter(default=0)
    window = Parameter(default=True)

    def requires(self):
        if self.window:
            return [WindowedSignal(size=self.nfft)]
        else:
            return []

    def on_start(self, source, *args, **kwargs):
        if not self.nfft:
            self.nfft = source.blocksize
        self.frequencies = rfftfreq(self.nfft, 1.0/source.samplerate)
        self.add_metadata(
            'frequencies', self.frequencies)

    def process(self, data, featuredata):
        if self.window:
            s = featuredata['WindowedSignal']
        else:
            s = data[0]
        return rfft(
            s,
            n=self.nfft,
            axis=self.axis)


class AbsSpectrumRfft(SpectrumRfft):

    def requires(self):
        return [SpectrumRfft(
            nfft=self.nfft,
            axis=self.axis,
            window=self.window
        )]

    def process(self, data, featuredata):
        return np.abs(featuredata['SpectrumRfft'])


class SpectralCentroid(Feature):
    axis = Parameter(0)

    def requires(self):
        return [AbsSpectrumRfft]

    def on_start(self, source, featureset, sink):
        self.channels = source.channels
        self.frequencies = featureset['SpectrumRfft'].frequencies

        if self.channels > 1:
            self.frequencies = np.tile(
                self.frequencies
                (self.channels, 1)).T

    def process(self, data, featuredata):
        return sepectral_centroid(
            self.frequencies,
            featuredata['AbsSpectrumRfft'],
            self.axis)


class SpectralFlatness(Feature):
    axis = Parameter(0)

    def requires(self):
        return [AbsSpectrumRfft]

    def process(self, data, featuredata):
        return spectral_flatness(featuredata['AbsSpectrumRfft'], self.axis)


class SpectralFlux(Feature):
    _lastspec = None
    _firstiter = True
    axis = Parameter(0)

    def on_start(self, source, featureset, sink):
        self.channels = source.channels

    def requires(self):
        return [AbsSpectrumRfft]

    def process(self, data, featuredata):
        curspec = featuredata['AbsSpectrumRfft']

        if self._firstiter:
            self._lastspec = curspec
            self._firstiter = False
            if self.channels > 1:
                return np.zeros(self.channels)
            else:
                return 0

        flux = spectral_flux(self._lastspec, curspec, self.axis)
        self._lastspec = curspec
        return flux


def sepectral_centroid(frequencies, spectrum, axis):
    return np.sum(
        frequencies * spectrum, axis=axis) / np.sum(spectrum, axis=axis)


def spectral_flatness(spectrum, axis):
    return scipy.stats.gmean(
        spectrum, axis=axis) / np.mean(spectrum, axis=axis)


def spectral_flux(spectrum1, spectrum2, axis):
    def _abs_max_ratio(s):
        return np.abs(s) / np.max(np.abs(s), axis=axis)
    d = _abs_max_ratio(spectrum2) - _abs_max_ratio(spectrum1)
    return 0.5 * np.sum(d + np.abs(d), axis=axis)


def crest_factor(signal):
    return np.max(np.abs(signal)) / np.sqrt(np.mean(signal*signal))


def zero_crossing_count(signal):
    return np.round(0.5 * np.sum(np.abs(np.diff(np.sign(signal)))))


def moments(signal):
    mu = np.mean(signal)
    sigma = signal - mu
    framecount = len(signal)

    # moments:
    ss = sigma*sigma
    mu_variance = np.mean(ss)
    sss = ss * sigma
    mu_skewness = np.mean(sss)
    mu_kurtosis = np.mean(sss * sigma)

    # standardize the moments:
    mu_kurtosis = mu_kurtosis / (mu_variance * mu_variance)
    mu_skewness = mu_skewness / np.sqrt(mu_variance)**3
    mu_variance = mu_variance * framecount / (framecount - 1)

    return mu, mu_variance, mu_skewness, mu_kurtosis
