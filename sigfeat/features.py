import numpy as np

from scipy.stats import gmean
from scipy.signal import get_window
from scipy.fftpack import rfft, rfftfreq

from .feature import Feature
from .feature import HiddenFeature
from .parameter import Parameter


class WindowedSignal(HiddenFeature):
    """WindowedSignal Feature provides a windowed block from source.

    Parameters
    ----------
    window : str
        Name of window to be used (supports all from scipy.signal.get_window).
        Default is 'hann'.
    size : int
        Size of the window.
    periodic : bool
        Periodic True (e.g. for ffts) or symmetric False.

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
class Rfft(HiddenFeature):
    """Rfft Spectrum feature (hidden per default)

    Parameters
    ----------
    nfft : int
    axis : int
    window : bool
        Whether to use a window or not. If you need a special window,
        create a WindowedSignal instance.
    """
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


class AbsRfft(Rfft):
    """Absolute Rfft Spectrum feature (hidden per default)

    Parameters
    ----------
    nfft : int
    axis : int
    window : bool
        Whether to use a window or not. If you need a special window,
        create a WindowedSignal instance.

    """
    def requires(self):
        return [Rfft(
            nfft=self.nfft,
            axis=self.axis,
            window=self.window
        )]

    def process(self, data, featuredata):
        return np.abs(featuredata['Rfft'])


class SpectralCentroidAbsRfft(Feature):
    """SpectralCentroid of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the centroid will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        return [AbsRfft]

    def on_start(self, source, featureset, sink):
        self.channels = source.channels
        self.frequencies = featureset['Rfft'].frequencies

        if self.channels > 1:
            self.frequencies = np.tile(
                self.frequencies
                (self.channels, 1)).T

    def process(self, data, featuredata):
        return centroid(
            self.frequencies,
            featuredata['AbsRfft'],
            self.axis)


class SpectralFlatnessAbsRfft(Feature):
    """SpectralFlatness of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the flatness will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        return [AbsRfft]

    def process(self, data, featuredata):
        return flatness(featuredata['AbsRfft'], self.axis)


class SpectralFluxAbsRfft(Feature):
    """SpectralFlux of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the flux will be calculated, default=0.

    """
    axis = Parameter(0)

    def on_start(self, source, featureset, sink):
        nfft = featureset['AbsRfft'].nfft
        if source.channels > 1:
            self._lastspec = np.ones(nfft, source.channels)
        else:
            self._lastspec = np.ones(nfft)

    def requires(self):
        return [AbsRfft]

    def process(self, data, featuredata):
        curspec = featuredata['AbsRfft']
        specflux = flux(self._lastspec, curspec, self.axis)
        self._lastspec = curspec
        if not specflux:
            return 0.0
        return specflux


class SpectralCrestFactorAbsRfft(Feature):

    def requires(self):
        return [AbsRfft]

    def process(self, data, result):
        return crest_factor(result['AbsRfft'])


class CrestFactor(Feature):
    """Crest Factor of Source data."""

    def process(self, data, result):
        return crest_factor(data[0])


class ZeroCrossingCout(Feature):
    """Counts Zero Crossings of Source data."""

    def process(self, data, result):
        return zero_crossing_count(data[0])


class StatMoments(Feature):
    """Estimates mu, variance, skewness and kurtosis of Source data."""
    labels = ['mu', 'mu_variance', 'mu_skewness', 'mu_kurtosis']

    def process(sefl, data, resutl):
        return moments(data[0])


class SquaredSignal(HiddenFeature):
    """Squared Signal data."""

    def process(self, data, result):
        sig = data[0]
        return sig*sig


class AbsSignal(HiddenFeature):

    def process(self, data, result):
        return np.abs(data[0])


class CentroidSquaredSignal(Feature):
    """Experimental Centroid of abs source data."""
    axis = Parameter(0)

    def requires(self):
        return [AbsSignal()]

    def on_start(self, source, featureset, sink):
        self.index = np.arange(source.blocksize) * (1.0 / source.samplerate)

    def process(self, data, result):
        return centroid(self.index, result['AbsSignal'], axis=self.axis)


class FlatnessSquaredSignal(Feature):
    """Experimental Flatness of abs source data."""
    axis = Parameter(0)

    def requires(self):
        return [AbsSignal()]

    def process(self, data, result):
        return flatness(result['AbsSignal'], axis=self.axis)


def centroid(index, values, axis):
    return np.sum(
        index * values, axis=axis) / np.sum(values, axis=axis)


def flatness(values, axis):
    return gmean(values, axis=axis) / np.mean(values, axis=axis)


def flux(values1, values2, axis):
    def _abs_max_ratio(s):
        return np.abs(s) / np.max(np.abs(s), axis=axis)
    d = _abs_max_ratio(values2) - _abs_max_ratio(values1)
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
