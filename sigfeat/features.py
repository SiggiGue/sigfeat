import numpy as np

from scipy.stats import gmean, skew, kurtosis
from scipy.signal import get_window
from scipy.fftpack import rfft, rfftfreq

from .feature import Feature
from .feature import HiddenFeature
from .parameter import Parameter


class Index(Feature):
    """Index of source."""
    def process(self, data, result):
        return data[1]


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


# SPECTRAL FEATURES ###########################################################

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


class SpectralCentroid(Feature):
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
                self.frequencies,
                (self.channels, 1)).T

    def process(self, data, featuredata):
        result = centroid(
            self.frequencies,
            featuredata['AbsRfft'],
            self.axis)
        return result


class SpectralFlatness(Feature):
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


class SpectralFlux(Feature):
    """SpectralFlux of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the flux will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        return [AbsRfft]

    def on_start(self, source, featureset, sink):
        nfft = featureset['AbsRfft'].nfft
        if source.channels > 1:
            self._lastspec = np.ones((nfft, source.channels))
        else:
            self._lastspec = np.ones(nfft)

    def process(self, data, featuredata):
        curspec = featuredata['AbsRfft']
        specflux = flux(self._lastspec, curspec, self.axis)
        self._lastspec = curspec
        return specflux


class SpectralCrestFactor(Feature):
    """Spectral Crest Factor"""
    def requires(self):
        return [AbsRfft]

    def process(self, data, result):
        return crest_factor(result['AbsRfft'])


class SpectralRolloff(Feature):
    """Spectral Rolloff from AbsRfft.

    The spectral rolloff is the frequency where the kappa percentage of
    energy is below and the 1-kappa percentage of energy is above.

    Parameters
    ----------
    kappa : scalar {0...1}
        Default 0.95
    """

    kappa = Parameter(0.85)

    def requires(self):
        return [AbsRfft]

    def on_start(self, source, featureset, sink):
        self.samplerate = source.samplerate

    def process(self, data, result):
        return rolloff(result['AbsRfft'], self.samplerate, self.kappa)


# TEMPORAL FEATURES ###########################################################

class CrestFactor(Feature):
    """Crest Factor of Source data."""

    def process(self, data, result):
        return crest_factor(data[0])


class ZeroCrossingRate(Feature):
    """Zero Crossings Rate of Source data."""

    def on_start(self, source, featureset, sink):
        self.factor = source.samplerate / source.blocksize

    def process(self, data, result):
        return zero_crossing_count(data[0])*self.factor


class StatMoments(Feature):
    """Estimates mu, variance, skewness and kurtosis of Source data."""
    labels = ['mu', 'mu_variance', 'mu_skewness', 'mu_kurtosis']

    def process(sefl, data, result):
        return moments(data[0])


class SquaredSignal(HiddenFeature):
    """Squared Signal data (as hidden feature)."""

    def process(self, data, result):
        sig = data[0]
        return sig*sig


class AbsSignal(HiddenFeature):
    """Abs from source data (as hidden feature)."""
    def process(self, data, result):
        return np.abs(data[0])


class CentroidAbsSignal(Feature):
    """Experimental Centroid of abs source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def requires(self):
        return [AbsSignal()]

    def on_start(self, source, featureset, sink):
        self.index = np.arange(source.blocksize) * (1.0 / source.samplerate)

    def process(self, data, result):
        return centroid(self.index, result['AbsSignal'], axis=self.axis)


class FlatnessAbsSignal(Feature):
    """Experimental Flatness of abs source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def requires(self):
        return [AbsSignal()]

    def process(self, data, result):
        return flatness(result['AbsSignal'], axis=self.axis)


class MeanSquare(HiddenFeature):
    """MeanSquare (MS) of squared Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def requires(self):
        return [SquaredSignal()]

    def process(self, data, result):
        return np.mean(result['SquaredSignal'], axis=self.axis)


class RootMeanSquare(Feature):
    """Root Mean Square (RMS) from Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def requires(self):
        return [MeanSquare(axis=self.axis)]

    def process(self, data, result):
        return np.sqrt(result['MeanSquare'])


class Peak(Feature):
    """Peak of AbsSignal from Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def requires(self):
        yield AbsSignal()

    def process(self, data, result):
        return np.max(result['AbsSignal'], axis=self.axis)


class Kurtosis(Feature):
    """Kurtosis of Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def process(self, data, result):
        return kurtosis(data[0], axis=self.axis)


class Skewness(Feature):
    """Skewness of Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def process(self, data, result):
        return skew(data[0], axis=self.axis)


class StandardDeviation(Feature):
    """StandardDeviation (STD) of Source data.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    """
    axis = Parameter(0)

    def process(self, data, result):
        return np.std(data[0], axis=self.axis)


# FUNCTIONS ###################################################################

def centroid(index, values, axis):
    return np.sum(index * values, axis=axis) / np.sum(values, axis=axis)


def flatness(values, axis):
    return gmean(values, axis=axis) / np.mean(values, axis=axis)


def flux(values1, values2, axis):
    def _abs_max_ratio(s):
        return s / np.max(s, axis=axis)
    d = _abs_max_ratio(values2) - _abs_max_ratio(values1)
    return 0.5 * np.sum(d + np.abs(d), axis=axis)


def rolloff(absvalues, samplerate, kappa=0.85):
    cumspec = np.cumsum(absvalues)
    rolloffindex = np.argmax(cumspec > kappa*cumspec[-1])
    frequency = rolloffindex * 0.5 * samplerate / len(absvalues)
    return frequency


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
