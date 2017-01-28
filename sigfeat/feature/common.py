import numpy as np
from scipy.signal import get_window
from scipy.stats import gmean

from .feature import Feature
from .feature import HiddenFeature
from ..parameter import Parameter


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
