import numpy as np
from scipy.fftpack import rfft, rfftfreq

from ..feature import Feature
from ..feature import HiddenFeature
from ..parameter import Parameter

from .common import WindowedSignal
from .common import centroid
from .common import crest_factor
from .common import flatness
from .common import flux
from .common import rolloff


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
