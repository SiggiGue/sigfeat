import numpy as np
from scipy.fftpack import rfft, rfftfreq

from ..base import Feature
from ..base import HiddenFeature
from ..base import Parameter

from .common import WindowedSignal
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
            yield WindowedSignal(size=self.nfft)
        else:
            return []

    def on_start(self, source, *args, **kwargs):
        if not self.nfft:
            self.nfft = source.blocksize
        self.frequencies = rfftfreq(self.nfft, 1.0/source.samplerate)
        self.add_metadata(
            'frequencies', self.frequencies)
        self.add_metadata('nfft', self.nfft)

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


class SumAbsRfft(Rfft):
    axis = Parameter(0)

    def requires(self):
        yield AbsRfft

    def process(self, data, resd):
        return np.sum(resd['AbsRfft'], axis=self.axis)


class SpectralCentroid(Feature):
    """Centroid of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the centroid will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield AbsRfft
        yield SumAbsRfft

    def on_start(self, source, featureset, sink):
        self.channels = source.channels
        self.frequencies = featureset['Rfft'].frequencies
        if self.channels > 1:
            self.frequencies = np.tile(
                self.frequencies,
                (self.channels, 1)).T

    @staticmethod
    def centroid(freqs, absrfft, sumabsrfft, axis):
        return np.sum(freqs * absrfft, axis=axis) / sumabsrfft

    def process(self, data, resd):
        result = self.centroid(
            self.frequencies,
            resd['AbsRfft'],
            resd['SumAbsRfft'],
            self.axis)
        return result


class SpectralSpread(SpectralCentroid):
    # TODO: Test
    """Spread of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the centroid will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield SpectralCentroid()

    @staticmethod
    def spread(freqs, absrfft, sumabsrfft, centroid, axis):
        return np.sum((freqs-centroid)**2 * absrfft, axis=axis) / sumabsrfft

    def process(self, data, resd):
        result = self.spread(
            self.frequencies,
            resd['AbsRfft'],
            resd['SumAbsRfft'],
            resd['SpectralCentroid'],
            self.axis)
        return result


class SpectralSkewness(SpectralCentroid):
    # TODO: Test
    """Skewness of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the centroid will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield SpectralCentroid()
        yield SpectralSpread()

    @staticmethod
    def skewness(freqs, absrfft, sumabsrfft, centroid, spread, axis):
        m3 = np.sum((freqs-centroid)**3 * absrfft, axis=axis) / sumabsrfft
        return m3 / np.sqrt(spread)**3

    def process(self, data, resd):
        result = self.skewness(
            self.frequencies,
            resd['AbsRfft'],
            resd['SumAbsRfft'],
            resd['SpectralCentroid'],
            resd['SpectralSpread'],
            self.axis)
        return result


class SpectralKurtosis(SpectralCentroid):
    # TODO: Test
    """Kurtosis of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the centroid will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield SpectralCentroid()
        yield SpectralSpread()

    @staticmethod
    def kurtosis(freqs, absrfft, sumabsrfft, centroid, spread, axis):
        m4 = np.sum((freqs-centroid)**4 * absrfft, axis=axis) / sumabsrfft
        return m4 / spread**2

    def process(self, data, resd):
        result = self.kurtosis(
            self.frequencies,
            resd['AbsRfft'],
            resd['SumAbsRfft'],
            resd['SpectralCentroid'],
            resd['SpectralSpread'],
            self.axis)
        return result


class SpectralFlatness(Feature):
    """Flatness of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the flatness will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield AbsRfft

    def process(self, data, featuredata):
        return flatness(featuredata['AbsRfft'], self.axis)


class SpectralFlux(Feature):
    """Flux of AbsRfft.

    Parameters
    ----------
    axis : int
        Axis along the flux will be calculated, default=0.

    """
    axis = Parameter(0)

    def requires(self):
        yield AbsRfft

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
    """Crest Factor of AbsRfft"""
    def requires(self):
        yield AbsRfft

    def process(self, data, result):
        return crest_factor(result['AbsRfft'])


class SpectralRolloff(Feature):
    """Rolloff from AbsRfft.

    The spectral rolloff is the frequency where the kappa percentage of
    energy is below and the 1-kappa percentage of energy is above.

    Parameters
    ----------
    kappa : scalar {0...1}
        Default 0.95
    """

    kappa = Parameter(0.95)

    def requires(self):
        yield AbsRfft

    def on_start(self, source, featureset, sink):
        self.samplerate = source.samplerate

    def process(self, data, result):
        return rolloff(result['AbsRfft'], self.samplerate, self.kappa)


class SpectralSlope(Feature):
    # TODO: Test
    def requires(self):
        yield AbsRfft

    def on_start(self, source, features, sink):
        self.frequencies = np.array([features['AbsRfft'].frequencies]).T

    def process(self, data, resd):
        absrfft = np.array(resd['AbsRfft'])
        absrfft -= np.mean(absrfft)
        w = np.linalg.lstsq(self.frequencies, absrfft)[0]
        return w[0]
