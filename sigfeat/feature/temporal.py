import numpy as np
from scipy.stats import skew, kurtosis

from ..base import Feature
from ..base import HiddenFeature
from ..base import Parameter

from .common import centroid
from .common import flatness
from .common import moments
from .common import zero_crossing_count


class CrestFactor(Feature):
    """Crest Factor of Source data.

    Depends on Peak and RootMeanSquare features.

    .. math::
        CF_m = \\frac{\max{(|x_m|)}}{x_{m, RMS}}

    Parameters
    ----------
    axis: int

    """
    axis = Parameter(0)

    def requires(self):
        yield Peak(axis=self.axis)
        yield RootMeanSquare(axis=self.axis)

    def process(self, data, resd):
        return resd['Peak'] / resd['RootMeanSquare']


class ZeroCrossingRate(Feature):
    """Zero Crossings Rate of Source data.

    .. math::
        ZCR_m = \\frac{f_s}{N} \\lfloor 0.5 + \\frac{1}{2}
                \\sum_n |\\mathrm{sgn}(x_m[n+1])-\\mathrm{sgn}(x_m[n])| \\rfloor

    """
    # TODO axis parameter

    def on_start(self, source, featureset, sink):
        self.factor = source.samplerate / source.blocksize

    def process(self, data, result):
        return zero_crossing_count(data[0]) * self.factor


class StatMoments(Feature):
    """Estimates mu, variance, skewness and kurtosis of Source data."""
    labels = ['mu', 'mu_variance', 'mu_skewness', 'mu_kurtosis']

    def process(sefl, data, result):
        return moments(data[0])


class SquaredSignal(HiddenFeature):
    """Squared Signal data (as hidden feature).

    .. math::
        y_m[n] = x_m[n]^2

    """

    def process(self, data, result):
        sig = data[0]
        return sig*sig


class AbsSignal(HiddenFeature):
    """Abs from source data (as hidden feature).

    .. math::
        y_m[n] = | x_m[n] |

    """
    def process(self, data, result):
        return np.abs(data[0])


class CentroidAbsSignal(Feature):
    """Experimental Centroid of abs source data.

    .. math::

        C = \\frac{\\sum_n t[n]x_m[n]}{
                \\frac{1}{N}\\sum_n x_m[n]}

    Parameters
    ----------
    axis: int
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

    .. math::

        F_m = \\frac{\\left(\\Pi_n |x_m[n]| \\right)^{1/N}}{
                \\frac{1}{N}\\sum_n |x_m[n]|}

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

    .. math::
        MS_m = \\frac{1}{N}\\sum_n x_m[n]^2

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

    .. math::
        RMS_m = \\sqrt{\\frac{1}{N}\\sum_n x_m[n]^2}

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

    .. math::
        P_m = \\max(|x_m|)

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

    Just use the RootMeanSquare instad if you do not really need the std.

    Parameters
    ----------
    axis : int
        Axis along which the feature is computed.

    See Also
    --------
    RMS

    """
    axis = Parameter(0)

    def process(self, data, result):
        return np.std(data[0], axis=self.axis)
