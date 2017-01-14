from pylab import np, plt
from scipy.signal import chirp
import pandas as pd

from sigfeat.parameter import Parameter
from sigfeat.source import ArraySource
from sigfeat.features import Feature
from sigfeat.features import SpectralFlux
from sigfeat.features import SpectralCentroid
from sigfeat.features import SpectralFlatness
from sigfeat.features import AbsSpectrumRfft
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink


class Index(Feature):
    """Index of source."""
    def process(self, data, result):
        return data[1]


class Abs(Feature):
    """Absolute value."""
    def process(self, data, result):
        return np.abs(data[0])


class MS(Feature):
    """Mean Square."""
    axis = Parameter(default=0)

    def process(self, data, result):
        return np.mean(data[0]*data[0], axis=self.axis)


class RMS(Feature):
    """Root Mean Square."""
    def requires(self):
        return [MS().hide()]

    def process(self, data, result):
        return np.sqrt(result['MS'])


class Peak(Feature):
    """Absolute Peak value."""
    def requires(self):
        return [Abs().hide(True)]

    def process(self, data, result):
        return np.max(result['Abs'])

t = np.linspace(0, 60, 60*44100)
x = np.sin(2*np.pi*1000*t)
x = chirp(
    t,
    f0=10,
    t1=60,
    f1=40,
    method='log'
    )

asrc = ArraySource(
    x,
    samplerate=44100,
    blocksize=4096,
    overlap=2048)

aspec = AbsSpectrumRfft(nfft=asrc.blocksize)
features = (
    aspec,
    Index(),
    RMS(),
    Peak(),
    MS(),
    SpectralFlux(),
    SpectralCentroid(),
    SpectralFlatness(),
    )

extractor = Extractor(*features)

sink = DefaultDictSink()

extractor.extract(asrc, sink)
df = pd.DataFrame(sink['results']).set_index('Index')
df.plot()
plt.show()
# for res in extractor.extract(ars):
#     print(res)
