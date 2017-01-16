from pylab import np, plt
from scipy.signal import chirp

from sigfeat.parameter import Parameter
from sigfeat.source import ArraySource
from sigfeat.preprocess import MeanMix
from sigfeat.features import Feature
from sigfeat.features import SpectralFluxAbsRfft
from sigfeat.features import SpectralCentroidAbsRfft
from sigfeat.features import SpectralFlatnessAbsRfft
from sigfeat.features import AbsRfft
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
    axis = Parameter(0)

    def requires(self):
        return [Abs().hide(True)]

    def process(self, data, result):
        return np.max(result['Abs'], axis=self.axis)

t = np.linspace(0, 60, 60*44100)
x = np.sin(2*np.pi*1000*t)
x = chirp(
    t,
    f0=10,
    t1=60,
    f1=4000,
    method='log'
    )

src = ArraySource(
    np.tile(x, (2, 1)).T,
    samplerate=44100,
    blocksize=4096,
    overlap=2048)

# src = SoundFileSource(
#     '86jazzy_mix07.wav',
#     blocksize=1024,
#     overlap=512)

aspec = AbsRfft()
features = (
    aspec,
    Index(),
    RMS(),
    Peak(),
    MS().hide(),
    SpectralFluxAbsRfft(),
    SpectralCentroidAbsRfft(),
    SpectralFlatnessAbsRfft(),
    )

extractor = Extractor(*features)

sink = DefaultDictSink()

extractor.extract(MeanMix(src), sink)

plt.figure()
for l, r in sink['results'].items():
    plt.plot(r/np.max(np.abs(r)), label=str(l))
plt.legend()
plt.show()
