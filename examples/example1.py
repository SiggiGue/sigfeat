from pylab import np, plt
from scipy.signal import chirp

from sigfeat import Extractor
from sigfeat.source import ArraySource
from sigfeat.preprocess import MeanMix
from sigfeat.sink import DefaultDictSink
from sigfeat import feature as fts


t = np.linspace(0, 2, 2*44100)
# x = np.sin(2*np.pi*1000*t)
x = chirp(
    t,
    f0=1000,
    t1=2,
    f1=4000,
    method='log'
    )

src = ArraySource(
    np.tile(x, (2, 1)).T,
    samplerate=44100,
    blocksize=2048,
    overlap=1024)

features = (
    fts.Index(),
    fts.RootMeanSquare(),
    fts.Peak(),
    fts.CrestFactor(),
    fts.ZeroCrossingRate(),
    fts.SpectralFlux(),
    fts.SpectralCentroid(),
    fts.SpectralFlatness(),
    fts.SpectralCrestFactor(),
    )

extractor = Extractor(*features)

sink = DefaultDictSink()

extractor.extract(MeanMix(src), sink)

res = sink['results']
idx = res.pop('Index')

for l, r in res.items():
    plt.figure(l)
    plt.title(l)
    plt.plot(idx, r, label=str(l))
plt.legend()
plt.show()
