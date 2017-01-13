from pylab import np, randn, plt
import pandas as pd

from sigfeat.source import ArraySource
from sigfeat.feature import Feature
from sigfeat.extractor import Extractor
from sigfeat.sink import DefaultDictSink


class Index(Feature):
    def process(self, data, result):
        return data[1]


class Abs(Feature):
    def process(self, data, result):
        return np.abs(data[0])


class MS(Feature):
    def process(self, data, result):
        return np.mean(data[0]*data[0])


class RMS(Feature):
    """Calculates the Root Mean Square value using the MS() feature.

    """
    def requires(self):
        return [MS().hide()]

    def process(self, data, result):
        return np.sqrt(result['MS'])


class Peak(Feature):
    def requires(self):
        return [Abs().hide(True)]

    def process(self, data, result):
        return np.max(result['Abs'])


ars = ArraySource(randn(3600*44100), 44100, blocksize=4096, overlap=0)
extractor = Extractor(RMS(), Peak(), MS())
sink = DefaultDictSink()

extractor.extract(ars, sink)
print(sink)


df = pd.DataFrame(sink['results'])
df.plot()
plt.show()
# for res in extractor.extract(ars):
#     print(res)
