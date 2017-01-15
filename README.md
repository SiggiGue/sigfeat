#Signal Feature Extraction Framework

Developed with focus on audio signals.

##Architecture

The main base classes are: Source, Feature, Extractor and Sink.

Seven principal classes build up the whole framework:

1. Feature
2. Source
3. Sink
4. Extractor
5. Preprocess
6. Parameter
7. Result


##Simple Example usage

```python

from pylab import plt, np

from sigfeat.source import SoundFileSource
from sigfeat.preprocess import MeanMix
from sigfeat import features as fts
from sigfeat.sink import DefaultDictSink
from sigfeat.extractor import Extractor


extractor = Extractor(
    fts.SpectralFlux(),
    fts.SpectralCentroid(),
    fts.SpectralFlatness(),
)


src = MeanMix(SoundFileSource('bonsai.wav'))
sink = DefaultDictSink()
extractor.extract(src, sink)

plt.figure(src.name)
for l, r in sink['results'].items():
    plt.plot(r/np.max(np.abs(r)), label=str(l))
plt.legend()
plt.show()

```
