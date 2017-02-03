#**SigFeat**: A Signal Feature Extraction Framework

[![Build Status](https://travis-ci.org/SiggiGue/sigfeat.svg?branch=master)](https://travis-ci.org/SiggiGue/sigfeat)

[![Coverage Status](https://coveralls.io/repos/github/SiggiGue/sigfeat/badge.svg?branch=master)](https://coveralls.io/github/SiggiGue/sigfeat?branch=master)


This library is developed with focus on audio signals but it's base functionality is
generalized to all kinds of (time)-signals.

The key advantages of this library are:

- SigFeat provides an **appropriate and simple abstraction layer** for the feature extraction problem:

  ![](./docs/diagram.png)

- SigFeat **minimizes computational cost** by avoiding repeated computation of (interim) results. (For instance if many features depend on a result of another feature, this feature result is only computed once. Simple example: all Spectral features use one FFT output.)
- SigFeat has a **low memory footprint** due to generators (except your own defined features blow up memory or your sources load all data at once...).

See the [examples](https://github.com/SiggiGue/sigfeat/tree/develop/examples) folder, the [feature](https://github.com/SiggiGue/sigfeat/tree/develop/sigfeat/feature), [source](https://github.com/SiggiGue/sigfeat/tree/develop/sigfeat/source) and [sink](https://github.com/SiggiGue/sigfeat/tree/develop/sigfeat/sink) subpackage to check the intended usage of the library.

Some parts of the library (the parameters module) are inspired by [luigi](https://github.com/spotify/luigi).

##Simple Example Usage

```python

from pylab import plt, np

from sigfeat import Extractor
from sigfeat.feature import SpectralFlux, SpectralCentroid, SpectralFlatness
from sigfeat.source.soundfile import SoundFileSource
from sigfeat.preprocess import MeanMix
from sigfeat.sink import DefaultDictSink


extractor = Extractor(
    SpectralFlux(),
    SpectralCentroid(),
    SpectralFlatness(),
)


source = MeanMix(SoundFileSource('bonsai.wav'))

sink = extractor.extract(source, DefaultDictSink())


plt.figure()

for label, result in sink['results'].items():
    plt.plot(result), label=str(label))

plt.legend()
plt.show()

```


##Structure

The main base classes are: Source, Feature, Extractor and Sink.

Seven principal classes build up the whole framework:

1. Feature
2. Source
3. Sink
4. Extractor
5. Preprocess
6. Parameter
7. Result


Additionally to the basic framework, some commonly known features,
useful sources and sinks are implemented. But it is not the focus
of this project to provide a complete feature collection.
It is up to the user to define own features, sources
and sinks. This framework allows many different usage scenarios... be creative.


## Requirements

The base framework only depends on six (sigfeat.base and sigfeat.extractor).

The namespaces sigfeat.feature .source and .preprocess depend on the
scipy stack, soundfile and pyfilterbank. It depends on which submodules
are used.
