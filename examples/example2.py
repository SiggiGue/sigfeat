from sigfeat.source import SoundFileSource
from sigfeat.preprocess import MeanMix
from sigfeat import features as fts
from sigfeat.sink import DefaultDictSink
from sigfeat.extractor import Extractor


extractor = Extractor(
    fts.SpectralFluxAbsRfft(),
    fts.SpectralCentroidAbsRfft(),
    fts.SpectralFlatnessAbsRfft(),
    fts.SpectralCrestFactorAbsRfft(),
    fts.CrestFactor(),
    fts.ZeroCrossingCout(),

)


if __name__ == '__main__':
    from pylab import plt, np

    src = MeanMix(SoundFileSource('bonsai.wav'))
    sink = DefaultDictSink()
    extractor.extract(src, sink)

    plt.figure(src.name)
    for l, r in sink['results'].items():
        plt.plot(r/np.max(np.abs(r)), label=str(l))
    plt.legend()
    plt.show()
