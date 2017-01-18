from sigfeat.source import SoundFileSource
from sigfeat.preprocess import MeanMix
from sigfeat import features as fts
from sigfeat.sink import DefaultDictSink
from sigfeat.extractor import Extractor


extractor = Extractor(
    fts.SpectralFlux(),
    fts.SpectralCentroid(),
    fts.SpectralFlatness(),
    fts.SpectralRolloff(),
    fts.SpectralCrestFactor(),
    fts.CrestFactor(),
    fts.ZeroCrossingRate(),
    fts.RootMeanSquare(),
    fts.Peak(),
    fts.Kurtosis(),
    fts.Skewness(),
)


if __name__ == '__main__':
    from pylab import plt
    import pandas as pd
    from pandas.tools.plotting import scatter_matrix

    src = MeanMix(SoundFileSource(
        'Test.wav',
        blocksize=4096,
        overlap=2048))
    sink = DefaultDictSink()
    extractor.extract(src, sink)

    plt.figure(src.source.name)
    for l, r in sink['results'].items():
        plt.plot(r, 'o-', label=str(l))
    plt.legend()

    df = pd.DataFrame(sink['results'])
    scatter_matrix(df)
    plt.show()
