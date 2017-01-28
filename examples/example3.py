from sigfeat import Extractor
from sigfeat.feature import MFCC


extmfcc = Extractor(MFCC())


if __name__ == '__main__':
    from pylab import plt, np

    from sigfeat.source import SoundFileSource
    from sigfeat.preprocess import MeanMix
    from sigfeat.sink import DefaultDictSink

    src = MeanMix(SoundFileSource(
        'Test.wav',
        blocksize=4096,
        overlap=2048))

    sink = DefaultDictSink()
    extmfcc.extract(src, sink)

    plt.matshow(np.array(sink['results']['MFCC']).T)
    plt.axis('tight')
    plt.show()
