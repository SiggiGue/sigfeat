import pytest
from sigfeat.base import Source
from sigfeat.source.soundfile import SoundFileSource
from sigfeat.source.array import ArraySource


def test_source():
    with pytest.raises(TypeError):
        Source()

    assert Source.generate(1) == NotImplemented

    class S(Source):
        def generate(self):
            yield from range(5)

    s = S()
    assert list(s) == list(range(5))


def test_array_source():
    a = ArraySource(
        list(range(10)),
        blocksize=2,
        overlap=1,
        samplerate=1)
    last = 0
    for i, b in enumerate(a):
        data, index = b
        assert len(b) == 2
        assert last == data[0]
        last = data[-1]


def create_soundfile():
    import numpy as np
    from io import BytesIO
    from soundfile import write, SoundFile
    bio = BytesIO()
    x = np.random.randn(10*2048)
    x /= np.max(np.abs(x))
    write(bio, x, 44100, format='WAV')
    bio.seek(0)
    return SoundFile(bio)


def test_sound_file_source():
    sf = create_soundfile()
    src = SoundFileSource(
        sf,
        blocksize=2048,
        overlab=1024)
    for data in src:
        blk, idx = data
        assert idx % 1024 == 0
        assert len(blk) == 2048

    with pytest.raises(Exception):
        src = SoundFileSource('asdlfjhhj987.warv')

if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
