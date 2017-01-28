from .base import Source
from .base import Parameter


class ArraySource(Source):
    """Source class for iterable arrays.

    Parameters
    ----------
    array : ndarray
        Expects an iterable array with .shape tuple.
    samplerate : int
    name : str
    blocksize : int
    overlap : int

    """

    def __init__(self, array, samplerate, name='', **parameters):
        from numpy import asarray, product
        array = asarray(array)
        self.unroll_parameters(parameters)
        self._array = array
        self.channels = product(array.shape[1:])
        self.add_metadata('name', name)
        self.add_metadata('arraylen', len(array))
        self.add_metadata('channels', self.channels)
        self.add_metadata('samplerate', samplerate)
        self.fetch_metadata_as_attrs()

    def generate(self):
        """Returns generator that yields blocks out of the array."""
        indexrange = range(
            0,
            len(self._array)-self.blocksize+1,
            self.blocksize-self.overlap)
        for index in indexrange:
            yield self._array[index:index+self.blocksize], index


class SoundFileSource(Source):
    """Source generating data from SoundFiles.

    The parameters are for the :meth:`SoundFile.blocks` method
    used for the blocks method of this Source class.

    Parameters
    ----------
    sf : SoundFile instance or str
    blocksize : int
    overlap : int
    frames : int
        The number of frames to yield from soundfile.
        If frames < 1, the file is read until the end.
    fill_value : scalar
        The value last block will filled up, if it is shorter tha blocksize.
    dtype : {'float64', 'float32', 'int32', 'int16'}, optional
        See :meth:`soundfile.SoundFile.read`.
    always_2d : bool
        Indicates wether all blocks are at least 2d numpy arrays.

    """
    frames = Parameter(default=-1)
    fill_value = Parameter(default=0)
    dtype = Parameter(default='float64')
    always_2d = Parameter(default=False)

    def __init__(self, sf=None, **parameters):
        self.unroll_parameters(parameters)
        if isinstance(sf, str):
            from soundfile import SoundFile
            sf = SoundFile(sf)

        self.sf = sf
        metadata = list(self._gen_metadata_from_sf(sf))
        self.extend_metadata(metadata)
        self.fetch_metadata_as_attrs()

    @staticmethod
    def _gen_metadata_from_sf(sf):
        ATTRS = ['name', 'mode', 'samplerate', 'channels', 'format',
                 'subtype', 'endian']
        for attr in ATTRS:
            yield attr, getattr(sf, attr)
        yield 'length', len(sf)

    def generate(self):
        """Returns generator that yields blocks from the SoundFile."""
        blocks = self.sf.blocks(
            blocksize=self.blocksize,
            overlap=self.overlap,
            frames=self.frames,
            dtype=self.dtype,
            fill_value=self.fill_value,
            always_2d=self.always_2d)
        blockshift = self.blocksize - self.overlap
        index = self.sf.tell()
        for block in blocks:
            yield block, index
            index += blockshift
        self.sf.close()
