from .parameter import AbstractParameterMixin, Parameter
from .metadata import AbstractMetadataMixin


class Source(AbstractParameterMixin, AbstractMetadataMixin):
    """Base Source Class.

    The parameters are mandatory for all kinds of sources.
    Every source must implement a `.blocks()` method
    returning a generator that yields the signal blocks.

    Parameters
    ----------
    blocksize : int
    overlap : int
    samplerate : int

    """
    blocksize = Parameter(default=1024)
    overlap = Parameter(default=0)
    samplerate = Parameter(default=44100)

    def __init__(self, **params):
        self.init_parameters(params)

    def blocks(self):
        """Must yield blocks."""
        return NotImplemented


class ArraySource(Source):
    """Source class for iterable arrays."""

    def __init__(self, array, name='', **params):
        self.init_parameters(params)
        self._array = array
        self.add_metadata('name', name)
        self.add_metadata('arraylen', len(array))
        self.fetch_metadata_as_attrs()

    def blocks(self):
        """Returns generator that yields blocks out of the array."""
        for index in range(0, len(self._array), self.blocksize-self.overlap):
            yield self._array[index:index+self.blocksize]


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
    always_2d = Parameter(default=True)

    def __init__(self, sf=None, **params):
        self.init_parameters(params)
        if isinstance(sf, str):
            from soundfile import SoundFile
            sf = SoundFile(sf)

        self.sf = sf
        self.params.update({'samplerate': sf.samplerate})
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

    def blocks(self):
        """Returns generator that yields blocks from the SoundFile."""
        yield from self.sf.blocks(
            blocksize=self.blocksize,
            overlap=self.overlap,
            frames=self.frames,
            dtype=self.dtype,
            fill_value=self.fill_value,
            always_2d=self.always_2d)


# Thought about a Block class like:
# class Block(object):
#     __slots__ = ('data', 'index', 'meta')
#
#     def __init__(self, data, index=None, meta=None):
#         self.data = data
#         self.index = index
#         self.meta = meta
#
# BUT Sourc.block() just yielding tuples is way faster!:
#
# %timeit b = Block(1243, 1234)
# The slowest run took 7.01 times longer than the fastest.
# This could mean that an intermediate result is being cached
# 1000000 loops, best of 3: 528 ns per loop
#
# %timeit b = 1243, 1234
# 100000000 loops, best of 3: 19.6 ns per loop
#
