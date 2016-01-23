from .parameter import AbstractParameterMixin, Parameter
from .metadata import AbstractMetadataMixin

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
#


class Source(AbstractParameterMixin, AbstractMetadataMixin):
    blocksize = Parameter(default=1024)
    overlap = Parameter(default=0)

    def __init__(self, **params):
        self.init_parameters(params)

    def blocks(self):
        return NotImplemented


class ArraySource(Source):
    def __init__(self, array, name='', **params):
        self.init_parameters(params)
        self._array = array
        self.add_metadata('name', name)
        self.add_metadata('arraylen', len(array))

    def blocks(self):
        pass


class SoundFileSource(Source):
    name = Parameter(default='')
    blocksize = Parameter(default=1024)
    overlap = Parameter(default=0)
    frames = Parameter(default=-1)

    def __init__(self, sf=None, **params):
        if isinstance(sf, str):
            from soundfile import SoundFile
            params.update({'name': sf})
            sf = SoundFile(sf)
        else:
            if 'name' not in params:
                params.update({'name': sf.name})
        self.sf = sf
        metadata = list(self._gen_metadata_from_sf(sf))
        self.extend_metadata(metadata)
        self.init_parameters(params)

    @staticmethod
    def _gen_metadata_from_sf(sf):
        ATTRS = ['name', 'mode', 'samplerate', 'channels', 'format',
                 'subtype', 'endian']
        for attr in ATTRS:
            yield attr, getattr(sf, attr)
        yield 'lenth', len(sf)

    def blocks(self):
        yield from self.sf.blocks(
            blocksize=self.blocksize,
            overlap=self.overlap,
            frames=self.frames)
