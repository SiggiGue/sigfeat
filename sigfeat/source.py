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
    noverlap = Parameter(default=512)

    def __init__(self, **params):
        self.init_parameters(params)
        self.init_metadata()

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
    sfname = Parameter(default='')

    def __init__(self, sf=None, **params):
        self.init_parameters(params)
        self._init_soundfile(sf)

    def _init_soundfile(self, sf):
        if isinstance(sf, str):
            from soundfile import SoundFile
            self._params.update({'sfname': sf})
            self.sf = SoundFile(sf)
        else:
            self.sf = sf
            if 'sfname' not in self._params:
                self._params.update({'sfname': sf.name})
