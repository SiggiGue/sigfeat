"""Implements abstract Source and Source subclasses.

- :py:class:`ArraySource`: Source for numpy arrays with slicing functionaliy.
- :py:class:`SoundFileSource`: Source for SoundFiles.
"""

import abc
import six
from .parameter import ParameterMixin, Parameter
from .metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Source(ParameterMixin, MetadataMixin):
    """Base Source Class.

    The parameters are mandatory for all kinds of sources.
    Every source must implement a `.blocks()` method
    returning a generator that yields the signal blocks.
    Source must have samplerate and channels as metadata.

    Parameters
    ----------
    blocksize : int
    overlap : int
    samplerate : scalar
    channels : int

    """
    blocksize = Parameter(default=1024)
    overlap = Parameter(default=0)

    def __init__(self, **parameters):
        self.unroll_parameters(parameters)
        self.add_metadata('samplerate', None)
        self.add_metadata('channels', None)
        self.fetch_metadata_as_attrs()

    def __iter__(self):
        return self.generate()

    @abc.abstractmethod
    def generate(self):
        """Must yield Result."""
        return NotImplemented


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
