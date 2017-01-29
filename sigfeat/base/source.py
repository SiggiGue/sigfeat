"""Implements abstract Source base class"""

import abc
import six
from .parameter import ParameterMixin, Parameter
from .metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Source(ParameterMixin, MetadataMixin):
    """Base Source Class.

    The parameters are mandatory for all kinds of sources.
    Every source must implement a ``.generate()`` method
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
        """Override this method. It must yield ``data``.

        Usually ``data = (block, index)``

        """
        return NotImplemented
