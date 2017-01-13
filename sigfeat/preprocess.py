"""Implements a simple preprocess base class.

For processing blocks of a source before extracting features.
The may be some preemphasis/winowing etc. needed before feature
extraction. Instances of Preprocess subclasses are consumed by the Extractor.

"""

import abc
import six
from .parameter import ParameterMixin


@six.add_metaclass(abc.ABCMeta)
class Preprocess(ParameterMixin):
    # TODO: Adding MetadataMixin? i think tha makes sense.
    @abc.abstractmethod
    def process(self, block, source):
        """Override this method."""
        return block

    def new(self):
        return self.__class__(**dict(self.parameters))


class WindowPreprocess(Preprocess):
    def __init__(self, window, source=None, **kw_get_win):
        if isinstance(window, str) and source:
            from scipy.signal.windows import get_window
            window = get_window(window, Nx=source.blocksize, **kw_get_win)
        else:
            from numpy import asarray
            window = asarray(window, dtype=float)
        self.window = window

    def process(self, block, source):
        return block * self.window
