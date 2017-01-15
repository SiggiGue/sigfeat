"""Implements a simple preprocess base class.

For processing blocks of a source before extracting features.
The may be some preemphasis/winowing etc. needed before feature
extraction. Instances of Preprocess subclasses are consumed by the Extractor.

"""

import abc
import six
import numpy as np

from .parameter import Parameter
from .feature import Feature


@six.add_metaclass(abc.ABCMeta)
class Preprocess(Feature):
    # TODO: Adding MetadataMixin? i think tha makes sense.

    def on_start(self, source):
        pass

    @abc.abstractmethod
    def process(self, data, **kwargs):
        """Override this method."""
        pass

    def new(self):
        return self.__class__(**dict(self.parameters))


class MonoMix(Preprocess):
    axis = Parameter(-1)
    channels = Parameter(1)

    def process(self, data):
        block = np.sum(data[0], axis=self.axis).flatten()
        data = block, *data[1:]
        return data

#
# class WindowPreprocess(Preprocess):
#     def __init__(self, window, source=None, **kw_get_win):
#         if isinstance(window, str) and source:
#             from scipy.signal.windows import get_window
#             window = get_window(window, Nx=source.blocksize, **kw_get_win)
#         else:
#             from numpy import asarray
#             window = asarray(window, dtype=float)
#         self.window = window
#
#     def process(self, block):
#         return block * self.windo
