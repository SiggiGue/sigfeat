"""Implements a simple preprocess base class.

For processing blocks of a source before extracting features.
The may be some preemphasis/winowing etc. needed before feature
extraction. Instances of Preprocess subclasses are consumed by the Extractor.

"""

import abc
import six
import numpy as np

from .parameter import Parameter
from .source import Source


# TODO Preprocess mus provide a source not being a feature.

@six.add_metaclass(abc.ABCMeta)
class Preprocess(Source):

    def __init__(self, source, **parameters):
        self.source = source
        for k, v in self.source.metadata:
            self.add_metadata(k, v)
        self.add_metadata('sourceclass', self.source.__class__.__name__)
        self.add_metadata('preprocessclass', self.__class__.__name__)
        self.fetch_metadata_as_attrs()

        src_params = dict(source.parameters)
        src_params.update(parameters)
        self.unroll_parameters(src_params)

    def generate(self):
        for data in self.source:
            yield self.process(data)

    @abc.abstractmethod
    def process(self, data):
        """Override this method."""
        return data


class SumMix(Preprocess):
    axis = Parameter(-1)
    channels = Parameter(1)

    def process(self, data):
        if self.source.channels > 1:
            block = np.sum(data[0], axis=self.axis).flatten()
            data = block, *data[1:]
        return data


class MeanMix(Preprocess):
    axis = Parameter(-1)
    channels = Parameter(1)

    def process(self, data):
        if self.source.channels > 1:
            block = np.mean(data[0], axis=self.axis).flatten()
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
