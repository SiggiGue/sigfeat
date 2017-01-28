"""Implements a simple preprocess base class.

For processing blocks of a source before extracting features.
The may be some preemphasis/winowing etc. needed before feature
extraction. Instances of Preprocess subclasses are consumed by the Extractor.

"""

import abc
import six

from ..source import Source


@six.add_metaclass(abc.ABCMeta)
class Preprocess(Source):
    """Preprocess base class.

    Behaves like a source. But you mus ovrride the process method.

    Examples
    --------

    src = YourPreprocess(YourSource(...))

    extractor.extract(src, ...)

    """
    def __init__(self, source, **parameters):
        self.source = source
        self.add_metadata('parent', dict(self.source.metadata))
        self.add_metadata('samplerate', self.source.samplerate)
        self.add_metadata('channels', self.source.channels)
        self.fetch_metadata_as_attrs()

        src_params = dict(source.parameters)
        src_params.update(parameters)
        self.unroll_parameters(src_params)

    def generate(self):
        for data in self.source:
            yield self.process(data)  # pragma: no coverage

    @abc.abstractmethod
    def process(self, data):
        """Override this method.

        Parameters
        ----------
        data : data from source

        """
        return data  # pragma: no coverage
