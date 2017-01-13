"""This module implements the abstract base Feature class.

:py:class:`Feature` is subclassed for implementing Signal Features.
:py:class:`FeatureSet` is used to create unique collections of multiple
Feature instances and is used in :py:class:`Extractor`.

TODO: What happens if i have a hidden feature but add a instance of tha same
that is not hidden? Mus be solved in the FeatureSet!

"""

import abc
import six

from collections import OrderedDict
from .parameter import ParameterMixin, Parameter
from .metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Feature(ParameterMixin, MetadataMixin):
    """Abstract Feature Base Class

    Parameters
    ----------
    name : str

    Notes
    -----
    At least the :py:meth:`process` method must be overridden.
    If your implemented feature depends on the results of another feature,
    you must override the :py:meth:`requires` method returning an iterable
    e.g. list of feature instances.

    """
    name = Parameter()

    def __init__(self,  **parameters):
        """Returns a Feature instance.
        Provide feature-parameters as keyword arguments.
        """
        self._hidden = False
        if 'name' not in parameters:
            name = self.__class__.__name__
            if hasattr(self.name, 'default'):
                if self.name.default:
                    name = self.name.default
            parameters['name'] = name

        self.unroll_parameters(parameters)
        self.validate_name()

    def start(self, *args, **kwargs):
        """Override this method if your feature needs some initialization.

        Extractor will give you kwargs source, sink.

        """
        pass

    def requires(self):
        """Override this method if your feature depends on other features."""
        return []

    @abc.abstractmethod
    def process(self, data, resultsd):
        """Override this method returning process results."""
        print('Processing', self.__class__.__name__)

    def on_finished(self, *args, **kwargs):
        """Override this method to be run after extraction."""
        pass

    def dependencies(self):
        """Yields the dependencies of this feature."""
        yield self
        for feature in self.requires():
            yield from feature.dependencies()

    def featureset(self, new=False):
        """Returns an ordered dict of all features unique in parameters.

        The dict is ordered by the dependency tree order.

        Parameters
        ----------
        new : boolean
            All features will be reinitialized.

        Returns
        -------
        featdict : OrderedDict
            Keys are ``fid`` and values are feature instances.

        """
        deps = reversed(tuple(self.dependencies()))
        if new:
            deps = [d.new() for d in deps]
        return OrderedDict((feat.fid, feat) for feat in deps)

    def validate_name(self):
        """Checks for uniqueness of feature name in all dependent features."""
        names = [f.name for f in self.featureset().values()]
        if not len(names) == len(set(names)):
            raise ValueError(
                'You have defined duplicate feature names '
                'this is not allowed: {}.'.format(names))

    def new(self):
        """Returns new initial feature instance with same parameters."""
        return self.__class__(**dict(self.parameters))

    def hide(self, b=True):
        """Hide the feature."""
        self._hidden = bool(b)
        return self

    @property
    def fid(self):
        """Returns the feature identifying tuple."""
        return self.__class__.__name__, str(self.parameters)

    @property
    def hidden(self):
        """Returns whether the feature is hidden or not."""
        return self._hidden


# Singletons for features was planned, but not
# implemented because stateful features would not
# be threadsafe. But the baseclass for
# parametrized singletons is implemented in
# _IndividualBorgs and can be subclassed if
# multithreading is not needed.

class _IndividualBorgs(object):
    "Add this as subclass if you want parametrized singletons."
    _instances = {}

    def __new__(cls, *args, **kwargs):
        fid = cls.__name__, str(args), str(kwargs)
        if fid not in cls._instances:
            instance = object.__new__(cls)
            instance._fid = fid
            cls._instances[fid] = instance
        else:
            instance = cls._instances[fid]
        return instance
