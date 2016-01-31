"""This module implements the abstract base Feature class.

:py:class:`Feature` is subclassed for implementing Signal Features.
:py:class:`FeatureSet` is used to create unique collections of multiple
Feature instances and is used in :py:class:`Extractor`.

TODO: What happens if i have a hidden feature but add a instance of tha same
that is not hidden? Mus be solved in the FeatureSet!

"""

import abc
import six
import textwrap

from collections import OrderedDict
from .parameter import ParameterMixin, Parameter
from .metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Feature(ParameterMixin, MetadataMixin):
    """Abstract Feature Base Class

    Parameters
    ----------
    name : str
    requirements : optional to def requires(self)

    Notes
    -----
    At least the :py:meth:`process` method must be overridden.
    If your implemented feature depends on the results of another feature,
    you must override the :py:meth:`requires` method returning an iterable
    of feature instances.
    """
    name = Parameter()

    def __init__(self,  requirements=None, **params):
        if 'name' not in params and not self.name.default:
            params['name'] = self.__class__.__name__
        self._hidden = False
        self.init_parameters(params)
        self._requirements = requirements
        self.init()

    def init(self):
        """Override this method if your feature needs some initialization."""
        pass

    def requires(self):
        """Override this method if your feature depends on other features."""
        return []

    @abc.abstractmethod
    def process(self, block, index, results, featureset, sink):
        """Override this method returning process results."""
        print('Processing', self.__class__.__name__)

    def dependencies(self):
        """Returns the dependencies of this feature."""
        if self._requirements:
            return self._requirements
        return self.requires()

    def new(self):
        """Returns new initial feature instance with same parameters."""
        return self.__class__(
            requirements=self._requirements, **dict(self.params))

    @property
    def fid(self):
        """Returns the feature identifying tuple."""
        return self.__class__.__name__, str(self.params)

    @property
    def hidden(self):
        """Returns whether the feature is hidden or not."""
        return self._hidden


def hide(feature):
    """Returns hidden feature."""
    feature._hidden = True
    return feature


def gen_dependencies(*features):
    """Yields dependencies of the given features and the features itself."""
    for feature in features:
        for dep in feature.dependencies():
            yield from gen_dependencies(dep)
        yield feature


class FeatureSet(OrderedDict):
    """FeatureSet
    Container for features, resolving the dependencies and
    ensuring uniqueness of features to be processed.

    Parameters
    ----------
    *features : instances of Feature

    """

    def __init__(self, *features):
        OrderedDict.__init__(self)
        self.add_features(*features)

    def add_features(self, *features):
        """Adds given features to this featureset."""
        for feat in features:
            if isinstance(feat, Feature):
                self.update(self._make_featureset(feat))
            elif isinstance(feat, FeatureSet):
                self.update(feat)
            else:
                raise ValueError(textwrap.dedent(
                    """Features must either instances of
                    Feature or instances of FeatureSet.
                    The given ""{}" is of type {}""".format(feat, type(feat))))

    def reset_features(self):
        """This creates new instances with same parameters.
        So features are in initial state.
        """
        for key, value in self.items():
            self[key] = value.new()

    @staticmethod
    def _make_featureset(*features):
        """Returns an OrderedDict of given features and dependencies."""
        return OrderedDict((feat.fid, feat) for feat in tuple(
                    gen_dependencies(*features)))
