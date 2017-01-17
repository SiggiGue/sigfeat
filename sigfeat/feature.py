"""This module implements the abstract base Feature class.

:py:class:`Feature` is subclassed for implementing Signal Features.
:py:class:`FeatureSet` is used to create unique collections of multiple
Feature instances and is used in :py:class:`Extractor`.

TODO: What happens if i have a hidden feature but add a instance of tha same
that is not hidden? Mus be solved in the FeatureSet!
TODO: Handle labels for multidimensional feature output.

"""

import abc
import six

from inspect import isclass
from collections import OrderedDict
from .parameter import ParameterMixin
from ._metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Feature(ParameterMixin, MetadataMixin):
    """Abstract Feature Base Class

    Parameters
    ----------
    name : str
    requirements : Feature instances iterable
        To override those returned by self.requires().

    Notes
    -----
    At least the :py:meth:`process` method must be overridden.
    If your implemented feature depends on the results of another feature,
    you must override the :py:meth:`requires` method returning an iterable
    e.g. list of feature instances.

    """
    _hidden = False

    def __init__(self,  name=None, requirements=None, **parameters):
        """Returns a Feature instance.

        Provide feature-parameters as keyword arguments.

        """
        if not name:
            name = self.__class__.__name__
            # if hasattr(self.name, 'default'):
            #     if self.name.default:
            #         name = self.name.default
        self.name = name
        if requirements:
            self._requirements = list(requirements)
        else:
            self._requirements = list()

        self.unroll_parameters(parameters)
        self.validate_name()

        self.add_metadata(
            'name', self.name)
        self.add_metadata(
            'dependencies', [str(i) for i in self.dependencies()][1:])

    def on_start(self, source, featureset, sink):
        """Override this method if your feature needs some initialization.

        The Extractor will give you kwargs source, featureset and sink.

        """
        pass

    def requires(self):
        """Override this method if your feature depends on other features.

        You can return Feature classes and Feature instances you need for
        your feature.

        """
        return []

    @abc.abstractmethod
    def process(self, data, result):
        """Override this method returning process results.

        The data from the processed source will be in data=block,index,...,
        The results from features your feature requires will be in the
        resuld argument as dictionary.

        Parameters
        ----------
        data : block, index
            Or the data yielded from your source.
        result : Result, dictlike
            The results from other features of current data.
            If you provide your required features in the requires() method,
            you will be able to access the results from those features.

        Returns
        -------
        resultdata : e.g. scalar, ndarray or other custom types.
            This is the feature result for the current data (block).

        """
        print('Processing', self.__class__.__name__)

    def on_finished(self, source, featureset, sink):
        """Override this method to be run after extraction.

        Parameters
        ----------
        source : source
        featureset : OrderedDict of features
        sink : Sink

        """
        pass

    def dependencies(self):
        """Yields the dependencies of this feature."""
        yield self
        if self._requirements:
            requirements = self._requirements
        else:
            requirements = self.requires()
        for feature in requirements:
            if not isclass(feature):
                yield from feature.dependencies()
            else:
                yield feature

    def gen_dependencies_instances(self, autoinst=True, err_missing=True):
        """Checks deps for being instance or class and yields instances."""
        deps = list(self.dependencies())
        for dep in deps:
            if isclass(dep):
                isin = [isinstance(d, dep) for d in deps]
                if any(isin):
                    yield deps[deps.index(isin)]
                elif err_missing:
                    raise ValueError(
                         'Must provide a Feature Instance of {}'.format(
                             dep))
                elif autoinst:
                    yield from dep().gen_dependencies_instances(
                        autoinst=autoinst, err_missing=err_missing)
            else:
                yield dep

    def featureset(self, new=False, autoinst=False, err_missing=True):
        """Returns an ordered dict of all features unique in name.

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
        deps = reversed(tuple(self.gen_dependencies_instances(
            autoinst, err_missing)))
        if new:
            deps = [d.new() for d in deps]
        return OrderedDict((feat.name, feat) for feat in deps)

    def validate_name(self):
        """Checks for uniqueness of feature name in all dependent features."""
        names = [f.name for f in self.featureset(
            autoinst=False, err_missing=False).values()]
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

    def __repr__(self):
        return "".join(self.fid)


class HiddenFeature(Feature):
    _hidden = True


def _validate_featureset(featureset):
    """Returns true if all required feature instances are available.
    Else raises an error."""
    for name, feature in featureset.items():
        for req in feature.requires():
            if hasattr(req, 'name') and isinstance(req.name, str):
                name = req.name
            else:
                name = req.__name__

            if name not in featureset:
                raise ValueError(
                    'You must provide a feature instance of {} '
                    'or try set autoinst=True if defaults are ok.'.format(
                        req))
    return featureset


def features_to_featureset(features,
                           new=False,
                           autoinst=False):
    """Returns an featureset of given features distinct in names.

    Parameters
    ----------
    features : iterable
    new : reinitialize features as new instances.
    autoinst : auto initialize missing feature classes if required.

    """
    featsets = (feat.featureset(
        new=new,
        autoinst=autoinst,
        err_missing=False) for feat in features)
    featureset = next(featsets)
    for fset in featsets:
        featureset.update(fset)

    return _validate_featureset(featureset)


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
