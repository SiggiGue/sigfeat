import abc
import six
import textwrap

from collections import OrderedDict
from .parameter import ParameterMixin, Parameter
from .metadata import MetadataMixin


@six.add_metaclass(abc.ABCMeta)
class Feature(ParameterMixin, MetadataMixin):
    name = Parameter()

    def __init__(self,  requirements=None, **params):
        if 'name' not in params and not self.name.default:
            params['name'] = self.__class__.__name__
        self._hidden = False
        self.init_parameters(params)
        self._requirements = requirements

    def requires(self):
        return []

    def _requires(self):
        if self._requirements:
            return self._requirements
        return self.requires()

    @abc.abstractmethod
    def process(self, block, index, results, featureset, sink):
        print('Processing', self.__class__.__name__)

    def dependencies(self):
        return self._requires()

    def new(self):
        return self.__class__(
            requirements=self._requirements, **dict(self.params))

    @property
    def fid(self):
        return self.__class__.__name__, str(self.params)

    @property
    def hidden(self):
        return self._hidden


def hide(feature):
    feature._hidden = True
    return feature


def gen_dependencies(*features):
    for feature in features:
        for dep in feature.dependencies():
            yield from gen_dependencies(dep)
        yield feature


class FeatureSet(OrderedDict):

    def __init__(self, *features):
        OrderedDict.__init__(self)
        self.add_features(*features)

    @staticmethod
    def _mkfeatureset(*feats):
        return OrderedDict((i.fid, i) for i in tuple(
                    gen_dependencies(*feats)))

    def add_features(self, *features):
        for feat in features:
            if isinstance(feat, Feature):
                self.update(self._mkfeatureset(feat))
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
