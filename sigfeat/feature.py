import textwrap
from collections import OrderedDict
from .parameter import AbstractParameterClassMixin


class Feature(AbstractParameterClassMixin):
    def __init__(self,  requirements=None, **params):
        self.init_parameters(params)
        self._requirements = requirements

    def requires(self):
        return []

    def _requires(self):
        if self._requirements:
            return self._requirements
        return self.requires()

    def process(self, block, source, sink):
        print('Processing', self.__class__.__name__)
        sink.receive(block)

    def dependencies(self):
        return self._requires()

    # def output(self):
    #     return []

    # def input(self):
    #     return self._requires()

    @property
    def fid(self):
        return self.__class__.__name__, str(self.params)


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
