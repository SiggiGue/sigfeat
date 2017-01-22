from collections import deque

import numpy as np

from ..feature import Feature
from ..parameter import Parameter


class Delta(Feature):
    """Returns a diffenetiated version of the given feature."""
    axis = Parameter(0)
    order = Parameter(1)

    def __init__(self, feature, **parameters):
        self.unroll_parameters(parameters)
        self.feature = feature
        self.name = ''.join((self.order * 'd', feature.name))
        self.values = deque([None]*(self.order+1), maxlen=self.order+1)
        self._requirements = []

    def requires(self):
        yield self.feature

    def process(self, data, resultd):
        self.values.append(resultd[self.feature.name])
        if None not in self.values:
            return np.diff(np.array(self.values), self.order, axis=self.axis)
