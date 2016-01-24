import abc
import six
from .parameter import ParameterMixin


@six.add_metaclass(abc.ABCMeta)
class Preprocess(ParameterMixin):

    @abc.abstractmethod
    def process(self, block, source):
        return block

    def new(self):
        return self.__class__(**dict(self.params))


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
