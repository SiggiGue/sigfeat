import pkg_resources as _pkg_resources

from .base import Source
from .base import Preprocess
from .base import Feature
from .base import Parameter
from .base import Sink

from .extractor import Extractor

__all__ = [
    'Source',
    'Preprocess',
    'Feature',
    'Parameter',
    'Sink',
    'Extractor'
]


def get_version():
    return _pkg_resources.get_distribution(
        __name__).version


__version__ = get_version()
