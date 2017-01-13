import pkg_resources as _pkg_resources


def get_version():
    return _pkg_resources.get_distribution(
        __name__).version


__version__ = get_version()
