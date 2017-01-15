class Result(dict):
    """Result dict. Behaves 'immutable' to the Feature.process method.

    Just a simple dict to hold the results from features.

    """
    __slots__ = ()

    def __setitem__(self, *args):
        raise TypeError('`Result` object does not support item assignment.')

    def _setitem(self, key, value):
        dict.__setitem__(self, key, value)
