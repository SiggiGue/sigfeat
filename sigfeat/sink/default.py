from .sink import Sink
from collections import defaultdict


class DefaultDictSink(Sink, dict):
    """DefaultDictSink

    The receive_append method appends input to 'results' defaultdict(list)

    """
    def __init__(self):
        self.results = defaultdict(list)
        self['results'] = self.results

    def receive(self, datad):
        """Updates the dict with given ``datad`` dictionary."""
        self.update(datad)

    def receive_append(self, resultd):
        """Appends given ``resultd`` dict to list fields in this dict."""
        for name, res in resultd.items():
            self.results[name] += [res]
