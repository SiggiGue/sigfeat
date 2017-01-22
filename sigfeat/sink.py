"""Implements abstract Sink class and some usefull Sink subclasses

- :py:class:`DefaultDictSink`:  Sink receiving results in dictionary.

"""

import abc
import six
from collections import defaultdict


@six.add_metaclass(abc.ABCMeta)
class Sink(object):
    """Sink base class."""

    @abc.abstractclassmethod
    def receive(self, datad):
        """Shall receive dictionaries directly written to source."""
        pass  # pragma: no coverage

    @abc.abstractclassmethod
    def receive_append(self, resultd):
        """Shall receive result dictionaries appending data to fields."""
        pass  # pragma: no coverage


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
