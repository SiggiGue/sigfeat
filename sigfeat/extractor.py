"""This Module contains the central Extractor class.

The Extractor consumes features instances
and is used to extract the Features from given sources into given
Sink.

"""

from .result import Result
from .feature import features_to_featureset


class Extractor(object):
    """Feature Extractor

    Parameters
    ----------
    *features : FeatureSet/itearable of Features
    autoinst : bool
        Autoinstantiate required feature Classes if no instance in featureset.
        If False you will get errors with hint
        which feature instance to provide.
    """
    def __init__(self, *features, autoinst=True):
        self.features = features
        self.featureset = features_to_featureset(
            self.features, autoinst=autoinst)

    def _extract(self, source):
        result = Result()
        for data in source:
            for fid, feature in self.featureset.items():
                output = feature.process(
                    data,
                    result)
                result._setitem(feature.name, output)
            yield result

    def extract(self, source, sink=None):
        """Extracts features from given source into given sink.

        Parameters
        ----------
        source : Source instance
        sink : Sink instance

        Returns
        -------
        res : Sink
            The sink with processed data.

        """
        for fid, feature in self.featureset.items():
            feature.on_start(
                source,
                self.featureset,
                sink)

        if sink is None:
            return self._extract(source)
        else:
            for result in self._extract(source):
                sink.receive_append(self._pop_hidden(result))

        for fid, feature in self.featureset.items():
            feature.on_finished(
                source,
                self.featureset,
                sink)

        sink.receive({
            'hiddenfeatures':
                self.get_features_parameters_and_metadata(hidden=True),
            'features':
                self.get_features_parameters_and_metadata(hidden=False),
            'source':
                self.get_parameters_and_metadata(source)
            })
        return sink

    def reset(self):
        """Resets the states of features.
        If a new source shall be processed this may be usefull/needed.
        """
        self.featureset = features_to_featureset(self.features, new=True)

    @staticmethod
    def get_parameters_and_metadata(obj):
        """Returns dict with parameters and metadata from given ``obj``."""
        return dict(
            parameters=dict(obj.parameters),
            metadata=dict(obj.metadata))

    def get_features_parameters_and_metadata(self, hidden=False):
        """Returns dict with parameters and metadata from self.featureset."""
        return {v.name: self.get_parameters_and_metadata(
            v) for k, v in self.featureset.items() if v.hidden == hidden}

    def _pop_hidden(self, results):
        """Returns resluts without hidden feature results."""
        for fid, feature in self.featureset.items():
            if feature.hidden:
                results.pop(feature.name)
        return results
