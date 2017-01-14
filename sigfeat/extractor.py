"""This Module contains the central Extractor class.

The Extractor consumes a FeatureSet and a Preprocess instance
and is used to extract the Features from given sources into
Sinks.

"""

from .result import Result
from .feature import features_to_featureset


class Extractor(object):
    """Feature Extractor

    Parameters
    ----------
    features : FeatureSet/itearable of Features
    preprocess : Preprocess instance, optional

    """
    def __init__(self, *features, preprocess=None):
        # if not isinstance(featureset, FeatureSet):
        #     featureset = FeatureSet(*featureset)
        self.features = features
        self.featureset = features_to_featureset(self.features)
        self.preprocess = preprocess

    def _extract(self, source):
        result = Result()
        for data in source:
            if self.preprocess:
                data = self.preprocess(data)
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
        if sink is not None:
            sink.receive({
                'features': self.get_features_parameters_and_metadata(),
                'source': self.get_parameters_and_metadata(source)
            })

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
        return sink

    def reset(self):
        """Resets the states of features and preprocess.
        If a new source shall be processed this may be usefull/needed.
        """
        self.featureset = features_to_featureset(self.features, new=True)
        self.preprocess = self.preprocess.new()

    @staticmethod
    def get_parameters_and_metadata(obj):
        """Returns dict with parameters and metadata from given ``obj``."""
        return dict(
            parameters=dict(obj.parameters),
            metadata=dict(obj.metadata))

    def get_features_parameters_and_metadata(self):
        """Returns dict with parameters and metadata from self.featureset."""
        return {v.name: self.get_parameters_and_metadata(
            v) for k, v in self.featureset.items() if not v.hidden}

    def _pop_hidden(self, results):
        """Returns resluts without hidden feature results."""
        for fid, feature in self.featureset.items():
            if feature.hidden:
                results.pop(feature.name)
        return results
