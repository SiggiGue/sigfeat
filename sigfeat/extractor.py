"""This Module contains the central Extractor class.

The Extractor consumes a FeatureSet and a Preprocess instance
and is used to extract the Features from given sources into
Sinks.

"""
from .feature import FeatureSet


class Extractor(object):
    """Feature Extractor

    Parameters
    ----------
    featureset : FeatureSet/itearable of Features
    preprocess : Preprocess instance, optional

    """
    def __init__(self,
                 featureset=None,
                 preprocess=None):
        if not isinstance(featureset, FeatureSet):
            featureset = FeatureSet(*featureset)

        self.featureset = featureset
        self.preprocess = preprocess

    def extract(self, source, sink):
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
        sink.receive({
            'features': self._get_features_pmd(),
            'source': self._get_pmd(source)
        })
        results = {}
        for block, index in source.blocks():
            if self.preprocess:
                block = self.preprocess(block)
            for fid, feature in self.featureset.items():
                results[feature.name] = feature.process(
                    block=block,
                    index=index,
                    results=results,
                    featureset=self.featureset,
                    sink=sink)

            sink.receive_append(self._pop_hidden(results))
        return sink

    def reset(self):
        """Resets the states of features and preprocess.
        If a new source shall be processed this may be usefull/needed.
        """
        self.featureset.reset_features()
        self.preprocess = self.preprocess.new()

    @staticmethod
    def _get_pmd(obj):
        """Returns dict with params and metadata from given ``obj``."""
        return dict(
            params=dict(obj.params),
            metadata=dict(obj.metadata))

    def _get_features_pmd(self):
        """Returns dict with params and metadata from self.featureset."""
        return {v.name: self._get_pmd(
            v) for k, v in self.featureset.items() if not v.hidden}

    def _pop_hidden(self, results):
        """Returns resluts without hidden feature results."""
        for fid, feature in self.featureset.items():
            if feature.hidden:
                results.pop(feature.name)
        return results
