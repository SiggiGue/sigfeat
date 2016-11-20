"""This Module contains the central Extractor class.

The Extractor consumes a FeatureSet and a Preprocess instance
and is used to extract the Features from given sources into
Sinks.

"""


def features_to_featureset(features, new=False):
    """Returns an featureset of given features distinct in parameters."""
    featsets = (feat.featureset(new=new) for feat in features)
    featureset = next(featsets)
    for fset in featsets:
        featureset.update(fset)
    return featureset


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
        for fid, feature in self.featureset.items():
            feature.start(source=source, sink=sink)
        results = {}
        for block, index in source.blocks():
            if self.preprocess:
                block = self.preprocess(block)
            for fid, feature in self.featureset.items():
                results[feature.name] = feature.process(
                    index=index,
                    block=block,
                    results=results)

            sink.receive_append(self._pop_hidden(results))
        return sink

    def reset(self):
        """Resets the states of features and preprocess.
        If a new source shall be processed this may be usefull/needed.
        """
        self.featureset = features_to_featureset(new=True)
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
