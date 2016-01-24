class Extractor(object):
    def __init__(self,
                 featureset=None,
                 preprocess=None):
        self.featureset = featureset
        self.preprocess = preprocess

    @staticmethod
    def _get_featuremd(feat):
        return dict(
            params=dict(feat.params),
            metadata=dict(feat.metadata))

    @staticmethod
    def _get_sourcemd(src):
        return dict(
            metadata=dict(src.metadata),
            parameters=dict(src.params))

    def extract(self, source, sink):
        results = {}
        sink.receive({
            'features': {
                v.name: self._get_featuremd(
                    v) for k, v in self.featureset.items()},
            'source': self._get_sourcemd(source)
        })
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

            sink.receive_append(self._only_write_to_sink(results))
        return sink

    def _only_write_to_sink(self, results):
        for feature in self.featureset.items():
            if not feature.write_to_sink:
                results.pop(feature.name)
        return results

    def reset(self):
        self.featureset.reset_features()
        self.preprocess = self.preprocess.new()
