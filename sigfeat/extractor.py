class Extractor(object):
    def __init__(self, source, sink, featureset):
        self._source = source
        self._sink = sink
        self._featureset = featureset

    def run(self):
        for block in self._source.blocks():
            for fid, feature in self._featureset.items():
                feature.process(block, self._source, self._sink)


# TODO need preprocess of blocks?
