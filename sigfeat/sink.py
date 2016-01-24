from collections import defaultdict


class DefaultDictSink(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, list)

    def receive(self, datad):
        self.update(datad)

    def receive_append(self, resultd):
        for name, res in resultd.items():
            self[name] += [res]
