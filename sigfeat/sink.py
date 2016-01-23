from collections import defaultdict


class DefaultDictSink(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, default=list)

    def receive(self, data):
        print(data)
