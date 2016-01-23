# class Metadata(object):
#     __slots__ = ('name', 'value')
#
#     def __init__(self, name, value):
#         self.name = name
#         self.value = value


class AbstractMetadataMixin:
    _metadata = None

    def _init_metadata_list(self):
        if self._metadata is None:
            self._metadata = list()

    def add_metadata(self, name, value):
        self._init_metadata_list()
        self._metadata.append((name, value))

    def extend_metadata(self, mdata):
        self._init_metadata_list()
        for name, value in mdata:
            self._metadata += [(name, value)]

    def fetch_metadata_as_attrs(self):
        self._init_metadata_list()
        for name, value in self.metadata:
            setattr(self, name, value)

    @property
    def metadata(self):
        return self._metadata

    # def init_metadata(self):
    #     self._metadata = list(self._gen_metadata())
    #
    # def _gen_metadata(self):
    #     for name in dir(self):
    #         obj = getattr(self, name)
    #         if isinstance(obj, Metadata):
    #             yield obj.name, obj.value
