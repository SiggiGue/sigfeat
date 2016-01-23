"""Mixin for Metadata.

Adds methods `add_metadata()`, `extend_metadata()` and
`fetch_metadata_as_attrs()` to the class.

Metadata will be added to sinks. So feature data sets
contain metadata for e.g. features and sources.

"""


class AbstractMetadataMixin:
    _metadata = None

    def _init_metadata_list(self):
        if self._metadata is None:
            self._metadata = [('classname', self.__class__.__name__)]

    def add_metadata(self, name, value):
        """Appends key value pair to metadata list."""
        self._init_metadata_list()
        self._metadata.append((name, value))

    def extend_metadata(self, mdata):
        """Extends the metadata with the given list of key value pairs."""
        self._init_metadata_list()
        for name, value in mdata:
            self._metadata += [(name, value)]

    def fetch_metadata_as_attrs(self):
        """Sets metadata as attributes of self."""
        self._init_metadata_list()
        for name, value in self.metadata:
            setattr(self, name, value)

    @property
    def metadata(self):
        """Returns metadata."""
        return self._metadata

    # def init_metadata(self):
    #     self._metadata = list(self._gen_metadata())
    #
    # def _gen_metadata(self):
    #     for name in dir(self):
    #         obj = getattr(self, name)
    #         if isinstance(obj, Metadata):
    #             yield obj.name, obj.value


# class Metadata(object):
#     __slots__ = ('name', 'value')
#
#     def __init__(self, name, value):
#         self.name = name
#         self.value = value
