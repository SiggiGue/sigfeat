"""Mixin for Metadata.

Adds methods `add_metadata()`, `extend_metadata()` and
`fetch_metadata_as_attrs()` to the class.

Metadata will be extracted into Sink. So created feature datasets
contain metadata for e.g. features and sources.

"""


class MetadataMixin:
    """MetadataMixin class
    Adding metadata functionality to classes.
    Overrides:

    ``self._metadata``
    ``self.metadata``
    ``_init_metadata_list``
    ``add_metadata``
    ``extend_metadata``
    ``fetch_metadata_as_attrs``

    """
    _metadata = None

    def _init_metadata_list(self):
        if self._metadata is None:
            self._metadata = [('classname', self.__class__.__name__)]

    def add_metadata(self, name, value):
        """Appends the key value pair to metadata list."""
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
        self._init_metadata_list()
        return self._metadata
