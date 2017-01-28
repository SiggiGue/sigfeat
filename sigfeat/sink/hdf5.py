from sigfeat.sink import Sink
import h5py
from datetime import datetime
import yaml


class Hdf5Sink(Sink, h5py.File):  # pragma: no coverage
    """Sink writing data directly into a hdf5 file.

    WARNING: This implementation is quite bad until now!
    Writing to the file must be performed chunk wise because
    writing overhead is quite big. Deafaultdict is 10 times
    faster at the moment.
    A better way would be DefaultDictSink and writing this once into
    a hdf file.

    """
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        self._pos = 0
        self._chunksize = 10000
        self._columns = dict()

    def receive(self, datad):
        """Receives a dictionary and directly writes it into hdf5 file."""
        _dump_dict_to_hdf(datad, self)

    def receive_append(self, resultd):
        """Appends the given dictionary to datasets in h5 file.
        If the key does not exist, a new resizable dataset is created.
        Else data will be appended to the datasets and if needed datasets
        will be resized.

        """
        for name, res in resultd.items():
            if name not in self._columns:
                if hasattr(res, '__iter__'):
                    cols = len(res)
                else:
                    cols = 1
                self._columns[name] = cols

            if name not in self:
                self.create_dataset(
                    name,
                    shape=(self._chunksize, self._columns[name]),
                    maxshape=(None, self._columns[name]))

            ds = self[name]
            if ds.shape[0] <= self._pos:
                ds.resize((ds.shape[0]+self._chunksize, self._columns[name]))
            ds[self._pos, ...] = res
            self._pos += 1

    def tighten_length(self):
        for key, num in self._columns.items():
            self[key].resize((self._pos, num))


def _dump_dict_to_hdf(d, hdf):  # pragma: no coverage
    """Adds keys of given dict as groups and values as datasets
    to the given hdf-file (by string or object) or group object.

    Parameters
    ----------
    d : dict
        The dictionary containing only string keys and
        data values or dicts again.
    hdf : string (path to file) or `h5py.File()` or `h5py.Group()`

    Returns
    -------
    hdf : obj
        `h5py.Group()` or `h5py.File()` instance

    """

    def _recurse(d, h):
        for k, v in d.items():
            isdt = None
            if isinstance(v, dict):
                g = h.create_group(k)
                _recurse(v, g)
            else:
                if isinstance(v, datetime):
                    v = v.timestamp()
                    isdt = True

                if hasattr(v, '__iter__'):
                    if all(isinstance(i, datetime) for i in v):
                        # print(all(isinstance(i, datetime) for i in v))
                        v = [item.timestamp() for item in v]
                        isdt = True

                try:
                    ds = h.create_dataset(name=k, data=v)
                    if isdt:
                        ds.attrs.create(
                            name='__type__',
                            data="datetime")
                except TypeError:
                    # Obviously the data was not serializable. To give it
                    # a last try; serialize it to json
                    # and save it to the hdf file:
                    ds = h.create_dataset(name=k, data=yaml.safe_dump(v))
                    ds.attrs.create(
                        name='__type__',
                        data="yaml")
                    # pass
                    # if this fails again, restructure your data!
    return _recurse(d, hdf)
