import abc
import six
from collections import defaultdict
import h5py
from datetime import datetime
import yaml


@six.add_metaclass(abc.ABCMeta)
class Sink(object):
    @abc.abstractclassmethod
    def receive(self, datad):
        """Shall receive dictionaries directly written to source."""
        pass

    @abc.abstractclassmethod
    def receive_append(self, resultd):
        """Shall receive result dictionaries appending data to the fields."""
        pass


class DefaultDictSink(Sink, defaultdict):
    def __init__(self):
        defaultdict.__init__(self, list)

    def receive(self, datad):
        self.update(datad)

    def receive_append(self, resultd):
        for name, res in resultd.items():
            self[name] += [res]


def _dump_dict_to_hdf(d, hdf):
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
                        print(all(isinstance(i, datetime) for i in v))
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


class Hdf5Sink(Sink, h5py.File):
    def __init__(self, *args, **kwargs):
        h5py.File.__init__(self, *args, **kwargs)
        self._pos = 0
        self._chunksize = 1000
        self._columns = {}

    def receive(self, datad):
        _dump_dict_to_hdf(datad, self)

    def receive_append(self, resultd):
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
            ds[self._pos, :] = res
            self._pos += 1
