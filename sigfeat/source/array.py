from numpy import asarray, product

from ..base import Source


class ArraySource(Source):
    """Source class for iterable arrays.

    Parameters
    ----------
    array : ndarray
        Expects an iterable array with .shape tuple.
    samplerate : int
    name : str
    blocksize : int
    overlap : int

    """

    def __init__(self, array, samplerate, name='', **parameters):
        array = asarray(array)
        self.unroll_parameters(parameters)
        self._array = array
        self.channels = product(array.shape[1:])
        self.add_metadata('name', name)
        self.add_metadata('arraylen', len(array))
        self.add_metadata('channels', self.channels)
        self.add_metadata('samplerate', samplerate)
        self.fetch_metadata_as_attrs()

    def generate(self):
        """Returns generator that yields blocks out of the array."""
        indexrange = range(
            0,
            len(self._array)-self.blocksize+1,
            self.blocksize-self.overlap)
        for index in indexrange:
            yield self._array[index:index+self.blocksize], index
