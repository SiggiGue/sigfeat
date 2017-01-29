from soundfile import SoundFile

from ..base import Source
from ..base import Parameter


class SoundFileSource(Source):
    """Source generating data from SoundFiles.

    The parameters are for the :meth:`SoundFile.blocks` method
    used for the blocks method of this Source class.

    Parameters
    ----------
    sf : SoundFile instance or str
    blocksize : int
    overlap : int
    frames : int
        The number of frames to yield from soundfile.
        If frames < 1, the file is read until the end.
    fill_value : scalar
        The value last block will filled up, if it is shorter tha blocksize.
    dtype : {'float64', 'float32', 'int32', 'int16'}, optional
        See :meth:`soundfile.SoundFile.read`.
    always_2d : bool
        Indicates wether all blocks are at least 2d numpy arrays.

    """
    frames = Parameter(default=-1)
    fill_value = Parameter(default=0)
    dtype = Parameter(default='float64')
    always_2d = Parameter(default=False)

    def __init__(self, sf=None, **parameters):
        self.unroll_parameters(parameters)
        if isinstance(sf, str):
            sf = SoundFile(sf)

        self.sf = sf
        metadata = list(self._gen_metadata_from_sf(sf))
        self.extend_metadata(metadata)
        self.fetch_metadata_as_attrs()

    @staticmethod
    def _gen_metadata_from_sf(sf):
        ATTRS = ['name', 'mode', 'samplerate', 'channels', 'format',
                 'subtype', 'endian']
        for attr in ATTRS:
            yield attr, getattr(sf, attr)
        yield 'length', len(sf)

    def generate(self):
        """Returns generator that yields blocks from the SoundFile."""
        blocks = self.sf.blocks(
            blocksize=self.blocksize,
            overlap=self.overlap,
            frames=self.frames,
            dtype=self.dtype,
            fill_value=self.fill_value,
            always_2d=self.always_2d)
        blockshift = self.blocksize - self.overlap
        index = self.sf.tell()
        for block in blocks:
            yield block, index
            index += blockshift
        self.sf.close()
