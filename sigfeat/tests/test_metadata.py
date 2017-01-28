import pytest
from sigfeat.base.metadata import MetadataMixin


class MetadataMixinSubclass(MetadataMixin):

    def __init__(self):
        self.fetch_metadata_as_attrs()


def test_add_metadata():
    s = MetadataMixinSubclass()
    s.add_metadata('bum', 1)
    s.add_metadata('eins', 'Bum')
    assert s.metadata[-1] == ('eins', 'Bum')


def test_extend_metadata():
    s = MetadataMixinSubclass()
    s.extend_metadata((
        ('a', 'b'),
        ('c', 'd')
    ))
    assert len(s.metadata) == 3


def test_fetch_metadata_as_attributes():
    s = MetadataMixinSubclass()
    s.add_metadata('test', 'test')
    s.fetch_metadata_as_attrs()
    assert s.test == 'test'


if __name__ == '__main__':
    pytest.main()
