import pytest
from sigfeat.feature import Feature
from sigfeat.parameter import Parameter


class A(Feature):
    param = Parameter()

    def process(self, *args, **kwargs):
        pass


class B(Feature):
    param = Parameter()

    def requires(self):
        yield A(name='a', param=1)

    def process(self, *args, **kwargs):
        pass


class C(Feature):
    param = Parameter()

    def requires(self):
        yield B(name='b1', param=1)
        yield A(name='a', param=1)

    def process(self, *args, **kwargs):
        pass


class D(Feature):
    def requires(self):
        yield C(param=1)
        yield B
        yield A(name='SpecialA', param=2)

    def process(self, *args, **kwargs):
        pass


def test_featureset():
    """Testing featureset method for instances and fid."""
    a1 = A(param=1)
    b1 = B(name='b')
    features = list(b1.featureset().items())
    assert len(features) == 2
    assert features[0][0] != a1.fid
    assert features[0][0] != a1
    a1 = A(name='a', param=1)
    assert features[0][0] == a1.name
    assert features[1][0] == b1.name
    assert features[1][1] == b1


def test_featureset_special():
    d = D(name='d')
    assert len(d.featureset()) == 5


def test_duplicate_name_error():
    with pytest.raises(ValueError):
        D(name='SpecialA')


def test_missing_feature_instance():
    class F(Feature):
        def requires(self):
            yield A
            yield B

        def process(self, *args):
            pass

    with pytest.raises(ValueError):
        F().featureset()

    f = F()
    f.featureset(autoinst=True, err_missing=False)


def test_fid():
    a = A(name='test_name', param=99)
    assert a.fid == (a.__class__.__name__, str(a.parameters))


def test_hide_and_hidden():
    d = D()
    assert not d.hidden
    d.hide(True)
    assert d.hidden
    d.hide(False)
    assert not d.hidden
    assert d.hide(True).hidden


if __name__ == '__main__':
    pytest.main()
