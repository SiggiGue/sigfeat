import pytest
from sigfeat.feature import Feature
from sigfeat.feature import features_to_featureset
from sigfeat.parameter import Parameter


# Defining some Feature classes:

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
        # info to this special case class requirements:
        # yeah B is yielded as a Feture class but in C is an instance of B
        # is yielded. So if you give a Class in the requirements a instance
        # of this class will picked. If no instance is provided you will
        # get an error since the instance is missing or you can
        # set a autoins=True and err_missing=False to automatically create
        # required feature instances if needed.
        yield C(param=1)
        yield B
        yield A(name='SpecialA', param=2)

    def process(self, *args, **kwargs):
        pass


# Test Feature methods

def test_requirements():
    b = B()
    req = next(b.requires())
    assert isinstance(req, A)
    assert req.fid == A(name='a', param=1).fid
    assert req.name == A(name='a', param=1).name
    assert len(b._requirements) == 0


def test_override_requirements():
    b = B(requirements=[A(param=2)])
    assert len(b._requirements) == 1
    deps = list(b.dependencies())
    assert deps[-1].param == 2


def test_on_start():
    assert A().on_start(1, 2, 3) is None


def test_process():
    with pytest.raises(TypeError):
        Feature()
        Feature.process(1, 2, 3)
    A().process(1, 2)


def test_on_finished():
    assert A().on_finished(1, 2, 3) is None


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


def test_featureset_new():
    b = B()
    feature = tuple(b.featureset(new=True).values())[-1]
    assert feature != b
    feature = tuple(b.featureset(new=False).values())[-1]
    assert feature == b


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


def test_features_to_featureset():
    fset = features_to_featureset(
            [A(), B(), D()]
        )
    assert len(fset) == 7  # since B depends not on A.name=='a'
    with pytest.raises(ValueError):
        fset = features_to_featureset(
            [D()]
        )

if __name__ == '__main__':
    pytest.main()
