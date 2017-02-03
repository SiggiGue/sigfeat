import pytest
from sigfeat.base.result import Result


def test_result_creation():
    res = Result()
    assert isinstance(res, Result)


def test_result_setitem():
    r = Result()
    with pytest.raises(TypeError):
        r['test'] = 'Test'

    r._setitem('test', 'Test')
    assert r['test'] == 'Test'


if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
