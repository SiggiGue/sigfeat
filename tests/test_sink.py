import pytest
from sigfeat.sink import Sink
from sigfeat.sink import DefaultDictSink


class MySink(Sink):
    def receive(self, data):
        return data

    def receive_append(self, data):
        return data


def test_my_sink():
    s = MySink()
    assert s.receive(1) == 1
    assert s.receive_append(2) == 2


def test_default_dict_sink():
    dds = DefaultDictSink()
    dds.receive({'data': 1})
    dds.receive_append({'test': 'result'})
    dds.receive_append({'test': '1'})
    assert dds['data'] == 1
    assert dds['results']['test'] == ['result', '1']


if __name__ == '__main__':
    pytest.main()  # pragma: no coverage
