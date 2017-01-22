import pytest
from sigfeat.parameter import Parameter
from sigfeat.parameter import ParameterMixin


class ParameterMixinSubclass(ParameterMixin):
    p1 = Parameter(default=0)
    p2 = Parameter('text')
    parameter_number_three = Parameter()


def test_parameterclass_defaults():
    pc = ParameterMixinSubclass()
    assert pc.parameters == (
        ('p1', 0),
        ('p2', 'text'),
        ('parameter_number_three', None))


def test_parameterclass_set_default():
    ParameterMixinSubclass.p1.default = 0


def test_parameterclass_overrided():
    pc = ParameterMixinSubclass()
    t = ParameterMixinSubclass(
        parameter_number_three='three',
        notyet_a_parameter='BOOM this will not be in parameters attribute!')
    assert t.parameters == (
        *pc.parameters[:-1],
        ('parameter_number_three', 'three'))


def test_get_class_parameters():
    params = ParameterMixinSubclass.get_class_parameters()
    assert len(params) == 3
    assert params == (
        ('p1', ParameterMixinSubclass.p1),
        ('p2', ParameterMixinSubclass.p2),
        ('parameter_number_three',
            ParameterMixinSubclass.parameter_number_three)
    )


def test_repr():
    res = ParameterMixinSubclass.p2.__repr__()
    assert isinstance(res, str)


if __name__ == '__main__':
    pytest.main()
