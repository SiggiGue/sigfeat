class Parameter(object):
    def __init__(self, default=None):
        self._default = default

    def validate(self, value):
        return value

    @property
    def default(self):
        return self._default

    @default.setter
    def default(self, value):
        self._default = self._validate(value)


class AbstractParameterMixin:

    def init_parameters(self, params):
        self._params = tuple(self._gen_param_values(params))
        for pname, pval in self._params:
            print(pname, pval)
            self._set_param(pname, pval)

    @property
    def params(self):
        return self._params

    def _set_param(self, name, value):
        return setattr(self, name, value)

    @classmethod
    def _gen_params(cls):
        for name in dir(cls):
            obj = getattr(cls, name)
            if isinstance(obj, Parameter):
                yield name, obj

    @classmethod
    def _gen_param_values(cls, paramsd):
        for pname, pobj in cls._gen_params():
            if pname in paramsd:
                yield pname, pobj.validate(paramsd[pname])
            else:
                yield pname, pobj.default