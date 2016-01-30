"""Parameter classes."""


class Parameter(object):
    """Adds a Parameter to the class."""

    def __init__(self, default=None):
        self._default = default

    def validate(self, value):
        """You can override this method for validation of parameter values."""
        return value

    @property
    def default(self):
        """Returns the default value."""
        return self._default

    @default.setter
    def default(self, value):
        """Sets the default value, validates value first."""
        self._default = self._validate(value)


class ParameterMixin:
    """ParameterMixin class
    Adds Parameter functionality to classes.

    """

    def init_parameters(self, params):
        """This method must be called to collect all parameters and provide
        them as attributes.

        By calling this function defined parameters will become attributes
        with values not beeing of type Parameter anymore.
        But all parameters will be placed in self._params as well.
        """
        self._params = tuple(self._gen_param_values(params))
        for pname, pval in self._params:
            self._set_param(pname, pval)

    @property
    def params(self):
        """Returns all parameters."""
        return self._params

    def _set_param(self, name, value):
        """Sets key value pair as attribut of self."""
        return setattr(self, name, value)

    @classmethod
    def _gen_params(cls):
        """Yields parameters form cls."""
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
