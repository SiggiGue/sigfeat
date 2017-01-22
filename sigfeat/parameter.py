"""This module implements the Parameter and its mixin class.

Purpose is, to distinguish parameters of objects from other attributes.
Parameters of instances will be extracted into Sink.

"""


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

    def __repr__(self):
        return "{}(default={})".format(
            self.__class__.__name__,
            str(self.default))


class ParameterMixin:
    """ParameterMixin class

    Adds Parameter functionality to classes.

    """
    _parameters = None

    def __init__(self, **parameters):
        self.unroll_parameters(parameters)

    def unroll_parameters(self, parameters):
        """This method must be called to collect all parameters and provide
        them as attributes.

        By calling this function defined parameters will become attributes
        with values not beeing of type Parameter anymore.
        But all parameters will be placed in self._parameters as well.

        """
        self._parameters = tuple(self._gen_param_values(parameters))
        for pname, pval in self._parameters:
            self._set_param(pname, pval)

    @property
    def parameters(self):
        """Returns all parameters."""
        if self._parameters:
            return self._parameters
        else:
            return tuple(self._gen_parameters())

    def _set_param(self, name, value):
        """Sets key value pair as attribut of self."""
        return setattr(self, name, value)

    @classmethod
    def get_class_parameters(cls):
        return tuple(cls._gen_parameters())

    @classmethod
    def _gen_parameters(cls):
        """Yields parameters form cls."""
        for name in dir(cls):
            obj = getattr(cls, name)
            if isinstance(obj, Parameter):
                yield name, obj

    @classmethod
    def _gen_param_values(cls, parametersd):
        for pname, pobj in cls._gen_parameters():
            if pname in parametersd:
                yield pname, pobj.validate(parametersd[pname])
            else:
                yield pname, pobj.default
