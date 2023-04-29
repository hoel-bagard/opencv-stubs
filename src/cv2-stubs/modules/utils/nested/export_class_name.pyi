import builtins
from typing import Any, TypeAlias

retval: TypeAlias = Any

class Params(builtins.object):
    float_value: float = 3.5
    int_value: int = 123

    def __init__(self, /, *args, **kwargs):
        """Initialize self.  See help(type(self)) for accurate signature."""

    def __repr__(self, /):
        """Return repr(self)."""


