import builtins
from typing import Any, TypeAlias

retval: TypeAlias = Any

class ExportClassName(builtins.object):
    def getFloatParam(self) -> retval:
        """"""

    def getIntParam(self) -> retval:
        """"""

    def create(self, params=...) -> retval:
        """"""

    def originalName(self) -> retval:
        """"""

    class Params(builtins.object):
        float_value: float = 3.5
        int_value: int = 123

        def __init__(self, /, *args, **kwargs):
            """Initialize self.  See help(type(self)) for accurate signature."""

        def __repr__(self, /):
            """Return repr(self)."""

def ExportClassName_create(params=...) -> retval:
    """
    .
    """

def ExportClassName_originalName() -> retval:
    """
    .
    """

def OriginalClassName_create(params=...) -> retval:
    """
    .
    """

def OriginalClassName_originalName() -> retval:
    """
    .
    """

def testEchoBooleanFunction(flag) -> retval:
    """
    .
    """
