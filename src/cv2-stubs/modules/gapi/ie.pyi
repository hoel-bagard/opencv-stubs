import builtins
from typing import Any, Final, overload, TypeAlias

retval: TypeAlias = Any

class PyParams(builtins.object):
    def cfgBatchSize(self, size) -> retval:
        """"""

    def cfgNumRequests(self, nireq) -> retval:
        """"""

    def constInput(self, layer_name, data, hint=...) -> retval:
        """"""

@overload
def params(tag, model, weights, device) -> retval:
    """ """

@overload
def params(tag, model, weights, device) -> retval:
    """
    .
    """

ASYNC: Final[int] = 1
Async: Final[int] = 1
SYNC: Final[int] = 0
Sync: Final[int] = 0
TRAIT_AS_IMAGE: Final[int] = 1
TRAIT_AS_TENSOR: Final[int] = 0
TraitAs_IMAGE: Final[int] = 1
TraitAs_TENSOR: Final[int] = 0
