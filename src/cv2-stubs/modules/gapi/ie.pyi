from typing import Any, TypeAlias

retval: TypeAlias = Any

class PyParams(builtins.object):
    def cfgBatchSize(self, size) -> retval:
        """"""

    def cfgNumRequests(self, nireq) -> retval:
        """"""

    def constInput(self, layer_name, data, hint = ...) -> retval:
        """"""


@overload
def params(tag, model, weights, device) -> retval:
    """
    """

@overload
def params(tag, model, weights, device) -> retval:
    """
        .
    """

ASYNC: int
Async: int
SYNC: int
Sync: int
TRAIT_AS_IMAGE: int
TRAIT_AS_TENSOR: int
TraitAs_IMAGE: int
TraitAs_TENSOR: int