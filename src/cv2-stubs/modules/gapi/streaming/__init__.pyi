import builtins
from typing import Any, Final, overload, TypeAlias

retval: TypeAlias = Any

class queue_capacity(builtins.object): ...

def desync(g) -> retval:
    """
    .
    """

def seqNo(arg1) -> retval:
    """
    .
    """

def seq_id(arg1) -> retval:
    """
    .
    """

@overload
def size(src) -> retval:
    """
    @brief Gets dimensions from Mat.

    @note Function textual ID is "org.opencv.streaming.size"

    @param src Input tensor
    @return Size (tensor dimensions).
    """

@overload
def size(src) -> retval:
    """
    @overload
    Gets dimensions from rectangle.

    @note Function textual ID is "org.opencv.streaming.sizeR"

    @param r Input rectangle.
    @return Size (rectangle dimensions).
    """

def timestamp(arg1) -> retval:
    """
    .
    """

SYNC_POLICY_DONT_SYNC: Final[int]
SYNC_POLICY_DROP: Final[int]
sync_policy_dont_sync: Final[int]
sync_policy_drop: Final[int]
