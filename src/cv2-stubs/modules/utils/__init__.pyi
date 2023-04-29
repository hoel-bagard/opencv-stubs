import builtins
from typing import Any, overload, TypeAlias

from . import fs, nested

argument: TypeAlias = Any
vec: TypeAlias = Any

retval: TypeAlias = Any

class ClassWithKeywordProperties(builtins.object):
    ...


class NativeMethodPatchedResult(builtins.tuple):
    ...


def dumpBool(argument) -> retval:
    ...

def dumpCString(argument) -> retval:
    ...

def dumpDouble(argument) -> retval:
    ...

def dumpFloat(argument) -> retval:
    ...

def dumpInputArray(argument) -> retval:
    ...

def dumpInputArrayOfArrays(argument) -> retval:
    ...

def dumpInputOutputArray(argument) -> tuple[retval, argument]:
    ...

def dumpInputOutputArrayOfArrays(argument) -> tuple[retval, argument]:
    ...

def dumpInt(argument) -> retval:
    ...

def dumpInt64(argument) -> retval:
    ...

def dumpRange(argument) -> retval:
    ...

def dumpRect(argument) -> retval:
    ...

def dumpRotatedRect(argument) -> retval:
    ...

def dumpSizeT(argument) -> retval:
    ...

def dumpString(argument) -> retval:
    ...

def dumpTermCriteria(argument) -> retval:
    ...

def dumpVec2i(value = ...) -> retval:
    ...

def dumpVectorOfDouble(vec) -> retval:
    ...

def dumpVectorOfInt(vec) -> retval:
    ...

def dumpVectorOfRect(vec) -> retval:
    ...

def generateVectorOfInt(len) -> vec:
    ...

def generateVectorOfMat(len, rows, cols, dtype, vec = ...) -> vec:
    ...

def generateVectorOfRect(len) -> vec:
    ...

def testAsyncArray(argument) -> retval:
    ...

def testAsyncException() -> retval:
    ...

@overload
def testOverloadResolution(value, point = ...) -> retval:
    ...

@overload
def testOverloadResolution(value, point = ...) -> retval:
    ...

def testRaiseGeneralException() -> None:
    ...

def testReservedKeywordConversion(positional_argument, lambda_ = ..., from_ = ...) -> retval:
    ...

def testRotatedRect(x, y, w, h, angle) -> retval:
    ...

def testRotatedRectVector(x, y, w, h, angle) -> retval:
    ...
