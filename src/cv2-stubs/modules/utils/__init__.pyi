from . import nested
from . import fs
from . import cv2

class ClassWithKeywordProperties(builtins.object):
    ...


class NativeMethodPatchedResult(builtins.tuple):
    ...


def dumpBool(argument) -> retval:
    """
        .
    """

def dumpCString(argument) -> retval:
    """
        .
    """

def dumpDouble(argument) -> retval:
    """
        .
    """

def dumpFloat(argument) -> retval:
    """
        .
    """

def dumpInputArray(argument) -> retval:
    """
        .
    """

def dumpInputArrayOfArrays(argument) -> retval:
    """
        .
    """

def dumpInputOutputArray(argument) -> tuple[retval, argument]:
    """
        .
    """

def dumpInputOutputArrayOfArrays(argument) -> tuple[retval, argument]:
    """
        .
    """

def dumpInt(argument) -> retval:
    """
        .
    """

def dumpInt64(argument) -> retval:
    """
        .
    """

def dumpRange(argument) -> retval:
    """
        .
    """

def dumpRect(argument) -> retval:
    """
        .
    """

def dumpRotatedRect(argument) -> retval:
    """
        .
    """

def dumpSizeT(argument) -> retval:
    """
        .
    """

def dumpString(argument) -> retval:
    """
        .
    """

def dumpTermCriteria(argument) -> retval:
    """
        .
    """

def dumpVec2i(value = ...) -> retval:
    """
        .
    """

def dumpVectorOfDouble(vec) -> retval:
    """
        .
    """

def dumpVectorOfInt(vec) -> retval:
    """
        .
    """

def dumpVectorOfRect(vec) -> retval:
    """
        .
    """

def generateVectorOfInt(len) -> vec:
    """
        .
    """

def generateVectorOfMat(len, rows, cols, dtype, vec = ...) -> vec:
    """
        .
    """

def generateVectorOfRect(len) -> vec:
    """
        .
    """

@overload
def Returns a new subclass of tuple with named fields.:
    """

        >>> Point = namedtuple('Point', ['x', 'y'])
        >>> Point.__doc__                   # docstring for the new class
        'Point(x, y)'
        >>> p = Point(11, y=22)             # instantiate with positional args or keywords
        >>> p[0] + p[1]                     # indexable like a plain tuple
        33
        >>> x, y = p                        # unpack like a regular tuple
        >>> x, y
        (11, 22)
        >>> p.x + p.y                       # fields also accessible by name
        33
        >>> d = p._asdict()                 # convert to a dictionary
        >>> d['x']
        11
        >>> Point(**d)                      # convert from a dictionary
    """

@overload
def Returns a new subclass of tuple with named fields.:
    """
        >>> p._replace(x=100)               # _replace() is like str.replace() but targets named fields
    """

@overload
def Returns a new subclass of tuple with named fields.:
    """
    """

def testAsyncArray(argument) -> retval:
    """
        .
    """

def testAsyncException() -> retval:
    """
        .
    """

@overload
def testOverloadResolution(value, point = ...) -> retval:
    """
    """

@overload
def testOverloadResolution(value, point = ...) -> retval:
    """
        .
    """

def :
    """
    """

def testRaiseGeneralException() -> None:
    """
        .
    """

def testReservedKeywordConversion(positional_argument, lambda_ = ..., from_ = ...) -> retval:
    """
        .
    """

def testRotatedRect(x, y, w, h, angle) -> retval:
    """
        .
    """

def testRotatedRectVector(x, y, w, h, angle) -> retval:
    """
        .
    """
