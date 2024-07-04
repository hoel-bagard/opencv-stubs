import builtins
from typing import Any, overload, TypeAlias

from . import draw, gst

retval: TypeAlias = Any

class GOutputs(builtins.object):
    def getGArray(self, type) -> retval:
        """"""

    def getGMat(self) -> retval:
        """"""

    def getGOpaque(self, type) -> retval:
        """"""

    def getGScalar(self) -> retval:
        """"""

class GStreamerPipeline(builtins.object): ...
class IStreamSource(builtins.object): ...

def get_streaming_source(pipeline, appsinkName, outputType=...) -> retval:
    """
    .
    """

@overload
def make_capture_src(path) -> retval:
    """
    * @brief OpenCV's VideoCapture-based streaming source.

    This class implements IStreamSource interface.
    Its constructor takes the same parameters as cv::VideoCapture does.

    Please make sure that videoio OpenCV module is available before using
    this in your application (G-API doesn't depend on it directly).

    @note stream sources are passed to G-API via shared pointers, so
     please gapi::make_src<> to create objects and ptr() to pass a
     GCaptureSource to cv::gin().
    """

@overload
def make_capture_src(path) -> retval:
    """
    .
    """

def make_gst_src(pipeline, outputType=...) -> retval:
    """
    .
    """
