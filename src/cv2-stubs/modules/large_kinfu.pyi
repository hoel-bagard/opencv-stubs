import builtins
from typing import Any, overload, TypeAlias

normals: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any

retval: TypeAlias = Any

class LargeKinfu(builtins.object):
    def getCloud(self, points=..., normals=...) -> tuple[points, normals]:
        """"""

    def getNormals(self, points, normals=...) -> normals:
        """"""

    def getPoints(self, points=...) -> points:
        """"""

    @overload
    def render(self, image=...) -> image:
        """"""

    @overload
    def render(self, cameraPose, image=...) -> image:
        """"""

    def reset(self) -> None:
        """"""

    def update(self, depth) -> retval:
        """"""

    def create(self, _params) -> retval:
        """"""

class Params(builtins.object):
    def coarseParams(self) -> retval:
        """
        @brief Coarse parameters
        A set of parameters which provides better speed, can fail to match frames
        in case of rapid sensor motion.
        """

    def defaultParams(self) -> retval:
        """
        @brief Default parameters
        A set of parameters which provides better model quality, can be very slow.
        """

    def hashTSDFParams(self, isCoarse) -> retval:
        """
        @brief HashTSDF parameters
        A set of parameters suitable for use with HashTSDFVolume
        """

def LargeKinfu_create(_params) -> retval:
    """
    .
    """

def Params_coarseParams() -> retval:
    """
    @brief Coarse parameters
            A set of parameters which provides better speed, can fail to match frames
            in case of rapid sensor motion.
    """

def Params_defaultParams() -> retval:
    """
    @brief Default parameters
            A set of parameters which provides better model quality, can be very slow.
    """

def Params_hashTSDFParams(isCoarse) -> retval:
    """
    @brief HashTSDF parameters
            A set of parameters suitable for use with HashTSDFVolume
    """
