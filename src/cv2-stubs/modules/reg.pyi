import builtins
from typing import Any, TypeAlias

linTr: TypeAlias = Any
img2: TypeAlias = Any
shift: TypeAlias = Any
projTr: TypeAlias = Any

retval: TypeAlias = Any

class Map(builtins.object):
    def compose(self, map) -> None:
        """"""

    def inverseMap(self) -> retval:
        """"""

    def inverseWarp(self, img1, img2=...) -> img2:
        """"""

    def scale(self, factor) -> None:
        """"""

    def warp(self, img1, img2=...) -> img2:
        """"""

class MapAffine(Map):
    def compose(self, map) -> None:
        """"""

    def getLinTr(self, linTr=...) -> linTr:
        """"""

    def getShift(self, shift=...) -> shift:
        """"""

    def inverseMap(self) -> retval:
        """"""

    def inverseWarp(self, img1, img2=...) -> img2:
        """"""

    def scale(self, factor) -> None:
        """"""

class MapProjec(Map):
    def compose(self, map) -> None:
        """"""

    def getProjTr(self, projTr=...) -> projTr:
        """"""

    def inverseMap(self) -> retval:
        """"""

    def inverseWarp(self, img1, img2=...) -> img2:
        """"""

    def normalize(self) -> None:
        """"""

    def scale(self, factor) -> None:
        """"""

class MapShift(Map):
    def compose(self, map) -> None:
        """"""

    def getShift(self, shift=...) -> shift:
        """"""

    def inverseMap(self) -> retval:
        """"""

    def inverseWarp(self, img1, img2=...) -> img2:
        """"""

    def scale(self, factor) -> None:
        """"""

class MapTypeCaster(builtins.object):
    def toAffine(self, sourceMap) -> retval:
        """"""

    def toProjec(self, sourceMap) -> retval:
        """"""

    def toShift(self, sourceMap) -> retval:
        """"""

class Mapper(builtins.object):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperGradAffine(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperGradEuclid(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperGradProj(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperGradShift(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperGradSimilar(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

class MapperPyramid(Mapper):
    def calculate(self, img1, img2, init=...) -> retval:
        """"""

    def getMap(self) -> retval:
        """"""

def MapTypeCaster_toAffine(sourceMap) -> retval:
    """
    .
    """

def MapTypeCaster_toProjec(sourceMap) -> retval:
    """
    .
    """

def MapTypeCaster_toShift(sourceMap) -> retval:
    """
    .
    """
