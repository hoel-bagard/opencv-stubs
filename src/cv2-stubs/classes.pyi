from abc import ABC, abstractmethod
from typing import Any, TypeAlias, overload

import numpy as np
import numpy.typing as npt

from typing_extensions import Self

InputArrayOfArrays: TypeAlias = npt.NDArray[Any]

class FlannBasedMatcher:
    def __new__(cls) -> Any:
        ...

    def __repr__(self) -> Any:
        ...

    def __init__(self) -> None:
        ...

    def create(self) -> Any:
        ...

    def __doc__(self) -> Any:
        ...

    def __module__(self) -> Any:
        ...

    def add(self, descriptors: list[Any]| list[list[Any]]):
        ...

    def clear(self) -> Any:
        ...

    def clone(self) -> Any:
        ...

    def empty(self) -> Any:
        ...

    def getTrainDescriptors(self) -> Any:
        ...

    def isMaskSupported(self) -> Any:
        ...

    def knnMatch(self) -> Any:
        ...

    def match(self) -> Any:
        ...

    def radiusMatch(self) -> Any:
        ...

    def read(self) -> Any:
        ...

    def train(self) -> Any:
        ...

    def write(self) -> Any:
        ...

    def getDefaultName(self) -> Any:
        ...

    def save(self) -> Any:
        ...

    def __hash__(self) -> Any:
        ...

    def __str__(self) -> Any:
        ...

    def __getattribute__(self) -> Any:
        ...

    def __setattr__(self) -> Any:
        ...

    def __delattr__(self) -> Any:
        ...

    def __lt__(self) -> Any:
        ...

    def __le__(self) -> Any:
        ...

    def __eq__(self) -> Any:
        ...

    def __ne__(self) -> Any:
        ...

    def __gt__(self) -> Any:
        ...

    def __ge__(self) -> Any:
        ...

    def __reduce_ex__(self) -> Any:
        ...

    def __reduce__(self) -> Any:
        ...

    def __subclasshook__(self) -> Any:
        ...

    def __init_subclass__(cls) -> Any:
        ...

    def __format__(self) -> Any:
        ...

    def __sizeof__(self) -> Any:
        ...

    def __dir__(self) -> Any:
        ...

    def __class__(self) -> Any:
        ...


class FileNode:
    # https://docs.opencv.org/4.x/de/dd9/classcv_1_1FileNode.html#a062a89a78d3cab1e7949b3d0fd3d60ee
    ...

class FileStorage:
    # https://docs.opencv.org/4.x/da/d56/classcv_1_1FileStorage.html
    def endWriteStruct(self) -> None:
        ...

    def getFirstTopLevelNode(self) -> FileNode:
        ...

    def getFormat(self) -> int:
        ...

    def getNode(self) -> Any:
        ...

    def isOpened(self) -> bool:
        ...

    def open(self, filename: str, flags: int, encoding: str = ...) -> bool:
        ...

    def release(self) -> None:
        ...

    def releaseAndGetString(self) -> str:
        ...

    def root(self, streamidx: int = ...) -> FileNode:
        ...

    def startWriteStruct(self, name: str, flags: int, typeName: str = ...) -> None:
        ...

    def write(self, name: str, val: float) -> None:
        ...

    def writeComment(self, comment: str, append: bool = ...) -> None:
        ...



class Algorithm(ABC):
    @abstractmethod
    def clear(self) -> None:
        ...

    @abstractmethod
    def empty(self) -> None:
        ...

    @abstractmethod
    def getDefaultName(self) -> str:
         ...

    def read(self, fn: FileNode) -> None:
         ...

    @abstractmethod
    def save(self, filename: str) -> None:
         ...

    @overload
    def write(self, fs: FileStorage) -> None:
        ...

    @overload
    def write(self, fs: FileStorage, name: str) -> None:
        ...

class DescriptorMatcher(Algorithm, ABC):
     def add(self, descriptors: InputArrayOfArrays) -> Any:
        ...

     def clone(self) -> Any:
        ...

     def create(self) -> Any:
        ...

     def getTrainDescriptors(self) -> Any:
        ...

     def isMaskSupported(self) -> Any:
        ...

     def knnMatch(self) -> Any:
        ...

     def match(self) -> Any:
        ...

     def radiusMatch(self) -> Any:
        ...

     def train(self) -> Any:
        ...

     def write(self) -> Any:
        ...


class Feature2D(Algorithm, ABC):
    ...


class FastFeatureDetector(Feature2D):
    ...
