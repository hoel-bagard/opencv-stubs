import builtins
from typing import Any, Final, TypeAlias

indices: TypeAlias = Any
dists: TypeAlias = Any

retval: TypeAlias = Any

class Index(builtins.object):
    def build(self, features, params, distType=...) -> None:
        """"""

    def getAlgorithm(self) -> retval:
        """"""

    def getDistance(self) -> retval:
        """"""

    def knnSearch(self, query, knn, indices=..., dists=..., params=...) -> tuple[indices, dists]:
        """"""

    def load(self, features, filename) -> retval:
        """"""

    def radiusSearch(self, query, radius, maxResults, indices=..., dists=..., params=...) -> tuple[retval, indices, dists]:
        """"""

    def release(self) -> None:
        """"""

    def save(self, filename) -> None:
        """"""

FLANN_INDEX_TYPE_16S: Final[int]
FLANN_INDEX_TYPE_16U: Final[int]
FLANN_INDEX_TYPE_32F: Final[int]
FLANN_INDEX_TYPE_32S: Final[int]
FLANN_INDEX_TYPE_64F: Final[int]
FLANN_INDEX_TYPE_8S: Final[int]
FLANN_INDEX_TYPE_8U: Final[int]
FLANN_INDEX_TYPE_ALGORITHM: Final[int]
FLANN_INDEX_TYPE_BOOL: Final[int]
FLANN_INDEX_TYPE_STRING: Final[int]
LAST_VALUE_FLANN_INDEX_TYPE: Final[int]
