from typing import Any, TypeAlias
retval: TypeAlias = Any

class Index(builtins.object):
    def build(self, features, params, distType = ...) -> None:
        """"""

    def getAlgorithm(self) -> retval:
        """"""

    def getDistance(self) -> retval:
        """"""

    def knnSearch(self, query, knn, indices = ..., dists = ..., params = ...) -> tuple[indices, dists]:
        """"""

    def load(self, features, filename) -> retval:
        """"""

    def radiusSearch(self, query, radius, maxResults, indices = ..., dists = ..., params = ...) -> tuple[retval, indices, dists]:
        """"""

    def release(self) -> None:
        """"""

    def save(self, filename) -> None:
        """"""


FLANN_INDEX_TYPE_16S: int
FLANN_INDEX_TYPE_16U: int
FLANN_INDEX_TYPE_32F: int
FLANN_INDEX_TYPE_32S: int
FLANN_INDEX_TYPE_64F: int
FLANN_INDEX_TYPE_8S: int
FLANN_INDEX_TYPE_8U: int
FLANN_INDEX_TYPE_ALGORITHM: int
FLANN_INDEX_TYPE_BOOL: int
FLANN_INDEX_TYPE_STRING: int
LAST_VALUE_FLANN_INDEX_TYPE: int