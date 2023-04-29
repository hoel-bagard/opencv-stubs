import builtins
from typing import Any, TypeAlias

retval: TypeAlias = Any

class Allocator(builtins.object):
    ...

def defaultAllocator() -> Allocator:
    """"""

def setDefaultAllocator(allocator) -> None:
    """"""

adjustROI: int
assignTo: int
channels: int
clone: int
col: int
colRange: int
convertTo: int
copyTo: int
create: int
cudaPtr: int
depth: int
download: int
elemSize: int
elemSize1: int
empty: int
isContinuous: int
locateROI: int
release: int
reshape: int
row: int
rowRange: int
setTo: int
size: int
step: int
step1: int
swap: int
type: int
updateContinuityFlag: int
upload: int
