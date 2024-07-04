from typing import Any, TypeAlias

retval: TypeAlias = Any

def setParallelForBackend(backendName, propagateNumThreads=...) -> retval:
    """
    @brief Change OpenCV parallel_for backend

    @note This call is not thread-safe. Consider calling this function from the `main()` before any other OpenCV processing functions (and without any other created threads).
    """
