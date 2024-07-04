from typing import Any, TypeAlias

from .. import functions as cv2

unwrappedPhaseMap: TypeAlias = Any
reliabilityMap: TypeAlias = Any

retval: TypeAlias = Any

class HistogramPhaseUnwrapping(PhaseUnwrapping):
    def getInverseReliabilityMap(self, reliabilityMap=...) -> reliabilityMap:
        """
        * @brief Get the reliability map computed from the wrapped phase map.

        * @param reliabilityMap Image where the reliability map is stored.
        """

    def create(self, parameters=...) -> retval:
        """
        * @brief Constructor

        * @param parameters HistogramPhaseUnwrapping parameters HistogramPhaseUnwrapping::Params: width,height of the phase map and histogram characteristics.
        """

class PhaseUnwrapping(cv2.Algorithm):
    def unwrapPhaseMap(self, wrappedPhaseMap, unwrappedPhaseMap=..., shadowMask=...) -> unwrappedPhaseMap:
        """
        * @brief Unwraps a 2D phase map.

        * @param wrappedPhaseMap The wrapped phase map of type CV_32FC1 that needs to be unwrapped.
        * @param unwrappedPhaseMap The unwrapped phase map.
        * @param shadowMask Optional CV_8UC1 mask image used when some pixels do not hold any phase information in the wrapped phase map.
        """

def HistogramPhaseUnwrapping_create(parameters=...) -> retval:
    """
    * @brief Constructor

         * @param parameters HistogramPhaseUnwrapping parameters HistogramPhaseUnwrapping::Params: width,height of the phase map and histogram characteristics.
    """
