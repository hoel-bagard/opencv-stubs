from typing import Any, TypeAlias

from .. import functions as cv2

_binaryMap: TypeAlias = Any
saliencyMap: TypeAlias = Any

retval: TypeAlias = Any

class MotionSaliency(Saliency): ...

class MotionSaliencyBinWangApr2014(MotionSaliency):
    def computeSaliency(self, image, saliencyMap=...) -> tuple[retval, saliencyMap]:
        """"""

    def getImageHeight(self) -> retval:
        """"""

    def getImageWidth(self) -> retval:
        """"""

    def init(self) -> retval:
        """
        @brief This function allows the correct initialization of all data structures that will be used by the
        algorithm.
        """

    def setImageHeight(self, val) -> None:
        """"""

    def setImageWidth(self, val) -> None:
        """"""

    def setImagesize(self, W, H) -> None:
        """
        @brief This is a utility function that allows to set the correct size (taken from the input image) in the
        corresponding variables that will be used to size the data structures of the algorithm.
        @param W width of input image
        @param H height of input image
        """

    def create(self) -> retval:
        """"""

class Objectness(Saliency): ...

class ObjectnessBING(Objectness):
    def computeSaliency(self, image, saliencyMap=...) -> tuple[retval, saliencyMap]:
        """"""

    def getBase(self) -> retval:
        """"""

    def getNSS(self) -> retval:
        """"""

    def getW(self) -> retval:
        """"""

    def getobjectnessValues(self) -> retval:
        """
        @brief Return the list of the rectangles' objectness value,

        in the same order as the *vector\<Vec4i\> objectnessBoundingBox* returned by the algorithm (in
        computeSaliencyImpl function). The bigger value these scores are, it is more likely to be an
        object window.
        """

    def read(self) -> None:
        """"""

    def setBBResDir(self, resultsDir) -> None:
        """
        @brief This is a utility function that allows to set an arbitrary path in which the algorithm will save the
        optional results

        (ie writing on file the total number and the list of rectangles returned by objectess, one for
        each row).
        @param resultsDir results' folder path
        """

    def setBase(self, val) -> None:
        """"""

    def setNSS(self, val) -> None:
        """"""

    def setTrainingPath(self, trainingPath) -> None:
        """
        @brief This is a utility function that allows to set the correct path from which the algorithm will load
        the trained model.
        @param trainingPath trained model path
        """

    def setW(self, val) -> None:
        """"""

    def write(self) -> None:
        """"""

    def create(self) -> retval:
        """"""

class Saliency(cv2.Algorithm):
    def computeSaliency(self, image, saliencyMap=...) -> tuple[retval, saliencyMap]:
        """
        * \brief Compute the saliency
        * \param image        The image.
        * \param saliencyMap      The computed saliency map.
        * \return true if the saliency map is computed, false otherwise
        """

class StaticSaliency(Saliency):
    def computeBinaryMap(self, _saliencyMap, _binaryMap=...) -> tuple[retval, _binaryMap]:
        """
        @brief This function perform a binary map of given saliency map. This is obtained in this
        way:

        In a first step, to improve the definition of interest areas and facilitate identification of
        targets, a segmentation by clustering is performed, using *K-means algorithm*. Then, to gain a
        binary representation of clustered saliency map, since values of the map can vary according to
        the characteristics of frame under analysis, it is not convenient to use a fixed threshold. So,
        *Otsu's algorithm* is used, which assumes that the image to be thresholded contains two classes
        of pixels or bi-modal histograms (e.g. foreground and back-ground pixels); later on, the
        algorithm calculates the optimal threshold separating those two classes, so that their
        intra-class variance is minimal.

        @param _saliencyMap the saliency map obtained through one of the specialized algorithms
        @param _binaryMap the binary map
        """

class StaticSaliencyFineGrained(StaticSaliency):
    def computeSaliency(self, image, saliencyMap=...) -> tuple[retval, saliencyMap]:
        """"""

    def create(self) -> retval:
        """"""

class StaticSaliencySpectralResidual(StaticSaliency):
    def computeSaliency(self, image, saliencyMap=...) -> tuple[retval, saliencyMap]:
        """"""

    def getImageHeight(self) -> retval:
        """"""

    def getImageWidth(self) -> retval:
        """"""

    def read(self, fn) -> None:
        """"""

    def setImageHeight(self, val) -> None:
        """"""

    def setImageWidth(self, val) -> None:
        """"""

    def create(self) -> retval:
        """"""

def MotionSaliencyBinWangApr2014_create() -> retval:
    """
    .
    """

def ObjectnessBING_create() -> retval:
    """
    .
    """

def StaticSaliencyFineGrained_create() -> retval:
    """
    .
    """

def StaticSaliencySpectralResidual_create() -> retval:
    """
    .
    """
