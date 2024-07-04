import builtins
from typing import Any, overload, TypeAlias

from .. import functions as cv2

matches: TypeAlias = Any
outImage: TypeAlias = Any
descriptors: TypeAlias = Any
keylines: TypeAlias = Any
keypoints: TypeAlias = Any
outImg: TypeAlias = Any

retval: TypeAlias = Any

class BinaryDescriptor(cv2.Algorithm):
    def compute(self, image, keylines, descriptors=..., returnFloatDescr=...) -> tuple[keylines, descriptors]:
        """
        @brief Requires descriptors computation

        @param image input image
        @param keylines vector containing lines for which descriptors must be computed
        @param descriptors
        @param returnFloatDescr flag (when set to true, original non-binary descriptors are returned)
        """

    def detect(self, image, mask=...) -> keypoints:
        """
        @brief Requires line detection

        @param image input image
        @param keypoints vector that will store extracted lines for one or more images
        @param mask mask matrix to detect only KeyLines of interest
        """

    def getNumOfOctaves(self) -> retval:
        """
        @brief Get current number of octaves
        """

    def getReductionRatio(self) -> retval:
        """
        @brief Get current reduction ratio (used in Gaussian pyramids)
        """

    def getWidthOfBand(self) -> retval:
        """
        @brief Get current width of bands
        """

    def setNumOfOctaves(self, octaves) -> None:
        """
        @brief Set number of octaves
        @param octaves number of octaves
        """

    def setReductionRatio(self, rRatio) -> None:
        """
        @brief Set reduction ratio (used in Gaussian pyramids)
        @param rRatio reduction ratio
        """

    def setWidthOfBand(self, width) -> None:
        """
        @brief Set width of bands
        @param width width of bands
        """

    def createBinaryDescriptor(self) -> retval:
        """
        @brief Create a BinaryDescriptor object with default parameters (or with the ones provided)
        and return a smart pointer to it
        """

class BinaryDescriptorMatcher(cv2.Algorithm):
    def knnMatch(self, queryDescriptors, trainDescriptors, k, mask=..., compactResult=...) -> matches:
        """
        @brief For every input query descriptor, retrieve the best *k* matching ones from a dataset provided from
        user or from the one internal to class

        @param queryDescriptors query descriptors
        @param trainDescriptors dataset of descriptors furnished by user
        @param matches vector to host retrieved matches
        @param k number of the closest descriptors to be returned for every input query
        @param mask mask to select which input descriptors must be matched to ones in dataset
        @param compactResult flag to obtain a compact result (if true, a vector that doesn't contain any matches for a given query is not inserted in final result)
        """

    def knnMatchQuery(self, queryDescriptors, matches, k, masks=..., compactResult=...) -> None:
        """
        @overload
        @param queryDescriptors query descriptors
        @param matches vector to host retrieved matches
        @param k number of the closest descriptors to be returned for every input query
        @param masks vector of masks to select which input descriptors must be matched to ones in dataset (the *i*-th mask in vector indicates whether each input query can be matched with descriptors in dataset relative to *i*-th image)
        @param compactResult flag to obtain a compact result (if true, a vector that doesn't contain any matches for a given query is not inserted in final result)
        """

    def match(self, queryDescriptors, trainDescriptors, mask=...) -> matches:
        """
        @brief For every input query descriptor, retrieve the best matching one from a dataset provided from user
        or from the one internal to class

        @param queryDescriptors query descriptors
        @param trainDescriptors dataset of descriptors furnished by user
        @param matches vector to host retrieved matches
        @param mask mask to select which input descriptors must be matched to one in dataset
        """

    def matchQuery(self, queryDescriptors, masks=...) -> matches:
        """
        @overload
        @param queryDescriptors query descriptors
        @param matches vector to host retrieved matches
        @param masks vector of masks to select which input descriptors must be matched to one in dataset (the *i*-th mask in vector indicates whether each input query can be matched with descriptors in dataset relative to *i*-th image)
        """

class DrawLinesMatchesFlags(builtins.object): ...

class KeyLine(builtins.object):
    def getEndPoint(self) -> retval:
        """
        Returns the end point of the line in the original image
        """

    def getEndPointInOctave(self) -> retval:
        """
        Returns the end point of the line in the octave it was extracted from
        """

    def getStartPoint(self) -> retval:
        """
        Returns the start point of the line in the original image
        """

    def getStartPointInOctave(self) -> retval:
        """
        Returns the start point of the line in the octave it was extracted from
        """

class LSDDetector(cv2.Algorithm):
    @overload
    def detect(self, image, scale, numOctaves, mask=...) -> keypoints:
        """
        @brief Detect lines inside an image.

        @param image input image
        @param keypoints vector that will store extracted lines for one or more images
        @param scale scale factor used in pyramids generation
        @param numOctaves number of octaves inside pyramid
        @param mask mask matrix to detect only KeyLines of interest
        """

    @overload
    def detect(self, images, keylines, scale, numOctaves, masks=...) -> None:
        """
        @overload
        @param images input images
        @param keylines set of vectors that will store extracted lines for one or more images
        @param scale scale factor used in pyramids generation
        @param numOctaves number of octaves inside pyramid
        @param masks vector of mask matrices to detect only KeyLines of interest from each input image
        """

    def createLSDDetector(self) -> retval:
        """
        @brief Creates ad LSDDetector object, using smart pointers.
        """

    def createLSDDetectorWithParams(self, params) -> retval:
        """"""

class LSDParam(builtins.object): ...

def BinaryDescriptor_createBinaryDescriptor() -> retval:
    """
    @brief Create a BinaryDescriptor object with default parameters (or with the ones provided)
      and return a smart pointer to it
    """

def LSDDetector_createLSDDetector() -> retval:
    """
    @brief Creates ad LSDDetector object, using smart pointers.
    """

def LSDDetector_createLSDDetectorWithParams(params) -> retval:
    """
    .
    """

def drawKeylines(image, keylines, outImage=..., color=..., flags=...) -> outImage:
    """
    @brief Draws keylines.

    @param image input image
    @param keylines keylines to be drawn
    @param outImage output image to draw on
    @param color color of lines to be drawn (if set to defaul value, color is chosen randomly)
    @param flags drawing flags
    """

def drawLineMatches(img1, keylines1, img2, keylines2, matches1to2, outImg=..., matchColor=..., singleLineColor=..., matchesMask=..., flags=...) -> outImg:
    """
    @brief Draws the found matches of keylines from two images.

    @param img1 first image
    @param keylines1 keylines extracted from first image
    @param img2 second image
    @param keylines2 keylines extracted from second image
    @param matches1to2 vector of matches
    @param outImg output matrix to draw on
    @param matchColor drawing color for matches (chosen randomly in case of default value)
    @param singleLineColor drawing color for keylines (chosen randomly in case of default value)
    @param matchesMask mask to indicate which matches must be drawn
    @param flags drawing flags, see DrawLinesMatchesFlags

    @note If both *matchColor* and *singleLineColor* are set to their default values, function draws
    matched lines and line connecting them with same color
    """
