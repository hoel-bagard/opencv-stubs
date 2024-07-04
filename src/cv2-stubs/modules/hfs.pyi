from typing import Any, TypeAlias

from .. import functions as cv2

retval: TypeAlias = Any

class HfsSegment(cv2.Algorithm):
    def getMinRegionSizeI(self) -> retval:
        """"""

    def getMinRegionSizeII(self) -> retval:
        """"""

    def getNumSlicIter(self) -> retval:
        """"""

    def getSegEgbThresholdI(self) -> retval:
        """"""

    def getSegEgbThresholdII(self) -> retval:
        """"""

    def getSlicSpixelSize(self) -> retval:
        """"""

    def getSpatialWeight(self) -> retval:
        """"""

    def performSegmentCpu(self, src, ifDraw=...) -> retval:
        """
        @brief do segmentation with cpu
        * This method is only implemented for reference.
        * It is highly NOT recommanded to use it.
        """

    def performSegmentGpu(self, src, ifDraw=...) -> retval:
        """
        @brief do segmentation gpu
        * @param src: the input image
        * @param ifDraw: if draw the image in the returned Mat. if this parameter is false, * then the content of the returned Mat is a matrix of index, describing the region * each pixel belongs to. And it's data type is CV_16U. If this parameter is true, * then the returned Mat is a segmented picture, and color of each region is the * average color of all pixels in that region. And it's data type is the same as * the input image
        """

    def setMinRegionSizeI(self, n) -> None:
        """
        @brief: set and get the parameter minRegionSizeI.
        * This parameter is used in the second stage
        * mentioned above. After the EGB segmentation, regions that have fewer
        * pixels then this parameter will be merged into it's adjacent region.
        """

    def setMinRegionSizeII(self, n) -> None:
        """
        @brief: set and get the parameter minRegionSizeII.
        * This parameter is used in the third stage
        * mentioned above. It serves the same purpose as minRegionSizeI
        """

    def setNumSlicIter(self, n) -> None:
        """
        @brief: set and get the parameter numSlicIter.
        * This parameter is used in the first stage. It
        * describes how many iteration to perform when executing SLIC.
        """

    def setSegEgbThresholdI(self, c) -> None:
        """
        @brief: set and get the parameter segEgbThresholdI.
        * This parameter is used in the second stage mentioned above.
        * It is a constant used to threshold weights of the edge when merging
        * adjacent nodes when applying EGB algorithm. The segmentation result
        * tends to have more regions remained if this value is large and vice versa.
        """

    def setSegEgbThresholdII(self, c) -> None:
        """
        @brief: set and get the parameter segEgbThresholdII.
        * This parameter is used in the third stage
        * mentioned above. It serves the same purpose as segEgbThresholdI.
        * The segmentation result tends to have more regions remained if
        * this value is large and vice versa.
        """

    def setSlicSpixelSize(self, n) -> None:
        """
        @brief: set and get the parameter slicSpixelSize.
        * This parameter is used in the first stage mentioned
        * above(the SLIC stage). It describes the size of each
        * superpixel when initializing SLIC. Every superpixel
        * approximately has \f$slicSpixelSize \times slicSpixelSize\f$
        * pixels in the beginning.
        """

    def setSpatialWeight(self, w) -> None:
        """
        @brief: set and get the parameter spatialWeight.
        * This parameter is used in the first stage
        * mentioned above(the SLIC stage). It describes how important is the role
        * of position when calculating the distance between each pixel and it's
        * center. The exact formula to calculate the distance is
        * \f$colorDistance + spatialWeight \times spatialDistance\f$.
        * The segmentation result tends to have more local consistency
        * if this value is larger.
        """

    def create(self, height, width, segEgbThresholdI=..., minRegionSizeI=..., segEgbThresholdII=..., minRegionSizeII=..., spatialWeight=..., slicSpixelSize=..., numSlicIter=...) -> retval:
        """
        @brief: create a hfs object
        * @param height: the height of the input image
        * @param width: the width of the input image
        * @param segEgbThresholdI: parameter segEgbThresholdI
        * @param minRegionSizeI: parameter minRegionSizeI
        * @param segEgbThresholdII: parameter segEgbThresholdII
        * @param minRegionSizeII: parameter minRegionSizeII
        * @param spatialWeight: parameter spatialWeight
        * @param slicSpixelSize: parameter slicSpixelSize
        * @param numSlicIter: parameter numSlicIter
        """

def HfsSegment_create(height, width, segEgbThresholdI=..., minRegionSizeI=..., segEgbThresholdII=..., minRegionSizeII=..., spatialWeight=..., slicSpixelSize=..., numSlicIter=...) -> retval:
    """
    @brief: create a hfs object
    * @param height: the height of the input image
    * @param width: the width of the input image
    * @param segEgbThresholdI: parameter segEgbThresholdI
    * @param minRegionSizeI: parameter minRegionSizeI
    * @param segEgbThresholdII: parameter segEgbThresholdII
    * @param minRegionSizeII: parameter minRegionSizeII
    * @param spatialWeight: parameter spatialWeight
    * @param slicSpixelSize: parameter slicSpixelSize
    * @param numSlicIter: parameter numSlicIter
    """
