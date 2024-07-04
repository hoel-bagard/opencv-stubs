import builtins
from typing import Any, Final, TypeAlias

import numpy as _np
import numpy.typing as _npt

from ... import functions as cv2
from . import segmentation

lines: TypeAlias = Any
qnimg: TypeAlias = Any
out: TypeAlias = Any
image: TypeAlias = Any
alphaPhiST: TypeAlias = Any
labels_out: TypeAlias = Any
dense_flow: TypeAlias = Any
dist: TypeAlias = Any
filtered_disparity_map: TypeAlias = Any
ellipses: TypeAlias = Any
boxes: TypeAlias = Any
scores: TypeAlias = Any
qimg: TypeAlias = Any
result: TypeAlias = Any
qcimg: TypeAlias = Any
_dst: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any

class AdaptiveManifoldFilter(cv2.Algorithm):
    def collectGarbage(self) -> None:
        """"""

    def filter(self, src, dst=..., joint=...) -> dst:
        """
        @brief Apply high-dimensional filtering using adaptive manifolds.

        @param src filtering image with any numbers of channels.
        @param dst output image.
        @param joint optional joint (also called as guided) image with any numbers of channels.
        """

    def create(self) -> retval:
        """"""

class ContourFitting(cv2.Algorithm):
    def estimateTransformation(self, src, dst, alphaPhiST=..., fdContour=...) -> tuple[alphaPhiST, dist]:
        """
        @brief Fit two closed curves using fourier descriptors. More details in @cite PersoonFu1977 and @cite BergerRaghunathan1998

        @param src Contour defining first shape.
        @param dst Contour defining second shape (Target).
        @param alphaPhiST : \f$ \alpha \f$=alphaPhiST(0,0), \f$ \phi \f$=alphaPhiST(0,1) (in radian), s=alphaPhiST(0,2), Tx=alphaPhiST(0,3), Ty=alphaPhiST(0,4) rotation center
        @param dist distance between src and dst after matching.
        @param fdContour false then src and dst are contours and true src and dst are fourier descriptors.
        """

    def getCtrSize(self) -> retval:
        """
        @returns number of fourier descriptors
        """

    def getFDSize(self) -> retval:
        """
        @returns number of fourier descriptors used for optimal curve matching
        """

    def setCtrSize(self, n) -> None:
        """
        @brief set number of Fourier descriptors used in estimateTransformation

        @param n number of Fourier descriptors equal to number of contour points after resampling.
        """

    def setFDSize(self, n) -> None:
        """
        @brief set number of Fourier descriptors when estimateTransformation used vector<Point>

        @param n number of fourier descriptors used for optimal curve matching.
        """

class DTFilter(cv2.Algorithm):
    def filter(self, src, dst=..., dDepth=...) -> dst:
        """
        @brief Produce domain transform filtering operation on source image.

        @param src filtering image with unsigned 8-bit or floating-point 32-bit depth and up to 4 channels.
        @param dst destination image.
        @param dDepth optional depth of the output image. dDepth can be set to -1, which will be equivalent to src.depth().
        """

class DisparityFilter(cv2.Algorithm):
    def filter(self, disparity_map_left, left_view, filtered_disparity_map=..., disparity_map_right=..., ROI=..., right_view=...) -> filtered_disparity_map:
        """
        @brief Apply filtering to the disparity map.

        @param disparity_map_left disparity map of the left view, 1 channel, CV_16S type. Implicitly assumes that disparity values are scaled by 16 (one-pixel disparity corresponds to the value of 16 in the disparity map). Disparity map can have any resolution, it will be automatically resized to fit left_view resolution.
        @param left_view left view of the original stereo-pair to guide the filtering process, 8-bit single-channel or three-channel image.
        @param filtered_disparity_map output disparity map.
        @param disparity_map_right optional argument, some implementations might also use the disparity map of the right view to compute confidence maps, for instance.
        @param ROI region of the disparity map to filter. Optional, usually it should be set automatically.
        @param right_view optional argument, some implementations might also use the right view of the original stereo-pair.
        """

class DisparityWLSFilter(DisparityFilter):
    def getConfidenceMap(self) -> retval:
        """
        @brief Get the confidence map that was used in the last filter call. It is a CV_32F one-channel image
        with values ranging from 0.0 (totally untrusted regions of the raw disparity map) to 255.0 (regions containing
        correct disparity values with a high degree of confidence).
        """

    def getDepthDiscontinuityRadius(self) -> retval:
        """
        @brief DepthDiscontinuityRadius is a parameter used in confidence computation. It defines the size of
        low-confidence regions around depth discontinuities.
        """

    def getLRCthresh(self) -> retval:
        """
        @brief LRCthresh is a threshold of disparity difference used in left-right-consistency check during
        confidence map computation. The default value of 24 (1.5 pixels) is virtually always good enough.
        """

    def getLambda(self) -> retval:
        """
        @brief Lambda is a parameter defining the amount of regularization during filtering. Larger values force
        filtered disparity map edges to adhere more to source image edges. Typical value is 8000.
        """

    def getROI(self) -> retval:
        """
        @brief Get the ROI used in the last filter call
        """

    def getSigmaColor(self) -> retval:
        """
        @brief SigmaColor is a parameter defining how sensitive the filtering process is to source image edges.
        Large values can lead to disparity leakage through low-contrast edges. Small values can make the filter too
        sensitive to noise and textures in the source image. Typical values range from 0.8 to 2.0.
        """

    def setDepthDiscontinuityRadius(self, _disc_radius) -> None:
        """
        @see getDepthDiscontinuityRadius
        """

    def setLRCthresh(self, _LRC_thresh) -> None:
        """
        @see getLRCthresh
        """

    def setLambda(self, _lambda) -> None:
        """
        @see getLambda
        """

    def setSigmaColor(self, _sigma_color) -> None:
        """
        @see getSigmaColor
        """

class EdgeAwareInterpolator(SparseMatchInterpolator):
    def getFGSLambda(self) -> retval:
        """
        @see setFGSLambda
        """

    def getFGSSigma(self) -> retval:
        """
        @see setFGSLambda
        """

    def getK(self) -> retval:
        """
        @see setK
        """

    def getLambda(self) -> retval:
        """
        @see setLambda
        """

    def getSigma(self) -> retval:
        """
        @see setSigma
        """

    def getUsePostProcessing(self) -> retval:
        """
        @see setUsePostProcessing
        """

    def setCostMap(self, _costMap) -> None:
        """
        @brief Interface to provide a more elaborated cost map, i.e. edge map, for the edge-aware term.
        *  This implementation is based on a rather simple gradient-based edge map estimation.
        *  To used more complex edge map estimator (e.g. StructuredEdgeDetection that has been
        *  used in the original publication) that may lead to improved accuracies, the internal
        *  edge map estimation can be bypassed here.
        *  @param _costMap a type CV_32FC1 Mat is required. *  @see cv::ximgproc::createSuperpixelSLIC
        """

    def setFGSLambda(self, _lambda) -> None:
        """
        @brief Sets the respective fastGlobalSmootherFilter() parameter.
        """

    def setFGSSigma(self, _sigma) -> None:
        """
        @see setFGSLambda
        """

    def setK(self, _k) -> None:
        """
        @brief K is a number of nearest-neighbor matches considered, when fitting a locally affine
        model. Usually it should be around 128. However, lower values would make the interpolation
        noticeably faster.
        """

    def setLambda(self, _lambda) -> None:
        """
        @brief Lambda is a parameter defining the weight of the edge-aware term in geodesic distance,
        should be in the range of 0 to 1000.
        """

    def setSigma(self, _sigma) -> None:
        """
        @brief Sigma is a parameter defining how fast the weights decrease in the locally-weighted affine
        fitting. Higher values can help preserve fine details, lower values can help to get rid of noise in the
        output flow.
        """

    def setUsePostProcessing(self, _use_post_proc) -> None:
        """
        @brief Sets whether the fastGlobalSmootherFilter() post-processing is employed. It is turned on by
        default.
        """

class EdgeBoxes(cv2.Algorithm):
    def getAlpha(self) -> retval:
        """
        @brief Returns the step size of sliding window search.
        """

    def getBeta(self) -> retval:
        """
        @brief Returns the nms threshold for object proposals.
        """

    def getBoundingBoxes(self, edge_map, orientation_map, scores=...) -> tuple[boxes, scores]:
        """
        @brief Returns array containing proposal boxes.

        @param edge_map edge image.
        @param orientation_map orientation map.
        @param boxes proposal boxes.
        @param scores of the proposal boxes, provided a vector of float types.
        """

    def getClusterMinMag(self) -> retval:
        """
        @brief Returns the cluster min magnitude.
        """

    def getEdgeMergeThr(self) -> retval:
        """
        @brief Returns the edge merge threshold.
        """

    def getEdgeMinMag(self) -> retval:
        """
        @brief Returns the edge min magnitude.
        """

    def getEta(self) -> retval:
        """
        @brief Returns adaptation rate for nms threshold.
        """

    def getGamma(self) -> retval:
        """
        @brief Returns the affinity sensitivity.
        """

    def getKappa(self) -> retval:
        """
        @brief Returns the scale sensitivity.
        """

    def getMaxAspectRatio(self) -> retval:
        """
        @brief Returns the max aspect ratio of boxes.
        """

    def getMaxBoxes(self) -> retval:
        """
        @brief Returns the max number of boxes to detect.
        """

    def getMinBoxArea(self) -> retval:
        """
        @brief Returns the minimum area of boxes.
        """

    def getMinScore(self) -> retval:
        """
        @brief Returns the min score of boxes to detect.
        """

    def setAlpha(self, value) -> None:
        """
        @brief Sets the step size of sliding window search.
        """

    def setBeta(self, value) -> None:
        """
        @brief Sets the nms threshold for object proposals.
        """

    def setClusterMinMag(self, value) -> None:
        """
        @brief Sets the cluster min magnitude.
        """

    def setEdgeMergeThr(self, value) -> None:
        """
        @brief Sets the edge merge threshold.
        """

    def setEdgeMinMag(self, value) -> None:
        """
        @brief Sets the edge min magnitude.
        """

    def setEta(self, value) -> None:
        """
        @brief Sets the adaptation rate for nms threshold.
        """

    def setGamma(self, value) -> None:
        """
        @brief Sets the affinity sensitivity
        """

    def setKappa(self, value) -> None:
        """
        @brief Sets the scale sensitivity.
        """

    def setMaxAspectRatio(self, value) -> None:
        """
        @brief Sets the max aspect ratio of boxes.
        """

    def setMaxBoxes(self, value) -> None:
        """
        @brief Sets max number of boxes to detect.
        """

    def setMinBoxArea(self, value) -> None:
        """
        @brief Sets the minimum area of boxes.
        """

    def setMinScore(self, value) -> None:
        """
        @brief Sets the min score of boxes to detect.
        """

class EdgeDrawing(cv2.Algorithm):
    def detectEdges(self, src) -> None:
        """
        @brief Detects edges in a grayscale image and prepares them to detect lines and ellipses.

        @param src 8-bit, single-channel, grayscale input image.
        """

    def detectEllipses(self, ellipses=...) -> ellipses:
        """
        @brief Detects circles and ellipses.

        @param ellipses  output Vec<6d> contains center point and perimeter for circles, center point, axes and angle for ellipses. @note you should call detectEdges() before calling this function.
        """

    def detectLines(self, lines=...) -> lines:
        """
        @brief Detects lines.

        @param lines  output Vec<4f> contains the start point and the end point of detected lines. @note you should call detectEdges() before calling this function.
        """

    def getEdgeImage(self, dst=...) -> dst:
        """
        @brief returns Edge Image prepared by detectEdges() function.

        @param dst returns 8-bit, single-channel output image.
        """

    def getGradientImage(self, dst=...) -> dst:
        """
        @brief returns Gradient Image prepared by detectEdges() function.

        @param dst returns 16-bit, single-channel output image.
        """

    def getSegmentIndicesOfLines(self) -> retval:
        """
        @brief Returns for each line found in detectLines() its edge segment index in getSegments()
        """

    def getSegments(self) -> retval:
        """
        @brief Returns std::vector<std::vector<Point>> of detected edge segments, see detectEdges()
        """

    def setParams(self, parameters) -> None:
        """
        @brief sets parameters.

        this function is meant to be used for parameter setting in other languages than c++ like python.
        @param parameters
        """

    class Params(builtins.object):
        AnchorThresholdValue: int = 0
        EdgeDetectionOperator: int = 0
        GradientThresholdValue: int = 20
        LineFitErrorThreshold: float = 1.0
        MaxDistanceBetweenTwoLines: float = 6.0
        MaxErrorThreshold: float = 1.3
        MinLineLength: int = -1
        MinPathLength: int = 10
        NFAValidation: bool = True
        PFmode: bool = False
        ScanInterval: int = 1
        Sigma: float = 1.0
        SumFlag: bool = True

class FastBilateralSolverFilter(cv2.Algorithm):
    def filter(self, src, confidence, dst=...) -> dst:
        """
        @brief Apply smoothing operation to the source image.

        @param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point 32-bit depth and up to 3 channels.
        @param confidence confidence image with unsigned 8-bit or floating-point 32-bit confidence and 1 channel.
        @param dst destination image.  @note Confidence images with CV_8U depth are expected to in [0, 255] and CV_32F in [0, 1] range.
        """

class FastGlobalSmootherFilter(cv2.Algorithm):
    def filter(self, src, dst=...) -> dst:
        """
        @brief Apply smoothing operation to the source image.

        @param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point 32-bit depth and up to 4 channels.
        @param dst destination image.
        """

class FastLineDetector(cv2.Algorithm):
    def detect(self, image, lines=...) -> lines:
        """
        @brief Finds lines in the input image.
        This is the output of the default parameters of the algorithm on the above
        shown image.

        ![image](pics/corridor_fld.jpg)

        @param image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use: `fld_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
        @param lines A vector of Vec4f elements specifying the beginning and ending point of a line.  Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are directed so that the brighter side is on their left.
        """

    def drawSegments(self, image, lines, draw_arrow=..., linecolor=..., linethickness=...) -> image:
        """
        @brief Draws the line segments on a given image.
        @param image The image, where the lines will be drawn. Should be bigger or equal to the image, where the lines were found.
        @param lines A vector of the lines that needed to be drawn.
        @param draw_arrow If true, arrow heads will be drawn.
        @param linecolor Line color.
        @param linethickness Line thickness.
        """

class GuidedFilter(cv2.Algorithm):
    def filter(self, src, dst=..., dDepth=...) -> dst:
        """
        @brief Apply Guided Filter to the filtering image.

        @param src filtering image with any numbers of channels.
        @param dst output image.
        @param dDepth optional depth of the output image. dDepth can be set to -1, which will be equivalent to src.depth().
        """

class RFFeatureGetter(cv2.Algorithm):
    def getFeatures(self, src, features, gnrmRad, gsmthRad, shrink, outNum, gradNum) -> None:
        """"""

class RICInterpolator(SparseMatchInterpolator):
    def getAlpha(self) -> retval:
        """
        @copybrief setAlpha
        *  @see setAlpha
        """

    def getFGSLambda(self) -> retval:
        """
        @copybrief setFGSLambda
        *  @see setFGSLambda
        """

    def getFGSSigma(self) -> retval:
        """
        @copybrief setFGSSigma
        *  @see setFGSSigma
        """

    def getK(self) -> retval:
        """
        @copybrief setK
        *  @see setK
        """

    def getMaxFlow(self) -> retval:
        """
        @copybrief setMaxFlow
        *  @see setMaxFlow
        """

    def getModelIter(self) -> retval:
        """
        @copybrief setModelIter
        *  @see setModelIter
        """

    def getRefineModels(self) -> retval:
        """
        @copybrief setRefineModels
        *  @see setRefineModels
        """

    def getSuperpixelMode(self) -> retval:
        """
        @copybrief setSuperpixelMode
        *  @see setSuperpixelMode
        """

    def getSuperpixelNNCnt(self) -> retval:
        """
        @copybrief setSuperpixelNNCnt
        *  @see setSuperpixelNNCnt
        """

    def getSuperpixelRuler(self) -> retval:
        """
        @copybrief setSuperpixelRuler
        *  @see setSuperpixelRuler
        """

    def getSuperpixelSize(self) -> retval:
        """
        @copybrief setSuperpixelSize
        *  @see setSuperpixelSize
        """

    def getUseGlobalSmootherFilter(self) -> retval:
        """
        @copybrief setUseGlobalSmootherFilter
        *  @see setUseGlobalSmootherFilter
        """

    def getUseVariationalRefinement(self) -> retval:
        """
        @copybrief setUseVariationalRefinement
        *  @see setUseVariationalRefinement
        """

    def setAlpha(self, alpha=...) -> None:
        """
        @brief Alpha is a parameter defining a global weight for transforming geodesic distance into weight.
        """

    def setCostMap(self, costMap) -> None:
        """
        @brief Interface to provide a more elaborated cost map, i.e. edge map, for the edge-aware term.
        *  This implementation is based on a rather simple gradient-based edge map estimation.
        *  To used more complex edge map estimator (e.g. StructuredEdgeDetection that has been
        *  used in the original publication) that may lead to improved accuracies, the internal
        *  edge map estimation can be bypassed here.
        *  @param costMap a type CV_32FC1 Mat is required. *  @see cv::ximgproc::createSuperpixelSLIC
        """

    def setFGSLambda(self, lambda_=...) -> None:
        """
        @brief Sets the respective fastGlobalSmootherFilter() parameter.
        """

    def setFGSSigma(self, sigma=...) -> None:
        """
        @brief Sets the respective fastGlobalSmootherFilter() parameter.
        """

    def setK(self, k=...) -> None:
        """
        @brief K is a number of nearest-neighbor matches considered, when fitting a locally affine
        *model for a superpixel segment. However, lower values would make the interpolation
        *noticeably faster. The original implementation of @cite Hu2017 uses 32.
        """

    def setMaxFlow(self, maxFlow=...) -> None:
        """
        @brief MaxFlow is a threshold to validate the predictions using a certain piece-wise affine model.
        * If the prediction exceeds the treshold the translational model will be applied instead.
        """

    def setModelIter(self, modelIter=...) -> None:
        """
        @brief Parameter defining the number of iterations for piece-wise affine model estimation.
        """

    def setRefineModels(self, refineModles=...) -> None:
        """
        @brief Parameter to choose wether additional refinement of the piece-wise affine models is employed.
        """

    def setSuperpixelMode(self, mode=...) -> None:
        """
        @brief Parameter to choose superpixel algorithm variant to use:
        * - cv::ximgproc::SLICType SLIC segments image using a desired region_size (value: 100)
        * - cv::ximgproc::SLICType SLICO will optimize using adaptive compactness factor (value: 101)
        * - cv::ximgproc::SLICType MSLIC will optimize using manifold methods resulting in more content-sensitive superpixels (value: 102).
        *  @see cv::ximgproc::createSuperpixelSLIC
        """

    def setSuperpixelNNCnt(self, spNN=...) -> None:
        """
        @brief Parameter defines the number of nearest-neighbor matches for each superpixel considered, when fitting a locally affine
        *model.
        """

    def setSuperpixelRuler(self, ruler=...) -> None:
        """
        @brief Parameter to tune enforcement of superpixel smoothness factor used for oversegmentation.
        *  @see cv::ximgproc::createSuperpixelSLIC
        """

    def setSuperpixelSize(self, spSize=...) -> None:
        """
        @brief Get the internal cost, i.e. edge map, used for estimating the edge-aware term.
        *  @see setCostMap
        """

    def setUseGlobalSmootherFilter(self, use_FGS=...) -> None:
        """
        @brief Sets whether the fastGlobalSmootherFilter() post-processing is employed.
        """

    def setUseVariationalRefinement(self, use_variational_refinement=...) -> None:
        """
        @brief Parameter to choose wether the VariationalRefinement post-processing  is employed.
        """

class RidgeDetectionFilter(cv2.Algorithm):
    def getRidgeFilteredImage(self, _img, out=...) -> out:
        """
        @brief Apply Ridge detection filter on input image.
        @param _img InputArray as supported by Sobel. img can be 1-Channel or 3-Channels.
        @param out OutputAray of structure as RidgeDetectionFilter::ddepth. Output image with ridges.
        """

    def create(self, ddepth=..., dx=..., dy=..., ksize=..., out_dtype=..., scale=..., delta=..., borderType=...) -> retval:
        """
        @brief Create pointer to the Ridge detection filter.
        @param ddepth  Specifies output image depth. Defualt is CV_32FC1
        @param dx Order of derivative x, default is 1
        @param dy  Order of derivative y, default is 1
        @param ksize Sobel kernel size , default is 3
        @param out_dtype Converted format for output, default is CV_8UC1
        @param scale Optional scale value for derivative values, default is 1
        @param delta  Optional bias added to output, default is 0
        @param borderType Pixel extrapolation method, default is BORDER_DEFAULT @see Sobel, threshold, getStructuringElement, morphologyEx.( for additional refinement)
        """

class ScanSegment(cv2.Algorithm):
    def getLabelContourMask(self, image=..., thick_line=...) -> image:
        """
        @brief Returns the mask of the superpixel segmentation stored in the ScanSegment object.

        The function return the boundaries of the superpixel segmentation.

        @param image Return: CV_8UC1 image mask where -1 indicates that the pixel is a superpixel border, and 0 otherwise.
        @param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border are masked.
        """

    def getLabels(self, labels_out=...) -> labels_out:
        """
        @brief Returns the segmentation labeling of the image.

        Each label represents a superpixel, and each pixel is assigned to one superpixel label.

        @param labels_out Return: A CV_32UC1 integer array containing the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
        """

    def getNumberOfSuperpixels(self) -> retval:
        """
        @brief Returns the actual superpixel segmentation from the last image processed using iterate.

        Returns zero if no image has been processed.
        """

    def iterate(self, img) -> None:
        """
        @brief Calculates the superpixel segmentation on a given image with the initialized
        parameters in the ScanSegment object.

        This function can be called again for other images without the need of initializing the algorithm with createScanSegment().
        This save the computational cost of allocating memory for all the structures of the algorithm.

        @param img Input image. Supported format: CV_8UC3. Image size must match with the initialized image size with the function createScanSegment(). It MUST be in Lab color space.
        """

class SparseMatchInterpolator(cv2.Algorithm):
    def interpolate(self, from_image, from_points, to_image, to_points, dense_flow=...) -> dense_flow:
        """
        @brief Interpolate input sparse matches.

        @param from_image first of the two matched images, 8-bit single-channel or three-channel.
        @param from_points points of the from_image for which there are correspondences in the to_image (Point2f vector or Mat of depth CV_32F)
        @param to_image second of the two matched images, 8-bit single-channel or three-channel.
        @param to_points points in the to_image corresponding to from_points (Point2f vector or Mat of depth CV_32F)
        @param dense_flow output dense matching (two-channel CV_32F image)
        """

class StructuredEdgeDetection(cv2.Algorithm):
    def computeOrientation(self, src, dst=...) -> dst:
        """
        @brief The function computes orientation from edge image.

        @param src edge image.
        @param dst orientation image.
        """

    def detectEdges(self, src, dst=...) -> dst:
        """
        @brief The function detects edges in src and draw them to dst.

        The algorithm underlies this function is much more robust to texture presence, than common
        approaches, e.g. Sobel
        @param src source image (RGB, float, in [0;1]) to detect edges
        @param dst destination image (grayscale, float, in [0;1]) where edges are drawn @sa Sobel, Canny
        """

    def edgesNms(self, edge_image, orientation_image, dst=..., r=..., s=..., m=..., isParallel=...) -> dst:
        """
        @brief The function edgenms in edge image and suppress edges where edge is stronger in orthogonal direction.

        @param edge_image edge image from detectEdges function.
        @param orientation_image orientation image from computeOrientation function.
        @param dst suppressed image (grayscale, float, in [0;1])
        @param r radius for NMS suppression.
        @param s radius for boundary suppression.
        @param m multiplier for conservative suppression.
        @param isParallel enables/disables parallel computing.
        """

class SuperpixelLSC(cv2.Algorithm):
    def enforceLabelConnectivity(self, min_element_size=...) -> None:
        """
        @brief Enforce label connectivity.

        @param min_element_size The minimum element size in percents that should be absorbed into a bigger superpixel. Given resulted average superpixel size valid value should be in 0-100 range, 25 means that less then a quarter sized superpixel should be absorbed, this is default.  The function merge component that is too small, assigning the previously found adjacent label to this component. Calling this function may change the final number of superpixels.
        """

    def getLabelContourMask(self, image=..., thick_line=...) -> image:
        """
        @brief Returns the mask of the superpixel segmentation stored in SuperpixelLSC object.

        @param image Return: CV_8U1 image mask where -1 indicates that the pixel is a superpixel border, and 0 otherwise.
        @param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border are masked.  The function return the boundaries of the superpixel segmentation.
        """

    def getLabels(self, labels_out=...) -> labels_out:
        """
        @brief Returns the segmentation labeling of the image.

        Each label represents a superpixel, and each pixel is assigned to one superpixel label.

        @param labels_out Return: A CV_32SC1 integer array containing the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].  The function returns an image with the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
        """

    def getNumberOfSuperpixels(self) -> retval:
        """
        @brief Calculates the actual amount of superpixels on a given segmentation computed
        and stored in SuperpixelLSC object.
        """

    def iterate(self, num_iterations=...) -> None:
        """
        @brief Calculates the superpixel segmentation on a given image with the initialized
        parameters in the SuperpixelLSC object.

        This function can be called again without the need of initializing the algorithm with
        createSuperpixelLSC(). This save the computational cost of allocating memory for all the
        structures of the algorithm.

        @param num_iterations Number of iterations. Higher number improves the result.  The function computes the superpixels segmentation of an image with the parameters initialized with the function createSuperpixelLSC(). The algorithms starts from a grid of superpixels and then refines the boundaries by proposing updates of edges boundaries.
        """

class SuperpixelSEEDS(cv2.Algorithm):
    def getLabelContourMask(self, image=..., thick_line=...) -> image:
        """
        @brief Returns the mask of the superpixel segmentation stored in SuperpixelSEEDS object.

        @param image Return: CV_8UC1 image mask where -1 indicates that the pixel is a superpixel border, and 0 otherwise.
        @param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border are masked.  The function return the boundaries of the superpixel segmentation.  @note -   (Python) A demo on how to generate superpixels in images from the webcam can be found at opencv_source_code/samples/python2/seeds.py -   (cpp) A demo on how to generate superpixels in images from the webcam can be found at opencv_source_code/modules/ximgproc/samples/seeds.cpp. By adding a file image as a command line argument, the static image will be used instead of the webcam. -   It will show a window with the video from the webcam with the superpixel boundaries marked in red (see below). Use Space to switch between different output modes. At the top of the window there are 4 sliders, from which the user can change on-the-fly the number of superpixels, the number of block levels, the strength of the boundary prior term to modify the shape, and the number of iterations at pixel level. This is useful to play with the parameters and set them to the user convenience. In the console the frame-rate of the algorithm is indicated.  ![image](pics/superpixels_demo.png)
        """

    def getLabels(self, labels_out=...) -> labels_out:
        """
        @brief Returns the segmentation labeling of the image.

        Each label represents a superpixel, and each pixel is assigned to one superpixel label.

        @param labels_out Return: A CV_32UC1 integer array containing the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].  The function returns an image with ssthe labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
        """

    def getNumberOfSuperpixels(self) -> retval:
        """
        @brief Calculates the superpixel segmentation on a given image stored in SuperpixelSEEDS object.

        The function computes the superpixels segmentation of an image with the parameters initialized
        with the function createSuperpixelSEEDS().
        """

    def iterate(self, img, num_iterations=...) -> None:
        """
        @brief Calculates the superpixel segmentation on a given image with the initialized
        parameters in the SuperpixelSEEDS object.

        This function can be called again for other images without the need of initializing the
        algorithm with createSuperpixelSEEDS(). This save the computational cost of allocating memory
        for all the structures of the algorithm.

        @param img Input image. Supported formats: CV_8U, CV_16U, CV_32F. Image size & number of channels must match with the initialized image size & channels with the function createSuperpixelSEEDS(). It should be in HSV or Lab color space. Lab is a bit better, but also slower.
        @param num_iterations Number of pixel level iterations. Higher number improves the result.  The function computes the superpixels segmentation of an image with the parameters initialized with the function createSuperpixelSEEDS(). The algorithms starts from a grid of superpixels and then refines the boundaries by proposing updates of blocks of pixels that lie at the boundaries from large to smaller size, finalizing with proposing pixel updates. An illustrative example can be seen below.  ![image](pics/superpixels_blocks2.png)
        """

class SuperpixelSLIC(cv2.Algorithm):
    def enforceLabelConnectivity(self, min_element_size=...) -> None:
        """
        @brief Enforce label connectivity.

        @param min_element_size The minimum element size in percents that should be absorbed into a bigger superpixel. Given resulted average superpixel size valid value should be in 0-100 range, 25 means that less then a quarter sized superpixel should be absorbed, this is default.  The function merge component that is too small, assigning the previously found adjacent label to this component. Calling this function may change the final number of superpixels.
        """

    def getLabelContourMask(self, image=..., thick_line=...) -> image:
        """
        @brief Returns the mask of the superpixel segmentation stored in SuperpixelSLIC object.

        @param image Return: CV_8U1 image mask where -1 indicates that the pixel is a superpixel border, and 0 otherwise.
        @param thick_line If false, the border is only one pixel wide, otherwise all pixels at the border are masked.  The function return the boundaries of the superpixel segmentation.
        """

    def getLabels(self, labels_out=...) -> labels_out:
        """
        @brief Returns the segmentation labeling of the image.

        Each label represents a superpixel, and each pixel is assigned to one superpixel label.

        @param labels_out Return: A CV_32SC1 integer array containing the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].  The function returns an image with the labels of the superpixel segmentation. The labels are in the range [0, getNumberOfSuperpixels()].
        """

    def getNumberOfSuperpixels(self) -> retval:
        """
        @brief Calculates the actual amount of superpixels on a given segmentation computed
        and stored in SuperpixelSLIC object.
        """

    def iterate(self, num_iterations=...) -> None:
        """
        @brief Calculates the superpixel segmentation on a given image with the initialized
        parameters in the SuperpixelSLIC object.

        This function can be called again without the need of initializing the algorithm with
        createSuperpixelSLIC(). This save the computational cost of allocating memory for all the
        structures of the algorithm.

        @param num_iterations Number of iterations. Higher number improves the result.  The function computes the superpixels segmentation of an image with the parameters initialized with the function createSuperpixelSLIC(). The algorithms starts from a grid of superpixels and then refines the boundaries by proposing updates of edges boundaries.
        """

def AdaptiveManifoldFilter_create() -> retval:
    """
    .
    """

def FastHoughTransform(src, dstMatDepth, dst=..., angleRange=..., op=..., makeSkew=...) -> dst:
    """
    * @brief   Calculates 2D Fast Hough transform of an image.
    * @param   dst         The destination image, result of transformation.
    * @param   src         The source (input) image.
    * @param   dstMatDepth The depth of destination image
    * @param   op          The operation to be applied, see cv::HoughOp
    * @param   angleRange  The part of Hough space to calculate, see cv::AngleRangeOption
    * @param   makeSkew    Specifies to do or not to do image skewing, see cv::HoughDeskewOption
    *
    * The function calculates the fast Hough transform for full, half or quarter
    * range of angles.
    """

def GradientDericheX(op, alpha, omega, dst=...) -> dst:
    """
    * @brief   Applies X Deriche filter to an image.
    *
    * For more details about this implementation, please see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
    *
    * @param   op         Source 8-bit or 16bit image, 1-channel or 3-channel image.
    * @param   dst        result CV_32FC image with same number of channel than _op.
    * @param   alpha double see paper
    * @param   omega   double see paper
    *
    """

def GradientDericheY(op, alpha, omega, dst=...) -> dst:
    """
    * @brief   Applies Y Deriche filter to an image.
    *
    * For more details about this implementation, please see http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.5736&rep=rep1&type=pdf
    *
    * @param   op         Source 8-bit or 16bit image, 1-channel or 3-channel image.
    * @param   dst        result CV_32FC image with same number of channel than _op.
    * @param   alpha double see paper
    * @param   omega   double see paper
    *
    """

def HoughPoint2Line(houghPoint, srcImgInfo, angleRange=..., makeSkew=..., rules=...) -> retval:
    """
    * @brief   Calculates coordinates of line segment corresponded by point in Hough space.
    * @param   houghPoint  Point in Hough space.
    * @param   srcImgInfo The source (input) image of Hough transform.
    * @param   angleRange  The part of Hough space where point is situated, see cv::AngleRangeOption
    * @param   makeSkew    Specifies to do or not to do image skewing, see cv::HoughDeskewOption
    * @param   rules       Specifies strictness of line segment calculating, see cv::RulesOption
    * @retval  [Vec4i]     Coordinates of line segment corresponded by point in Hough space.
    * @remarks If rules parameter set to RO_STRICT
               then returned line cut along the border of source image.
    * @remarks If rules parameter set to RO_WEAK then in case of point, which belongs
               the incorrect part of Hough image, returned line will not intersect source image.
    *
    * The function calculates coordinates of line segment corresponded by point in Hough space.
    """

def PeiLinNormalization(I: _npt.NDArray[_np.float64], T: _npt.NDArray[_np.float64] = ...) -> _npt.NDArray[_np.float64]:  # noqa: E741
    """Calculates an affine transformation that normalize given image using Pei&Lin Normalization.

    Assume given image I=T(I¯) where I¯ is a normalized image and T is an affine transformation distorting this image by translation, rotation, scaling and skew. The function returns an affine transformation matrix corresponding to the transformation T-1 described in [PeiLin95]. For more details about this implementation, please see [PeiLin95] Soo-Chang Pei and Chao-Nan Lin. Image normalization for pattern recognition. Image and Vision Computing, Vol. 13, N.10, pp. 711-723, 1995.

    Args:
        I: Given transformed image.

    Returns
        Transformation matrix corresponding to inversed image transformation

    """

def RadonTransform(src, dst=..., theta=..., start_angle=..., end_angle=..., crop=..., norm=...) -> dst:
    """
    * @brief   Calculate Radon Transform of an image.
    * @param   src         The source (input) image.
    * @param   dst         The destination image, result of transformation.
    * @param   theta       Angle resolution of the transform in degrees.
    * @param   start_angle Start angle of the transform in degrees.
    * @param   end_angle   End angle of the transform in degrees.
    * @param   crop        Crop the source image into a circle.
    * @param   norm        Normalize the output Mat to grayscale and convert type to CV_8U
    *
    * This function calculates the Radon Transform of a given image in any range.
    * See https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf for detail.
    * If the input type is CV_8U, the output will be CV_32S.
    * If the input type is CV_32F or CV_64F, the output will be CV_64F
    * The output size will be num_of_integral x src_diagonal_length.
    * If crop is selected, the input image will be crop into square then circle,
    * and output size will be num_of_integral x min_edge.
    *
    """

def RidgeDetectionFilter_create(ddepth=..., dx=..., dy=..., ksize=..., out_dtype=..., scale=..., delta=..., borderType=...) -> retval:
    """
    @brief Create pointer to the Ridge detection filter.
        @param ddepth  Specifies output image depth. Defualt is CV_32FC1
        @param dx Order of derivative x, default is 1
        @param dy  Order of derivative y, default is 1
        @param ksize Sobel kernel size , default is 3
        @param out_dtype Converted format for output, default is CV_8UC1
        @param scale Optional scale value for derivative values, default is 1
        @param delta  Optional bias added to output, default is 0
        @param borderType Pixel extrapolation method, default is BORDER_DEFAULT
        @see Sobel, threshold, getStructuringElement, morphologyEx.( for additional refinement)
    """

def amFilter(joint, src, sigma_s, sigma_r, dst=..., adjust_outliers=...) -> dst:
    """
    @brief Simple one-line Adaptive Manifold Filter call.

    @param joint joint (also called as guided) image or array of images with any numbers of channels.

    @param src filtering image with any numbers of channels.

    @param dst output image.

    @param sigma_s spatial standard deviation.

    @param sigma_r color space standard deviation, it is similar to the sigma in the color space into
    bilateralFilter.

    @param adjust_outliers optional, specify perform outliers adjust operation or not, (Eq. 9) in the
    original paper.

    @note Joint images with CV_8U and CV_16U depth converted to images with CV_32F depth and [0; 1]
    color range before processing. Hence color space sigma sigma_r must be in [0; 1] range, unlike same
    sigmas in bilateralFilter and dtFilter functions. @sa bilateralFilter, dtFilter, guidedFilter
    """

def anisotropicDiffusion(src, alpha, K, niters, dst=...) -> dst:
    """
    @brief Performs anisotropic diffusion on an image.

     The function applies Perona-Malik anisotropic diffusion to an image. This is the solution to the partial differential equation:

     \f[{\frac  {\partial I}{\partial t}}={\mathrm  {div}}\left(c(x,y,t)\nabla I\right)=\nabla c\cdot \nabla I+c(x,y,t)\Delta I\f]

     Suggested functions for c(x,y,t) are:

     \f[c\left(\|\nabla I\|\right)=e^{{-\left(\|\nabla I\|/K\right)^{2}}}\f]

     or

     \f[ c\left(\|\nabla I\|\right)={\frac {1}{1+\left({\frac  {\|\nabla I\|}{K}}\right)^{2}}} \f]

     @param src Source image with 3 channels.
     @param dst Destination image of the same size and the same number of channels as src .
     @param alpha The amount of time to step forward by on each iteration (normally, it's between 0 and 1).
     @param K sensitivity to the edges
     @param niters The number of iterations
    """

def bilateralTextureFilter(src, dst=..., fr=..., numIter=..., sigmaAlpha=..., sigmaAvg=...) -> dst:
    """
    @brief Applies the bilateral texture filter to an image. It performs structure-preserving texture filter.
    For more details about this filter see @cite Cho2014.

    @param src Source image whose depth is 8-bit UINT or 32-bit FLOAT

    @param dst Destination image of the same size and type as src.

    @param fr Radius of kernel to be used for filtering. It should be positive integer

    @param numIter Number of iterations of algorithm, It should be positive integer

    @param sigmaAlpha Controls the sharpness of the weight transition from edges to smooth/texture regions, where
    a bigger value means sharper transition. When the value is negative, it is automatically calculated.

    @param sigmaAvg Range blur parameter for texture blurring. Larger value makes result to be more blurred. When the
    value is negative, it is automatically calculated as described in the paper.

    @sa rollingGuidanceFilter, bilateralFilter
    """

def colorMatchTemplate(img, templ, result=...) -> result:
    """
    * @brief    Compares a color template against overlapped color image regions.
    *
    * @param   img        Image where the search is running. It must be 3 channels image
    * @param   templ       Searched template. It must be not greater than the source image and have 3 channels
    * @param   result     Map of comparison results. It must be single-channel 64-bit floating-point
    """

def computeBadPixelPercent(GT, src, ROI, thresh=...) -> retval:
    """
    @brief Function for computing the percent of "bad" pixels in the disparity map
    (pixels where error is higher than a specified threshold)

    @param GT ground truth disparity map

    @param src disparity map to evaluate

    @param ROI region of interest

    @param thresh threshold used to determine "bad" pixels

    @result returns mean square error between GT and src
    """

def computeMSE(GT, src, ROI) -> retval:
    """
    @brief Function for computing mean square error for disparity maps

    @param GT ground truth disparity map

    @param src disparity map to evaluate

    @param ROI region of interest

    @result returns mean square error between GT and src
    """

def contourSampling(src, nbElt, out=...) -> out:
    """
    * @brief   Contour sampling .
        *
        * @param   src   contour type vector<Point> , vector<Point2f>  or vector<Point2d>
        * @param   out   Mat of type CV_64FC2 and nbElt rows
        * @param   nbElt number of points in out contour
        *
    """

def covarianceEstimation(src, windowRows, windowCols, dst=...) -> dst:
    """
    @brief Computes the estimated covariance matrix of an image using the sliding
    window forumlation.

    @param src The source image. Input image must be of a complex type.
    @param dst The destination estimated covariance matrix. Output matrix will be size (windowRows*windowCols, windowRows*windowCols).
    @param windowRows The number of rows in the window.
    @param windowCols The number of cols in the window.
    The window size parameters control the accuracy of the estimation.
    The sliding window moves over the entire image from the top-left corner
    to the bottom right corner. Each location of the window represents a sample.
    If the window is the size of the image, then this gives the exact covariance matrix.
    For all other cases, the sizes of the window will impact the number of samples
    and the number of elements in the estimated covariance matrix.
    """

def createAMFilter(sigma_s, sigma_r, adjust_outliers=...) -> retval:
    """
    @brief Factory method, create instance of AdaptiveManifoldFilter and produce some initialization routines.

    @param sigma_s spatial standard deviation.

    @param sigma_r color space standard deviation, it is similar to the sigma in the color space into
    bilateralFilter.

    @param adjust_outliers optional, specify perform outliers adjust operation or not, (Eq. 9) in the
    original paper.

    For more details about Adaptive Manifold Filter parameters, see the original article @cite Gastal12 .

    @note Joint images with CV_8U and CV_16U depth converted to images with CV_32F depth and [0; 1]
    color range before processing. Hence color space sigma sigma_r must be in [0; 1] range, unlike same
    sigmas in bilateralFilter and dtFilter functions.
    """

def createContourFitting(ctr=..., fd=...) -> retval:
    """
    * @brief create ContourFitting algorithm object
        *
        * @param ctr number of Fourier descriptors equal to number of contour points after resampling.
        * @param fd Contour defining second shape (Target).
    """

def createDTFilter(guide, sigmaSpatial, sigmaColor, mode=..., numIters=...) -> retval:
    """
    @brief Factory method, create instance of DTFilter and produce initialization routines.

    @param guide guided image (used to build transformed distance, which describes edge structure of
    guided image).

    @param sigmaSpatial \f${\sigma}_H\f$ parameter in the original article, it's similar to the sigma in the
    coordinate space into bilateralFilter.

    @param sigmaColor \f${\sigma}_r\f$ parameter in the original article, it's similar to the sigma in the
    color space into bilateralFilter.

    @param mode one form three modes DTF_NC, DTF_RF and DTF_IC which corresponds to three modes for
    filtering 2D signals in the article.

    @param numIters optional number of iterations used for filtering, 3 is quite enough.

    For more details about Domain Transform filter parameters, see the original article @cite Gastal11 and
    [Domain Transform filter homepage](http://www.inf.ufrgs.br/~eslgastal/DomainTransform/).
    """

def createDisparityWLSFilter(matcher_left) -> retval:
    """
    @brief Convenience factory method that creates an instance of DisparityWLSFilter and sets up all the relevant
    filter parameters automatically based on the matcher instance. Currently supports only StereoBM and StereoSGBM.

    @param matcher_left stereo matcher instance that will be used with the filter
    """

def createDisparityWLSFilterGeneric(use_confidence) -> retval:
    """
    @brief More generic factory method, create instance of DisparityWLSFilter and execute basic
    initialization routines. When using this method you will need to set-up the ROI, matchers and
    other parameters by yourself.

    @param use_confidence filtering with confidence requires two disparity maps (for the left and right views) and is
    approximately two times slower. However, quality is typically significantly better.
    """

def createEdgeAwareInterpolator() -> retval:
    """
    @brief Factory method that creates an instance of the
    EdgeAwareInterpolator.
    """

def createEdgeBoxes(alpha=..., beta=..., eta=..., minScore=..., maxBoxes=..., edgeMinMag=..., edgeMergeThr=..., clusterMinMag=..., maxAspectRatio=..., minBoxArea=..., gamma=..., kappa=...) -> retval:
    """
    @brief Creates a Edgeboxes

    @param alpha step size of sliding window search.
    @param beta nms threshold for object proposals.
    @param eta adaptation rate for nms threshold.
    @param minScore min score of boxes to detect.
    @param maxBoxes max number of boxes to detect.
    @param edgeMinMag edge min magnitude. Increase to trade off accuracy for speed.
    @param edgeMergeThr edge merge threshold. Increase to trade off accuracy for speed.
    @param clusterMinMag cluster min magnitude. Increase to trade off accuracy for speed.
    @param maxAspectRatio max aspect ratio of boxes.
    @param minBoxArea minimum area of boxes.
    @param gamma affinity sensitivity.
    @param kappa scale sensitivity.
    """

def createEdgeDrawing() -> retval:
    """
    @brief Creates a smart pointer to a EdgeDrawing object and initializes it
    """

def createFastBilateralSolverFilter(guide, sigma_spatial, sigma_luma, sigma_chroma, lambda_=..., num_iter=..., max_tol=...) -> retval:
    """
    @brief Factory method, create instance of FastBilateralSolverFilter and execute the initialization routines.

    @param guide image serving as guide for filtering. It should have 8-bit depth and either 1 or 3 channels.

    @param sigma_spatial parameter, that is similar to spatial space sigma (bandwidth) in bilateralFilter.

    @param sigma_luma parameter, that is similar to luma space sigma (bandwidth) in bilateralFilter.

    @param sigma_chroma parameter, that is similar to chroma space sigma (bandwidth) in bilateralFilter.

    @param lambda smoothness strength parameter for solver.

    @param num_iter number of iterations used for solver, 25 is usually enough.

    @param max_tol convergence tolerance used for solver.

    For more details about the Fast Bilateral Solver parameters, see the original paper @cite BarronPoole2016.
    """

def createFastGlobalSmootherFilter(guide, lambda_, sigma_color, lambda_attenuation=..., num_iter=...) -> retval:
    """
    @brief Factory method, create instance of FastGlobalSmootherFilter and execute the initialization routines.

    @param guide image serving as guide for filtering. It should have 8-bit depth and either 1 or 3 channels.

    @param lambda parameter defining the amount of regularization

    @param sigma_color parameter, that is similar to color space sigma in bilateralFilter.

    @param lambda_attenuation internal parameter, defining how much lambda decreases after each iteration. Normally,
    it should be 0.25. Setting it to 1.0 may lead to streaking artifacts.

    @param num_iter number of iterations used for filtering, 3 is usually enough.

    For more details about Fast Global Smoother parameters, see the original paper @cite Min2014. However, please note that
    there are several differences. Lambda attenuation described in the paper is implemented a bit differently so do not
    expect the results to be identical to those from the paper; sigma_color values from the paper should be multiplied by 255.0 to
    achieve the same effect. Also, in case of image filtering where source and guide image are the same, authors
    propose to dynamically update the guide image after each iteration. To maximize the performance this feature
    was not implemented here.
    """

def createFastLineDetector(length_threshold=..., distance_threshold=..., canny_th1=..., canny_th2=..., canny_aperture_size=..., do_merge=...) -> retval:
    """
    @brief Creates a smart pointer to a FastLineDetector object and initializes it

    @param length_threshold    Segment shorter than this will be discarded
    @param distance_threshold  A point placed from a hypothesis line
                               segment farther than this will be regarded as an outlier
    @param canny_th1           First threshold for hysteresis procedure in Canny()
    @param canny_th2           Second threshold for hysteresis procedure in Canny()
    @param canny_aperture_size Aperturesize for the sobel operator in Canny().
                               If zero, Canny() is not applied and the input image is taken as an edge image.
    @param do_merge            If true, incremental merging of segments will be performed
    """

def createGuidedFilter(guide, radius, eps) -> retval:
    """
    @brief Factory method, create instance of GuidedFilter and produce initialization routines.

    @param guide guided image (or array of images) with up to 3 channels, if it have more then 3
    channels then only first 3 channels will be used.

    @param radius radius of Guided Filter.

    @param eps regularization term of Guided Filter. \f${eps}^2\f$ is similar to the sigma in the color
    space into bilateralFilter.

    For more details about Guided Filter parameters, see the original article @cite Kaiming10 .
    """

def createQuaternionImage(img, qimg=...) -> qimg:
    """
    * @brief   creates a quaternion image.
    *
    * @param   img         Source 8-bit, 32-bit or 64-bit image, with 3-channel image.
    * @param   qimg        result CV_64FC4 a quaternion image( 4 chanels zero channel and B,G,R).
    """

def createRFFeatureGetter() -> retval:
    """
    .
    """

def createRICInterpolator() -> retval:
    """
    @brief Factory method that creates an instance of the
    RICInterpolator.
    """

def createRightMatcher(matcher_left) -> retval:
    """
    @brief Convenience method to set up the matcher for computing the right-view disparity map
    that is required in case of filtering with confidence.

    @param matcher_left main stereo matcher instance that will be used with the filter
    """

def createScanSegment(image_width, image_height, num_superpixels, slices=..., merge_small=...) -> retval:
    """
    @brief Initializes a ScanSegment object.

    The function initializes a ScanSegment object for the input image. It stores the parameters of
    the image: image_width and image_height. It also sets the parameters of the F-DBSCAN superpixel
    algorithm, which are: num_superpixels, threads, and merge_small.

    @param image_width Image width.
    @param image_height Image height.
    @param num_superpixels Desired number of superpixels. Note that the actual number may be smaller
    due to restrictions (depending on the image size). Use getNumberOfSuperpixels() to
    get the actual number.
    @param slices Number of processing threads for parallelisation. Setting -1 uses the maximum number
    of threads. In practice, four threads is enough for smaller images and eight threads for larger ones.
    @param merge_small merge small segments to give the desired number of superpixels. Processing is
    much faster without merging, but many small segments will be left in the image.
    """

def createStructuredEdgeDetection(model, howToGetFeatures=...) -> retval:
    """
    .
    """

def createSuperpixelLSC(image, region_size=..., ratio=...) -> retval:
    """
    @brief Class implementing the LSC (Linear Spectral Clustering) superpixels

    @param image Image to segment
    @param region_size Chooses an average superpixel size measured in pixels
    @param ratio Chooses the enforcement of superpixel compactness factor of superpixel

    The function initializes a SuperpixelLSC object for the input image. It sets the parameters of
    superpixel algorithm, which are: region_size and ruler. It preallocate some buffers for future
    computing iterations over the given image. An example of LSC is ilustrated in the following picture.
    For enanched results it is recommended for color images to preprocess image with little gaussian blur
    with a small 3 x 3 kernel and additional conversion into CieLAB color space.

    ![image](pics/superpixels_lsc.png)
    """

def createSuperpixelSEEDS(image_width, image_height, image_channels, num_superpixels, num_levels, prior=..., histogram_bins=..., double_step=...) -> retval:
    """
    @brief Initializes a SuperpixelSEEDS object.

    @param image_width Image width.
    @param image_height Image height.
    @param image_channels Number of channels of the image.
    @param num_superpixels Desired number of superpixels. Note that the actual number may be smaller
    due to restrictions (depending on the image size and num_levels). Use getNumberOfSuperpixels() to
    get the actual number.
    @param num_levels Number of block levels. The more levels, the more accurate is the segmentation,
    but needs more memory and CPU time.
    @param prior enable 3x3 shape smoothing term if \>0. A larger value leads to smoother shapes. prior
    must be in the range [0, 5].
    @param histogram_bins Number of histogram bins.
    @param double_step If true, iterate each block level twice for higher accuracy.

    The function initializes a SuperpixelSEEDS object for the input image. It stores the parameters of
    the image: image_width, image_height and image_channels. It also sets the parameters of the SEEDS
    superpixel algorithm, which are: num_superpixels, num_levels, use_prior, histogram_bins and
    double_step.

    The number of levels in num_levels defines the amount of block levels that the algorithm use in the
    optimization. The initialization is a grid, in which the superpixels are equally distributed through
    the width and the height of the image. The larger blocks correspond to the superpixel size, and the
    levels with smaller blocks are formed by dividing the larger blocks into 2 x 2 blocks of pixels,
    recursively until the smaller block level. An example of initialization of 4 block levels is
    illustrated in the following figure.

    ![image](pics/superpixels_blocks.png)
    """

def createSuperpixelSLIC(image, algorithm=..., region_size=..., ruler=...) -> retval:
    """
    @brief Initialize a SuperpixelSLIC object

    @param image Image to segment
    @param algorithm Chooses the algorithm variant to use:
    SLIC segments image using a desired region_size, and in addition SLICO will optimize using adaptive compactness factor,
    while MSLIC will optimize using manifold methods resulting in more content-sensitive superpixels.
    @param region_size Chooses an average superpixel size measured in pixels
    @param ruler Chooses the enforcement of superpixel smoothness factor of superpixel

    The function initializes a SuperpixelSLIC object for the input image. It sets the parameters of choosed
    superpixel algorithm, which are: region_size and ruler. It preallocate some buffers for future
    computing iterations over the given image. For enanched results it is recommended for color images to
    preprocess image with little gaussian blur using a small 3 x 3 kernel and additional conversion into
    CieLAB color space. An example of SLIC versus SLICO and MSLIC is ilustrated in the following picture.

    ![image](pics/superpixels_slic.png)
    """

def dtFilter(guide, src, sigmaSpatial, sigmaColor, dst=..., mode=..., numIters=...) -> dst:
    """
    @brief Simple one-line Domain Transform filter call. If you have multiple images to filter with the same
    guided image then use DTFilter interface to avoid extra computations on initialization stage.

    @param guide guided image (also called as joint image) with unsigned 8-bit or floating-point 32-bit
    depth and up to 4 channels.
    @param src filtering image with unsigned 8-bit or floating-point 32-bit depth and up to 4 channels.
    @param dst destination image
    @param sigmaSpatial \f${\sigma}_H\f$ parameter in the original article, it's similar to the sigma in the
    coordinate space into bilateralFilter.
    @param sigmaColor \f${\sigma}_r\f$ parameter in the original article, it's similar to the sigma in the
    color space into bilateralFilter.
    @param mode one form three modes DTF_NC, DTF_RF and DTF_IC which corresponds to three modes for
    filtering 2D signals in the article.
    @param numIters optional number of iterations used for filtering, 3 is quite enough.
    @sa bilateralFilter, guidedFilter, amFilter
    """

def edgePreservingFilter(src, d, threshold, dst=...) -> dst:
    """
    * @brief Smoothes an image using the Edge-Preserving filter.
        *
        * The function smoothes Gaussian noise as well as salt & pepper noise.
        * For more details about this implementation, please see
        * [ReiWoe18]  Reich, S. and W&#246;rg&#246;tter, F. and Dellen, B. (2018). A Real-Time Edge-Preserving Denoising Filter. Proceedings of the 13th International Joint Conference on Computer Vision, Imaging and Computer Graphics Theory and Applications (VISIGRAPP): Visapp, 85-94, 4. DOI: 10.5220/0006509000850094.
        *
        * @param src Source 8-bit 3-channel image.
        * @param dst Destination image of the same size and type as src.
        * @param d Diameter of each pixel neighborhood that is used during filtering. Must be greater or equal 3.
        * @param threshold Threshold, which distinguishes between noise, outliers, and data.
    """

def fastBilateralSolverFilter(guide, src, confidence, dst=..., sigma_spatial=..., sigma_luma=..., sigma_chroma=..., lambda_=..., num_iter=..., max_tol=...) -> dst:
    """
    @brief Simple one-line Fast Bilateral Solver filter call. If you have multiple images to filter with the same
    guide then use FastBilateralSolverFilter interface to avoid extra computations.

    @param guide image serving as guide for filtering. It should have 8-bit depth and either 1 or 3 channels.

    @param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point 32-bit depth and up to 4 channels.

    @param confidence confidence image with unsigned 8-bit or floating-point 32-bit confidence and 1 channel.

    @param dst destination image.

    @param sigma_spatial parameter, that is similar to spatial space sigma (bandwidth) in bilateralFilter.

    @param sigma_luma parameter, that is similar to luma space sigma (bandwidth) in bilateralFilter.

    @param sigma_chroma parameter, that is similar to chroma space sigma (bandwidth) in bilateralFilter.

    @param lambda smoothness strength parameter for solver.

    @param num_iter number of iterations used for solver, 25 is usually enough.

    @param max_tol convergence tolerance used for solver.

    For more details about the Fast Bilateral Solver parameters, see the original paper @cite BarronPoole2016.

    @note Confidence images with CV_8U depth are expected to in [0, 255] and CV_32F in [0, 1] range.
    """

def fastGlobalSmootherFilter(guide, src, lambda_, sigma_color, dst=..., lambda_attenuation=..., num_iter=...) -> dst:
    """
    @brief Simple one-line Fast Global Smoother filter call. If you have multiple images to filter with the same
    guide then use FastGlobalSmootherFilter interface to avoid extra computations.

    @param guide image serving as guide for filtering. It should have 8-bit depth and either 1 or 3 channels.

    @param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point 32-bit depth and up to 4 channels.

    @param dst destination image.

    @param lambda parameter defining the amount of regularization

    @param sigma_color parameter, that is similar to color space sigma in bilateralFilter.

    @param lambda_attenuation internal parameter, defining how much lambda decreases after each iteration. Normally,
    it should be 0.25. Setting it to 1.0 may lead to streaking artifacts.

    @param num_iter number of iterations used for filtering, 3 is usually enough.
    """

def findEllipses(image, ellipses=..., scoreThreshold=..., reliabilityThreshold=..., centerDistanceThreshold=...) -> ellipses:
    """
    @brief Finds ellipses fastly in an image using projective invariant pruning.
    *
    * The function detects ellipses in images using projective invariant pruning.
    * For more details about this implementation, please see @cite jia2017fast
    * Jia, Qi et al, (2017).
    * A Fast Ellipse Detector using Projective Invariant Pruning. IEEE Transactions on Image Processing.
    *
    @param image input image, could be gray or color.
    @param ellipses output vector of found ellipses. each vector is encoded as five float $x, y, a, b, radius, score$.
    @param scoreThreshold float, the threshold of ellipse score.
    @param reliabilityThreshold float, the threshold of reliability.
    @param centerDistanceThreshold float, the threshold of center distance.
    """

def fourierDescriptor(src, dst=..., nbElt=..., nbFD=...) -> dst:
    """
    * @brief   Fourier descriptors for planed closed curves
        *
        * For more details about this implementation, please see @cite PersoonFu1977
        *
        * @param   src   contour type vector<Point> , vector<Point2f>  or vector<Point2d>
        * @param   dst   Mat of type CV_64FC2 and nbElt rows A VERIFIER
        * @param   nbElt number of rows in dst or getOptimalDFTSize rows if nbElt=-1
        * @param   nbFD number of FD return in dst dst = [FD(1...nbFD/2) FD(nbFD/2-nbElt+1...:nbElt)]
        *
    """

def getDisparityVis(src, dst=..., scale=...) -> dst:
    """
    @brief Function for creating a disparity map visualization (clamped CV_8U image)

    @param src input disparity map (CV_16S depth)

    @param dst output visualization

    @param scale disparity map will be multiplied by this value for visualization
    """

def guidedFilter(guide, src, radius, eps, dst=..., dDepth=...) -> dst:
    """
    @brief Simple one-line Guided Filter call.

    If you have multiple images to filter with the same guided image then use GuidedFilter interface to
    avoid extra computations on initialization stage.

    @param guide guided image (or array of images) with up to 3 channels, if it have more then 3
    channels then only first 3 channels will be used.

    @param src filtering image with any numbers of channels.

    @param dst output image.

    @param radius radius of Guided Filter.

    @param eps regularization term of Guided Filter. \f${eps}^2\f$ is similar to the sigma in the color
    space into bilateralFilter.

    @param dDepth optional depth of the output image.

    @sa bilateralFilter, dtFilter, amFilter
    """

def jointBilateralFilter(joint, src, d, sigmaColor, sigmaSpace, dst=..., borderType=...) -> dst:
    """
    @brief Applies the joint bilateral filter to an image.

    @param joint Joint 8-bit or floating-point, 1-channel or 3-channel image.

    @param src Source 8-bit or floating-point, 1-channel or 3-channel image with the same depth as joint
    image.

    @param dst Destination image of the same size and type as src .

    @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
    it is computed from sigmaSpace .

    @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
    farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in
    larger areas of semi-equal color.

    @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
    farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).
    When d\>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is
    proportional to sigmaSpace .

    @param borderType

    @note bilateralFilter and jointBilateralFilter use L1 norm to compute difference between colors.

    @sa bilateralFilter, amFilter
    """

def l0Smooth(src, dst=..., lambda_=..., kappa=...) -> dst:
    """
    @brief Global image smoothing via L0 gradient minimization.

    @param src source image for filtering with unsigned 8-bit or signed 16-bit or floating-point depth.

    @param dst destination image.

    @param lambda parameter defining the smooth term weight.

    @param kappa parameter defining the increasing factor of the weight of the gradient data term.

    For more details about L0 Smoother, see the original paper @cite xu2011image.
    """

def niBlackThreshold(_src, maxValue, type, blockSize, k, _dst=..., binarizationMethod=..., r=...) -> _dst:
    """
    @brief Performs thresholding on input images using Niblack's technique or some of the
    popular variations it inspired.

    The function transforms a grayscale image to a binary image according to the formulae:
    -   **THRESH_BINARY**
        \f[dst(x,y) =  \fork{\texttt{maxValue}}{if \(src(x,y) > T(x,y)\)}{0}{otherwise}\f]
    -   **THRESH_BINARY_INV**
        \f[dst(x,y) =  \fork{0}{if \(src(x,y) > T(x,y)\)}{\texttt{maxValue}}{otherwise}\f]
    where \f$T(x,y)\f$ is a threshold calculated individually for each pixel.

    The threshold value \f$T(x, y)\f$ is determined based on the binarization method chosen. For
    classic Niblack, it is the mean minus \f$ k \f$ times standard deviation of
    \f$\texttt{blockSize} \times\texttt{blockSize}\f$ neighborhood of \f$(x, y)\f$.

    The function can't process the image in-place.

    @param _src Source 8-bit single-channel image.
    @param _dst Destination image of the same size and the same type as src.
    @param maxValue Non-zero value assigned to the pixels for which the condition is satisfied,
    used with the THRESH_BINARY and THRESH_BINARY_INV thresholding types.
    @param type Thresholding type, see cv::ThresholdTypes.
    @param blockSize Size of a pixel neighborhood that is used to calculate a threshold value
    for the pixel: 3, 5, 7, and so on.
    @param k The user-adjustable parameter used by Niblack and inspired techniques. For Niblack, this is
    normally a value between 0 and 1 that is multiplied with the standard deviation and subtracted from
    the mean.
    @param binarizationMethod Binarization method to use. By default, Niblack's technique is used.
    Other techniques can be specified, see cv::ximgproc::LocalBinarizationMethods.
    @param r The user-adjustable parameter used by Sauvola's technique. This is the dynamic range
    of standard deviation.
    @sa  threshold, adaptiveThreshold
    """

def qconj(qimg, qcimg=...) -> qcimg:
    """
    * @brief   calculates conjugate of a quaternion image.
    *
    * @param   qimg         quaternion image.
    * @param   qcimg        conjugate of qimg
    """

def qdft(img, flags, sideLeft, qimg=...) -> qimg:
    """
    * @brief    Performs a forward or inverse Discrete quaternion Fourier transform of a 2D quaternion array.
    *
    * @param   img        quaternion image.
    * @param   qimg       quaternion image in dual space.
    * @param   flags      quaternion image in dual space. only DFT_INVERSE flags is supported
    * @param   sideLeft   true the hypercomplex exponential is to be multiplied on the left (false on the right ).
    """

def qmultiply(src1, src2, dst=...) -> dst:
    """
    * @brief   Calculates the per-element quaternion product of two arrays
    *
    * @param   src1         quaternion image.
    * @param   src2         quaternion image.
    * @param   dst        product dst(I)=src1(I) . src2(I)
    """

def qunitary(qimg, qnimg=...) -> qnimg:
    """
    * @brief   divides each element by its modulus.
    *
    * @param   qimg         quaternion image.
    * @param   qnimg        conjugate of qimg
    """

def readGT(src_path, dst=...) -> tuple[retval, dst]:
    """
    @brief Function for reading ground truth disparity maps. Supports basic Middlebury
    and MPI-Sintel formats. Note that the resulting disparity map is scaled by 16.

    @param src_path path to the image, containing ground-truth disparity map

    @param dst output disparity map, CV_16S depth

    @result returns zero if successfully read the ground truth
    """

def rollingGuidanceFilter(src, dst=..., d=..., sigmaColor=..., sigmaSpace=..., numOfIter=..., borderType=...) -> dst:
    """
    @brief Applies the rolling guidance filter to an image.

    For more details, please see @cite zhang2014rolling

    @param src Source 8-bit or floating-point, 1-channel or 3-channel image.

    @param dst Destination image of the same size and type as src.

    @param d Diameter of each pixel neighborhood that is used during filtering. If it is non-positive,
    it is computed from sigmaSpace .

    @param sigmaColor Filter sigma in the color space. A larger value of the parameter means that
    farther colors within the pixel neighborhood (see sigmaSpace ) will be mixed together, resulting in
    larger areas of semi-equal color.

    @param sigmaSpace Filter sigma in the coordinate space. A larger value of the parameter means that
    farther pixels will influence each other as long as their colors are close enough (see sigmaColor ).
    When d\>0 , it specifies the neighborhood size regardless of sigmaSpace . Otherwise, d is
    proportional to sigmaSpace .

    @param numOfIter Number of iterations of joint edge-preserving filtering applied on the source image.

    @param borderType

    @note  rollingGuidanceFilter uses jointBilateralFilter as the edge-preserving filter.

    @sa jointBilateralFilter, bilateralFilter, amFilter
    """

def thinning(src, dst=..., thinningType=...) -> dst:
    """
    @brief Applies a binary blob thinning operation, to achieve a skeletization of the input image.

    The function transforms a binary blob image into a skeletized form using the technique of Zhang-Suen.

    @param src Source 8-bit single-channel image, containing binary blobs, with blobs having 255 pixel values.
    @param dst Destination image of the same size and the same type as src. The function can work in-place.
    @param thinningType Value that defines which thinning algorithm should be used. See cv::ximgproc::ThinningTypes
    """

def transformFD(src, t, dst=..., fdContour=...) -> dst:
    """
    * @brief   transform a contour
        *
        * @param   src   contour or Fourier Descriptors if fd is true
        * @param   t   transform Mat given by estimateTransformation
        * @param   dst   Mat of type CV_64FC2 and nbElt rows
        * @param   fdContour true src are Fourier Descriptors. fdContour false src is a contour
        *
    """

def weightedMedianFilter(joint, src, r, dst=..., sigma=..., weightType=..., mask=...) -> dst:
    """
    * @brief   Applies weighted median filter to an image.
    *
    * For more details about this implementation, please see @cite zhang2014100+
    *
    * @param   joint       Joint 8-bit, 1-channel or 3-channel image.
    * @param   src         Source 8-bit or floating-point, 1-channel or 3-channel image.
    * @param   dst         Destination image.
    * @param   r           Radius of filtering kernel, should be a positive integer.
    * @param   sigma       Filter range standard deviation for the joint image.
    * @param   weightType  weightType The type of weight definition, see WMFWeightType
    * @param   mask        A 0-1 mask that has the same size with I. This mask is used to ignore the effect of some pixels. If the pixel value on mask is 0,
    *                           the pixel will be ignored when maintaining the joint-histogram. This is useful for applications like optical flow occlusion handling.
    *
    * @sa medianBlur, jointBilateralFilter
    """

AM_FILTER: Final[int]
ARO_0_45: int
ARO_315_0: int
ARO_315_135: Final[int]
ARO_315_45: Final[int]
ARO_45_135: Final[int]
ARO_45_90: int
ARO_90_135: int
ARO_CTR_HOR: Final[int]
ARO_CTR_VER: Final[int]
BINARIZATION_NIBLACK: Final[int]
BINARIZATION_NICK: Final[int]
BINARIZATION_SAUVOLA: Final[int]
BINARIZATION_WOLF: Final[int]
DTF_IC: Final[int]
DTF_NC: Final[int]
DTF_RF: Final[int]
EDGE_DRAWING_LSD: Final[int]
EDGE_DRAWING_PREWITT: Final[int]
EDGE_DRAWING_SCHARR: Final[int]
EDGE_DRAWING_SOBEL: Final[int]
EdgeDrawing_LSD: Final[int]
EdgeDrawing_PREWITT: Final[int]
EdgeDrawing_SCHARR: Final[int]
EdgeDrawing_SOBEL: Final[int]
FHT_ADD: Final[int]
FHT_AVE: Final[int]
FHT_MAX: Final[int]
FHT_MIN: Final[int]
GUIDED_FILTER: Final[int]
HDO_DESKEW: Final[int]
HDO_RAW: Final[int]
MSLIC: Final[int]
SLIC: Final[int]
SLICO: Final[int]
THINNING_GUOHALL: Final[int]
THINNING_ZHANGSUEN: Final[int]
WMF_COS: Final[int]
WMF_EXP: Final[int]
WMF_IV1: Final[int]
WMF_IV2: Final[int]
WMF_JAC: Final[int]
WMF_OFF: Final[int]
