import builtins
from typing import Any, TypeAlias

contour: TypeAlias = Any

retval: TypeAlias = Any

class IntelligentScissorsMB(builtins.object):
    def applyImage(self, image) -> retval:
        """
        @brief Specify input image and extract image features
        *
        * @param image input image. Type is #CV_8UC1 / #CV_8UC3
        """

    def applyImageFeatures(self, non_edge, gradient_direction, gradient_magnitude, image=...) -> retval:
        """
        @brief Specify custom features of input image
        *
        * Customized advanced variant of applyImage() call.
        *
        * @param non_edge Specify cost of non-edge pixels. Type is CV_8UC1. Expected values are `{0, 1}`.
        * @param gradient_direction Specify gradient direction feature. Type is CV_32FC2. Values are expected to be normalized: `x^2 + y^2 == 1`
        * @param gradient_magnitude Specify cost of gradient magnitude function: Type is CV_32FC1. Values should be in range `[0, 1]`.
        * @param image **Optional parameter**. Must be specified if subset of features is specified (non-specified features are calculated internally)
        """

    def buildMap(self, sourcePt) -> None:
        """
        @brief Prepares a map of optimal paths for the given source point on the image
        *
        * @note applyImage() / applyImageFeatures() must be called before this call
        *
        * @param sourcePt The source point used to find the paths
        """

    def getContour(self, targetPt, contour=..., backward=...) -> contour:
        """
        @brief Extracts optimal contour for the given target point on the image
        *
        * @note buildMap() must be called before this call
        *
        * @param targetPt The target point
        * @param[out] contour The list of pixels which contains optimal path between the source and the target points of the image. Type is CV_32SC2 (compatible with `std::vector<Point>`)
        * @param backward Flag to indicate reverse order of retrived pixels (use "true" value to fetch points from the target to the source point)
        """

    def setEdgeFeatureCannyParameters(self, threshold1, threshold2, apertureSize=..., L2gradient=...) -> retval:
        """
        @brief Switch edge feature extractor to use Canny edge detector
        *
        * @note "Laplacian Zero-Crossing" feature extractor is used by default (following to original article)
        *
        * @sa Canny
        """

    def setEdgeFeatureZeroCrossingParameters(self, gradient_magnitude_min_value=...) -> retval:
        """
        @brief Switch to "Laplacian Zero-Crossing" edge feature extractor and specify its parameters
        *
        * This feature extractor is used by default according to article.
        *
        * Implementation has additional filtering for regions with low-amplitude noise.
        * This filtering is enabled through parameter of minimal gradient amplitude (use some small value 4, 8, 16).
        *
        * @note Current implementation of this feature extractor is based on processing of grayscale images (color image is converted to grayscale image first).
        *
        * @note Canny edge detector is a bit slower, but provides better results (especially on color images): use setEdgeFeatureCannyParameters().
        *
        * @param gradient_magnitude_min_value Minimal gradient magnitude value for edge pixels (default: 0, check is disabled)
        """

    def setGradientMagnitudeMaxLimit(self, gradient_magnitude_threshold_max=...) -> retval:
        """
        @brief Specify gradient magnitude max value threshold
        *
        * Zero limit value is used to disable gradient magnitude thresholding (default behavior, as described in original article).
        * Otherwize pixels with `gradient magnitude >= threshold` have zero cost.
        *
        * @note Thresholding should be used for images with irregular regions (to avoid stuck on parameters from high-contract areas, like embedded logos).
        *
        * @param gradient_magnitude_threshold_max Specify gradient magnitude max value threshold (default: 0, disabled)
        """

    def setWeights(self, weight_non_edge, weight_gradient_direction, weight_gradient_magnitude) -> retval:
        """
        @brief Specify weights of feature functions
        *
        * Consider keeping weights normalized (sum of weights equals to 1.0)
        * Discrete dynamic programming (DP) goal is minimization of costs between pixels.
        *
        * @param weight_non_edge Specify cost of non-edge pixels (default: 0.43f)
        * @param weight_gradient_direction Specify cost of gradient direction function (default: 0.43f)
        * @param weight_gradient_magnitude Specify cost of gradient magnitude function (default: 0.14f)
        """
