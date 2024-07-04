from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

result: TypeAlias = Any
signature: TypeAlias = Any
matchesGMS: TypeAlias = Any
retval: TypeAlias = Any

class AffineFeature2D(cv2.Feature2D): ...

class BEBLID(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def setScaleFactor(self, scale_factor) -> None:
        """"""

    def create(self, scale_factor, n_bits=...) -> retval:
        """
        @brief Creates the BEBLID descriptor.
        @param scale_factor Adjust the sampling window around detected keypoints: - <b> 1.00f </b> should be the scale for ORB keypoints - <b> 6.75f </b> should be the scale for SIFT detected keypoints - <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints - <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
        @param n_bits Determine the number of bits in the descriptor. Should be either BEBLID::SIZE_512_BITS or BEBLID::SIZE_256_BITS.
        """

class BoostDesc(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def getUseScaleOrientation(self) -> retval:
        """"""

    def setScaleFactor(self, scale_factor) -> None:
        """"""

    def setUseScaleOrientation(self, use_scale_orientation) -> None:
        """"""

    def create(self, desc=..., use_scale_orientation=..., scale_factor=...) -> retval:
        """"""

class BriefDescriptorExtractor(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getDescriptorSize(self) -> retval:
        """"""

    def getUseOrientation(self) -> retval:
        """"""

    def setDescriptorSize(self, bytes) -> None:
        """"""

    def setUseOrientation(self, use_orientation) -> None:
        """"""

    def create(self, bytes=..., use_orientation=...) -> retval:
        """"""

class DAISY(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getH(self) -> retval:
        """"""

    def getInterpolation(self) -> retval:
        """"""

    def getNorm(self) -> retval:
        """"""

    def getQHist(self) -> retval:
        """"""

    def getQRadius(self) -> retval:
        """"""

    def getQTheta(self) -> retval:
        """"""

    def getRadius(self) -> retval:
        """"""

    def getUseOrientation(self) -> retval:
        """"""

    def setH(self, H) -> None:
        """"""

    def setInterpolation(self, interpolation) -> None:
        """"""

    def setNorm(self, norm) -> None:
        """"""

    def setQHist(self, q_hist) -> None:
        """"""

    def setQRadius(self, q_radius) -> None:
        """"""

    def setQTheta(self, q_theta) -> None:
        """"""

    def setRadius(self, radius) -> None:
        """"""

    def setUseOrientation(self, use_orientation) -> None:
        """"""

    def create(self, radius=..., q_radius=..., q_theta=..., q_hist=..., norm=..., H=..., interpolation=..., use_orientation=...) -> retval:
        """"""

class FREAK(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getOrientationNormalized(self) -> retval:
        """"""

    def getPatternScale(self) -> retval:
        """"""

    def getScaleNormalized(self) -> retval:
        """"""

    def setNOctaves(self, nOctaves) -> None:
        """"""

    def setOrientationNormalized(self, orientationNormalized) -> None:
        """"""

    def setPatternScale(self, patternScale) -> None:
        """"""

    def setScaleNormalized(self, scaleNormalized) -> None:
        """"""

    def create(self, orientationNormalized=..., scaleNormalized=..., patternScale=..., nOctaves=..., selectedPairs=...) -> retval:
        """
        @param orientationNormalized Enable orientation normalization.
        @param scaleNormalized Enable scale normalization.
        @param patternScale Scaling of the description pattern.
        @param nOctaves Number of octaves covered by the detected keypoints.
        @param selectedPairs (Optional) user defined selected pairs indexes,
        """

class HarrisLaplaceFeatureDetector(cv2.Feature2D):
    def getCornThresh(self) -> retval:
        """"""

    def getDOGThresh(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getMaxCorners(self) -> retval:
        """"""

    def getNumLayers(self) -> retval:
        """"""

    def getNumOctaves(self) -> retval:
        """"""

    def setCornThresh(self, corn_thresh_) -> None:
        """"""

    def setDOGThresh(self, DOG_thresh_) -> None:
        """"""

    def setMaxCorners(self, maxCorners_) -> None:
        """"""

    def setNumLayers(self, num_layers_) -> None:
        """"""

    def setNumOctaves(self, numOctaves_) -> None:
        """"""

    def create(self, numOctaves=..., corn_thresh=..., DOG_thresh=..., maxCorners=..., num_layers=...) -> retval:
        """
        * @brief Creates a new implementation instance.
        *
        * @param numOctaves the number of octaves in the scale-space pyramid
        * @param corn_thresh the threshold for the Harris cornerness measure
        * @param DOG_thresh the threshold for the Difference-of-Gaussians scale selection
        * @param maxCorners the maximum number of corners to consider
        * @param num_layers the number of intermediate scales per octave
        """

class LATCH(cv2.Feature2D):
    def getBytes(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getHalfSSDsize(self) -> retval:
        """"""

    def getRotationInvariance(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def setBytes(self, bytes) -> None:
        """"""

    def setHalfSSDsize(self, half_ssd_size) -> None:
        """"""

    def setRotationInvariance(self, rotationInvariance) -> None:
        """"""

    def setSigma(self, sigma) -> None:
        """"""

    def create(self, bytes=..., rotationInvariance=..., half_ssd_size=..., sigma=...) -> retval:
        """"""

class LUCID(cv2.Feature2D):
    def getBlurKernel(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getLucidKernel(self) -> retval:
        """"""

    def setBlurKernel(self, blur_kernel) -> None:
        """"""

    def setLucidKernel(self, lucid_kernel) -> None:
        """"""

    def create(self, lucid_kernel=..., blur_kernel=...) -> retval:
        """
        * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
        * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
        """

class MSDDetector(cv2.Feature2D):
    def getComputeOrientation(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getKNN(self) -> retval:
        """"""

    def getNScales(self) -> retval:
        """"""

    def getNmsRadius(self) -> retval:
        """"""

    def getNmsScaleRadius(self) -> retval:
        """"""

    def getPatchRadius(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def getSearchAreaRadius(self) -> retval:
        """"""

    def getThSaliency(self) -> retval:
        """"""

    def setComputeOrientation(self, compute_orientation) -> None:
        """"""

    def setKNN(self, kNN) -> None:
        """"""

    def setNScales(self, use_orientation) -> None:
        """"""

    def setNmsRadius(self, nms_radius) -> None:
        """"""

    def setNmsScaleRadius(self, nms_scale_radius) -> None:
        """"""

    def setPatchRadius(self, patch_radius) -> None:
        """"""

    def setScaleFactor(self, scale_factor) -> None:
        """"""

    def setSearchAreaRadius(self, use_orientation) -> None:
        """"""

    def setThSaliency(self, th_saliency) -> None:
        """"""

    def create(self, m_patch_radius=..., m_search_area_radius=..., m_nms_radius=..., m_nms_scale_radius=..., m_th_saliency=..., m_kNN=..., m_scale_factor=..., m_n_scales=..., m_compute_orientation=...) -> retval:
        """"""

class PCTSignatures(cv2.Algorithm):
    def computeSignature(self, image, signature=...) -> signature:
        """
        * @brief Computes signature of given image.
        * @param image Input image of CV_8U type.
        * @param signature Output computed signature.
        """

    def computeSignatures(self, images, signatures) -> None:
        """
        * @brief Computes signatures for multiple images in parallel.
        * @param images Vector of input images of CV_8U type.
        * @param signatures Vector of computed signatures.
        """

    def getClusterMinSize(self) -> retval:
        """
        * @brief This parameter multiplied by the index of iteration gives lower limit for cluster size.
        *       Clusters containing fewer points than specified by the limit have their centroid dismissed
        *       and points are reassigned.
        """

    def getDistanceFunction(self) -> retval:
        """
        * @brief Distance function selector used for measuring distance between two points in k-means.
        """

    def getDropThreshold(self) -> retval:
        """
        * @brief Remove centroids in k-means whose weight is lesser or equal to given threshold.
        """

    def getGrayscaleBits(self) -> retval:
        """
        * @brief Color resolution of the greyscale bitmap represented in allocated bits
        *       (i.e., value 4 means that 16 shades of grey are used).
        *       The greyscale bitmap is used for computing contrast and entropy values.
        """

    def getInitSeedCount(self) -> retval:
        """
        * @brief Number of initial seeds (initial number of clusters) for the k-means algorithm.
        """

    def getInitSeedIndexes(self) -> retval:
        """
        * @brief Initial seeds (initial number of clusters) for the k-means algorithm.
        """

    def getIterationCount(self) -> retval:
        """
        * @brief Number of iterations of the k-means clustering.
        *       We use fixed number of iterations, since the modified clustering is pruning clusters
        *       (not iteratively refining k clusters).
        """

    def getJoiningDistance(self) -> retval:
        """
        * @brief Threshold euclidean distance between two centroids.
        *       If two cluster centers are closer than this distance,
        *       one of the centroid is dismissed and points are reassigned.
        """

    def getMaxClustersCount(self) -> retval:
        """
        * @brief Maximal number of generated clusters. If the number is exceeded,
        *       the clusters are sorted by their weights and the smallest clusters are cropped.
        """

    def getSampleCount(self) -> retval:
        """
        * @brief Number of initial samples taken from the image.
        """

    def getSamplingPoints(self) -> retval:
        """
        * @brief Initial samples taken from the image.
        *       These sampled features become the input for clustering.
        """

    def getWeightA(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightB(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightContrast(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightEntropy(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightL(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightX(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWeightY(self) -> retval:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def getWindowRadius(self) -> retval:
        """
        * @brief Size of the texture sampling window used to compute contrast and entropy
        *       (center of the window is always in the pixel selected by x,y coordinates
        *       of the corresponding feature sample).
        """

    def setClusterMinSize(self, clusterMinSize) -> None:
        """
        * @brief This parameter multiplied by the index of iteration gives lower limit for cluster size.
        *       Clusters containing fewer points than specified by the limit have their centroid dismissed
        *       and points are reassigned.
        """

    def setDistanceFunction(self, distanceFunction) -> None:
        """
        * @brief Distance function selector used for measuring distance between two points in k-means.
        *       Available: L0_25, L0_5, L1, L2, L2SQUARED, L5, L_INFINITY.
        """

    def setDropThreshold(self, dropThreshold) -> None:
        """
        * @brief Remove centroids in k-means whose weight is lesser or equal to given threshold.
        """

    def setGrayscaleBits(self, grayscaleBits) -> None:
        """
        * @brief Color resolution of the greyscale bitmap represented in allocated bits
        *       (i.e., value 4 means that 16 shades of grey are used).
        *       The greyscale bitmap is used for computing contrast and entropy values.
        """

    def setInitSeedIndexes(self, initSeedIndexes) -> None:
        """
        * @brief Initial seed indexes for the k-means algorithm.
        """

    def setIterationCount(self, iterationCount) -> None:
        """
        * @brief Number of iterations of the k-means clustering.
        *       We use fixed number of iterations, since the modified clustering is pruning clusters
        *       (not iteratively refining k clusters).
        """

    def setJoiningDistance(self, joiningDistance) -> None:
        """
        * @brief Threshold euclidean distance between two centroids.
        *       If two cluster centers are closer than this distance,
        *       one of the centroid is dismissed and points are reassigned.
        """

    def setMaxClustersCount(self, maxClustersCount) -> None:
        """
        * @brief Maximal number of generated clusters. If the number is exceeded,
        *       the clusters are sorted by their weights and the smallest clusters are cropped.
        """

    def setSamplingPoints(self, samplingPoints) -> None:
        """
        * @brief Sets sampling points used to sample the input image.
        * @param samplingPoints Vector of sampling points in range [0..1) * @note Number of sampling points must be greater or equal to clusterization seed count.
        """

    def setTranslation(self, idx, value) -> None:
        """
        * @brief Translations of the individual axes of the feature space.
        * @param idx ID of the translation
        * @param value Value of the translation * @note *       WEIGHT_IDX = 0; *       X_IDX = 1; *       Y_IDX = 2; *       L_IDX = 3; *       A_IDX = 4; *       B_IDX = 5; *       CONTRAST_IDX = 6; *       ENTROPY_IDX = 7;
        """

    def setTranslations(self, translations) -> None:
        """
        * @brief Translations of the individual axes of the feature space.
        * @param translations Values of all translations. * @note *       WEIGHT_IDX = 0; *       X_IDX = 1; *       Y_IDX = 2; *       L_IDX = 3; *       A_IDX = 4; *       B_IDX = 5; *       CONTRAST_IDX = 6; *       ENTROPY_IDX = 7;
        """

    def setWeight(self, idx, value) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space.
        * @param idx ID of the weight
        * @param value Value of the weight * @note *       WEIGHT_IDX = 0; *       X_IDX = 1; *       Y_IDX = 2; *       L_IDX = 3; *       A_IDX = 4; *       B_IDX = 5; *       CONTRAST_IDX = 6; *       ENTROPY_IDX = 7;
        """

    def setWeightA(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightB(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightContrast(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightEntropy(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightL(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightX(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeightY(self, weight) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space
        *       (x,y = position; L,a,b = color in CIE Lab space; c = contrast. e = entropy)
        """

    def setWeights(self, weights) -> None:
        """
        * @brief Weights (multiplicative constants) that linearly stretch individual axes of the feature space.
        * @param weights Values of all weights. * @note *       WEIGHT_IDX = 0; *       X_IDX = 1; *       Y_IDX = 2; *       L_IDX = 3; *       A_IDX = 4; *       B_IDX = 5; *       CONTRAST_IDX = 6; *       ENTROPY_IDX = 7;
        """

    def setWindowRadius(self, radius) -> None:
        """
        * @brief Size of the texture sampling window used to compute contrast and entropy
        *       (center of the window is always in the pixel selected by x,y coordinates
        *       of the corresponding feature sample).
        """

    @overload
    def create(self, initSampleCount=..., initSeedCount=..., pointDistribution=...) -> retval:
        """
        * @brief Creates PCTSignatures algorithm using sample and seed count.
        *       It generates its own sets of sampling points and clusterization seed indexes.
        * @param initSampleCount Number of points used for image sampling.
        * @param initSeedCount Number of initial clusterization seeds. *       Must be lower or equal to initSampleCount
        * @param pointDistribution Distribution of generated points. Default: UNIFORM. *       Available: UNIFORM, REGULAR, NORMAL. * @return Created algorithm.
        """

    @overload
    def create(self, initSamplingPoints, initSeedCount) -> retval:
        """
        * @brief Creates PCTSignatures algorithm using pre-generated sampling points
        *       and number of clusterization seeds. It uses the provided
        *       sampling points and generates its own clusterization seed indexes.
        * @param initSamplingPoints Sampling points used in image sampling.
        * @param initSeedCount Number of initial clusterization seeds. *       Must be lower or equal to initSamplingPoints.size(). * @return Created algorithm.
        """

    def create(self, initSamplingPoints, initClusterSeedIndexes) -> retval:
        """
        * @brief Creates PCTSignatures algorithm using pre-generated sampling points
        *       and clusterization seeds indexes.
        * @param initSamplingPoints Sampling points used in image sampling.
        * @param initClusterSeedIndexes Indexes of initial clusterization seeds. *       Its size must be lower or equal to initSamplingPoints.size(). * @return Created algorithm.
        """

    def drawSignature(self, source, signature, result=..., radiusToShorterSideRatio=..., borderThickness=...) -> result:
        """
        * @brief Draws signature in the source image and outputs the result.
        *       Signatures are visualized as a circle
        *       with radius based on signature weight
        *       and color based on signature color.
        *       Contrast and entropy are not visualized.
        * @param source Source image.
        * @param signature Image signature.
        * @param result Output result.
        * @param radiusToShorterSideRatio Determines maximal radius of signature in the output image.
        * @param borderThickness Border thickness of the visualized signature.
        """

    def generateInitPoints(self, initPoints, count, pointDistribution) -> None:
        """
        * @brief Generates initial sampling points according to selected point distribution.
        * @param initPoints Output vector where the generated points will be saved.
        * @param count Number of points to generate.
        * @param pointDistribution Point distribution selector. *       Available: UNIFORM, REGULAR, NORMAL. * @note Generated coordinates are in range [0..1)
        """

class PCTSignaturesSQFD(cv2.Algorithm):
    def computeQuadraticFormDistance(self, _signature0, _signature1) -> retval:
        """
        * @brief Computes Signature Quadratic Form Distance of two signatures.
        * @param _signature0 The first signature.
        * @param _signature1 The second signature.
        """

    def computeQuadraticFormDistances(self, sourceSignature, imageSignatures, distances) -> None:
        """
        * @brief Computes Signature Quadratic Form Distance between the reference signature
        *       and each of the other image signatures.
        * @param sourceSignature The signature to measure distance of other signatures from.
        * @param imageSignatures Vector of signatures to measure distance from the source signature.
        * @param distances Output vector of measured distances.
        """

    def create(self, distanceFunction=..., similarityFunction=..., similarityParameter=...) -> retval:
        """
        * @brief Creates the algorithm instance using selected distance function,
        *       similarity function and similarity function parameter.
        * @param distanceFunction Distance function selector. Default: L2 *       Available: L0_25, L0_5, L1, L2, L2SQUARED, L5, L_INFINITY
        * @param similarityFunction Similarity function selector. Default: HEURISTIC *       Available: MINUS, GAUSSIAN, HEURISTIC
        * @param similarityParameter Parameter of the similarity function.
        """

class SURF(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getExtended(self) -> retval:
        """"""

    def getHessianThreshold(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getUpright(self) -> retval:
        """"""

    def setExtended(self, extended) -> None:
        """"""

    def setHessianThreshold(self, hessianThreshold) -> None:
        """"""

    def setNOctaveLayers(self, nOctaveLayers) -> None:
        """"""

    def setNOctaves(self, nOctaves) -> None:
        """"""

    def setUpright(self, upright) -> None:
        """"""

    def create(self, hessianThreshold=..., nOctaves=..., nOctaveLayers=..., extended=..., upright=...) -> retval:
        """
        @param hessianThreshold Threshold for hessian keypoint detector used in SURF.
        @param nOctaves Number of pyramid octaves the keypoint detector will use.
        @param nOctaveLayers Number of octave layers within each octave.
        @param extended Extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors).
        @param upright Up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation).
        """

class StarDetector(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getLineThresholdBinarized(self) -> retval:
        """"""

    def getLineThresholdProjected(self) -> retval:
        """"""

    def getMaxSize(self) -> retval:
        """"""

    def getResponseThreshold(self) -> retval:
        """"""

    def getSuppressNonmaxSize(self) -> retval:
        """"""

    def setLineThresholdBinarized(self, _lineThresholdBinarized) -> None:
        """"""

    def setLineThresholdProjected(self, _lineThresholdProjected) -> None:
        """"""

    def setMaxSize(self, _maxSize) -> None:
        """"""

    def setResponseThreshold(self, _responseThreshold) -> None:
        """"""

    def setSuppressNonmaxSize(self, _suppressNonmaxSize) -> None:
        """"""

    def create(self, maxSize=..., responseThreshold=..., lineThresholdProjected=..., lineThresholdBinarized=..., suppressNonmaxSize=...) -> retval:
        """"""

class TBMR(AffineFeature2D):
    def getMaxAreaRelative(self) -> retval:
        """"""

    def getMinArea(self) -> retval:
        """"""

    def getNScales(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def setMaxAreaRelative(self, maxArea) -> None:
        """"""

    def setMinArea(self, minArea) -> None:
        """"""

    def setNScales(self, n_scales) -> None:
        """"""

    def setScaleFactor(self, scale_factor) -> None:
        """"""

    def create(self, min_area=..., max_area_relative=..., scale_factor=..., n_scales=...) -> retval:
        """"""

class TEBLID(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def create(self, scale_factor, n_bits=...) -> retval:
        """
        @brief Creates the TEBLID descriptor.
        @param scale_factor Adjust the sampling window around detected keypoints: - <b> 1.00f </b> should be the scale for ORB keypoints - <b> 6.75f </b> should be the scale for SIFT detected keypoints - <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints - <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
        @param n_bits Determine the number of bits in the descriptor. Should be either TEBLID::SIZE_256_BITS or TEBLID::SIZE_512_BITS.
        """

class VGG(cv2.Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def getUseNormalizeDescriptor(self) -> retval:
        """"""

    def getUseNormalizeImage(self) -> retval:
        """"""

    def getUseScaleOrientation(self) -> retval:
        """"""

    def setScaleFactor(self, scale_factor) -> None:
        """"""

    def setSigma(self, isigma) -> None:
        """"""

    def setUseNormalizeDescriptor(self, dsc_normalize) -> None:
        """"""

    def setUseNormalizeImage(self, img_normalize) -> None:
        """"""

    def setUseScaleOrientation(self, use_scale_orientation) -> None:
        """"""

    def create(self, desc=..., isigma=..., img_normalize=..., use_scale_orientation=..., scale_factor=..., dsc_normalize=...) -> retval:
        """"""

def BEBLID_create(scale_factor, n_bits=...) -> retval:
    """
    @brief Creates the BEBLID descriptor.
        @param scale_factor Adjust the sampling window around detected keypoints:
        - <b> 1.00f </b> should be the scale for ORB keypoints
        - <b> 6.75f </b> should be the scale for SIFT detected keypoints
        - <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints
        - <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
        @param n_bits Determine the number of bits in the descriptor. Should be either
         BEBLID::SIZE_512_BITS or BEBLID::SIZE_256_BITS.
    """

def BoostDesc_create(desc=..., use_scale_orientation=..., scale_factor=...) -> retval:
    """
    .
    """

def BriefDescriptorExtractor_create(bytes=..., use_orientation=...) -> retval:
    """
    .
    """

def DAISY_create(radius=..., q_radius=..., q_theta=..., q_hist=..., norm=..., H=..., interpolation=..., use_orientation=...) -> retval:
    """
    .
    """

def FREAK_create(orientationNormalized=..., scaleNormalized=..., patternScale=..., nOctaves=..., selectedPairs=...) -> retval:
    """
    @param orientationNormalized Enable orientation normalization.
        @param scaleNormalized Enable scale normalization.
        @param patternScale Scaling of the description pattern.
        @param nOctaves Number of octaves covered by the detected keypoints.
        @param selectedPairs (Optional) user defined selected pairs indexes,
    """

def HarrisLaplaceFeatureDetector_create(numOctaves=..., corn_thresh=..., DOG_thresh=..., maxCorners=..., num_layers=...) -> retval:
    """
    * @brief Creates a new implementation instance.
         *
         * @param numOctaves the number of octaves in the scale-space pyramid
         * @param corn_thresh the threshold for the Harris cornerness measure
         * @param DOG_thresh the threshold for the Difference-of-Gaussians scale selection
         * @param maxCorners the maximum number of corners to consider
         * @param num_layers the number of intermediate scales per octave
    """

def LATCH_create(bytes=..., rotationInvariance=..., half_ssd_size=..., sigma=...) -> retval:
    """
    .
    """

def LUCID_create(lucid_kernel=..., blur_kernel=...) -> retval:
    """
    * @param lucid_kernel kernel for descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
         * @param blur_kernel kernel for blurring image prior to descriptor construction, where 1=3x3, 2=5x5, 3=7x7 and so forth
    """

def MSDDetector_create(m_patch_radius=..., m_search_area_radius=..., m_nms_radius=..., m_nms_scale_radius=..., m_th_saliency=..., m_kNN=..., m_scale_factor=..., m_n_scales=..., m_compute_orientation=...) -> retval:
    """
    .
    """

def PCTSignaturesSQFD_create(distanceFunction=..., similarityFunction=..., similarityParameter=...) -> retval:
    """
    * @brief Creates the algorithm instance using selected distance function,
        *       similarity function and similarity function parameter.
        * @param distanceFunction Distance function selector. Default: L2
        *       Available: L0_25, L0_5, L1, L2, L2SQUARED, L5, L_INFINITY
        * @param similarityFunction Similarity function selector. Default: HEURISTIC
        *       Available: MINUS, GAUSSIAN, HEURISTIC
        * @param similarityParameter Parameter of the similarity function.
    """

@overload
def PCTSignatures_create(initSampleCount=..., initSeedCount=..., pointDistribution=...) -> retval:
    """
    * @brief Creates PCTSignatures algorithm using sample and seed count.
        *       It generates its own sets of sampling points and clusterization seed indexes.
        * @param initSampleCount Number of points used for image sampling.
        * @param initSeedCount Number of initial clusterization seeds.
        *       Must be lower or equal to initSampleCount
        * @param pointDistribution Distribution of generated points. Default: UNIFORM.
        *       Available: UNIFORM, REGULAR, NORMAL.
        * @return Created algorithm.
    """

@overload
def PCTSignatures_create(initSampleCount=..., initSeedCount=..., pointDistribution=...) -> retval:
    """
    * @brief Creates PCTSignatures algorithm using pre-generated sampling points
        *       and number of clusterization seeds. It uses the provided
        *       sampling points and generates its own clusterization seed indexes.
        * @param initSamplingPoints Sampling points used in image sampling.
        * @param initSeedCount Number of initial clusterization seeds.
        *       Must be lower or equal to initSamplingPoints.size().
        * @return Created algorithm.
    """

@overload
def PCTSignatures_create(initSampleCount=..., initSeedCount=..., pointDistribution=...) -> retval:
    """
    * @brief Creates PCTSignatures algorithm using pre-generated sampling points
        *       and clusterization seeds indexes.
        * @param initSamplingPoints Sampling points used in image sampling.
        * @param initClusterSeedIndexes Indexes of initial clusterization seeds.
        *       Its size must be lower or equal to initSamplingPoints.size().
        * @return Created algorithm.
    """

def PCTSignatures_drawSignature(source, signature, result=..., radiusToShorterSideRatio=..., borderThickness=...) -> result:
    """
    * @brief Draws signature in the source image and outputs the result.
        *       Signatures are visualized as a circle
        *       with radius based on signature weight
        *       and color based on signature color.
        *       Contrast and entropy are not visualized.
        * @param source Source image.
        * @param signature Image signature.
        * @param result Output result.
        * @param radiusToShorterSideRatio Determines maximal radius of signature in the output image.
        * @param borderThickness Border thickness of the visualized signature.
    """

def PCTSignatures_generateInitPoints(initPoints, count, pointDistribution) -> None:
    """
    * @brief Generates initial sampling points according to selected point distribution.
        * @param initPoints Output vector where the generated points will be saved.
        * @param count Number of points to generate.
        * @param pointDistribution Point distribution selector.
        *       Available: UNIFORM, REGULAR, NORMAL.
        * @note Generated coordinates are in range [0..1)
    """

def SIFT_create(nfeatures=..., nOctaveLayers=..., contrastThreshold=..., edgeThreshold=..., sigma=...) -> retval:
    """
    Use cv.SIFT_create() instead
    """

@overload
def SURF_create(hessianThreshold=..., nOctaves=..., nOctaveLayers=..., extended=..., upright=...) -> retval:
    """
    @param hessianThreshold Threshold for hessian keypoint detector used in SURF.
        @param nOctaves Number of pyramid octaves the keypoint detector will use.
        @param nOctaveLayers Number of octave layers within each octave.
        @param extended Extended descriptor flag (true - use extended 128-element descriptors; false - use
        64-element descriptors).
        @param upright Up-right or rotated features flag (true - do not compute orientation of features;
    """

@overload
def SURF_create(hessianThreshold=..., nOctaves=..., nOctaveLayers=..., extended=..., upright=...) -> retval:
    """ """

def StarDetector_create(maxSize=..., responseThreshold=..., lineThresholdProjected=..., lineThresholdBinarized=..., suppressNonmaxSize=...) -> retval:
    """
    .
    """

def TBMR_create(min_area=..., max_area_relative=..., scale_factor=..., n_scales=...) -> retval:
    """
    .
    """

def TEBLID_create(scale_factor, n_bits=...) -> retval:
    """
    @brief Creates the TEBLID descriptor.
        @param scale_factor Adjust the sampling window around detected keypoints:
        - <b> 1.00f </b> should be the scale for ORB keypoints
        - <b> 6.75f </b> should be the scale for SIFT detected keypoints
        - <b> 6.25f </b> is default and fits for KAZE, SURF detected keypoints
        - <b> 5.00f </b> should be the scale for AKAZE, MSD, AGAST, FAST, BRISK keypoints
        @param n_bits Determine the number of bits in the descriptor. Should be either
         TEBLID::SIZE_256_BITS or TEBLID::SIZE_512_BITS.
    """

def VGG_create(desc=..., isigma=..., img_normalize=..., use_scale_orientation=..., scale_factor=..., dsc_normalize=...) -> retval:
    """
    .
    """

def matchGMS(size1, size2, keypoints1, keypoints2, matches1to2, withRotation=..., withScale=..., thresholdFactor=...) -> matchesGMS:
    """
    @brief GMS (Grid-based Motion Statistics) feature matching strategy described in @cite Bian2017gms .
        @param size1 Input size of image1.
        @param size2 Input size of image2.
        @param keypoints1 Input keypoints of image1.
        @param keypoints2 Input keypoints of image2.
        @param matches1to2 Input 1-nearest neighbor matches.
        @param matchesGMS Matches returned by the GMS matching strategy.
        @param withRotation Take rotation transformation into account.
        @param withScale Take scale transformation into account.
        @param thresholdFactor The higher, the less matches.
        @note
            Since GMS works well when the number of features is large, we recommend to use the ORB feature and set FastThreshold to 0 to get as many as possible features quickly.
            If matching results are not satisfying, please add more features. (We use 10000 for images with 640 X 480).
            If your images have big rotation and scale changes, please set withRotation or withScale to true.
    """

def matchLOGOS(keypoints1, keypoints2, nn1, nn2, matches1to2) -> None:
    """
    @brief LOGOS (Local geometric support for high-outlier spatial verification) feature matching strategy described in @cite Lowry2018LOGOSLG .
        @param keypoints1 Input keypoints of image1.
        @param keypoints2 Input keypoints of image2.
        @param nn1 Index to the closest BoW centroid for each descriptors of image1.
        @param nn2 Index to the closest BoW centroid for each descriptors of image2.
        @param matches1to2 Matches returned by the LOGOS matching strategy.
        @note
            This matching strategy is suitable for features matching against large scale database.
            First step consists in constructing the bag-of-words (BoW) from a representative image database.
            Image descriptors are then represented by their closest codevector (nearest BoW centroid).
    """

BEBLID_SIZE_256_BITS: Final[int]
BEBLID_SIZE_512_BITS: Final[int]
DAISY_NRM_FULL: Final[int]
DAISY_NRM_NONE: Final[int]
DAISY_NRM_PARTIAL: Final[int]
DAISY_NRM_SIFT: Final[int]
PCTSIGNATURES_GAUSSIAN: Final[int]
PCTSIGNATURES_HEURISTIC: Final[int]
PCTSIGNATURES_L0_25: int
PCTSIGNATURES_L0_5: int
PCTSIGNATURES_L1: Final[int]
PCTSIGNATURES_L2: Final[int]
PCTSIGNATURES_L2SQUARED: Final[int]
PCTSIGNATURES_L5: Final[int]
PCTSIGNATURES_L_INFINITY: Final[int]
PCTSIGNATURES_MINUS: Final[int]
PCTSIGNATURES_NORMAL: Final[int]
PCTSIGNATURES_REGULAR: Final[int]
PCTSIGNATURES_UNIFORM: Final[int]
PCTSignatures_GAUSSIAN: Final[int]
PCTSignatures_HEURISTIC: Final[int]
PCTSignatures_L0_25: int
PCTSignatures_L0_5: int
PCTSignatures_L1: Final[int]
PCTSignatures_L2: Final[int]
PCTSignatures_L2SQUARED: Final[int]
PCTSignatures_L5: Final[int]
PCTSignatures_L_INFINITY: Final[int]
PCTSignatures_MINUS: Final[int]
PCTSignatures_NORMAL: Final[int]
PCTSignatures_REGULAR: Final[int]
PCTSignatures_UNIFORM: Final[int]
TEBLID_SIZE_256_BITS: Final[int]
TEBLID_SIZE_512_BITS: Final[int]
