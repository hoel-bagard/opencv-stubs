import builtins
from abc import ABC, abstractmethod
from typing import Any, overload, TypeAlias

import numpy as np
import numpy.typing as npt
from typing_extensions import Self

from .modules import *

qrcode: TypeAlias = Any
foundLocations: TypeAlias = Any
disparity: TypeAlias = Any
qrcodes: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any
numDetections: TypeAlias = Any
straight_qrcode: TypeAlias = Any
nearestPt: TypeAlias = Any
msers: TypeAlias = Any
votes: TypeAlias = Any
phase: TypeAlias = Any
bboxes: TypeAlias = Any
ymap: TypeAlias = Any
flow_v: TypeAlias = Any
edge: TypeAlias = Any
fgmask: TypeAlias = Any
face_feature: TypeAlias = Any
angleOfs: TypeAlias = Any
leadingEdgeList: TypeAlias = Any
nfa: TypeAlias = Any
imgDescriptor: TypeAlias = Any
numpy: TypeAlias = Any
descriptors: TypeAlias = Any
objects: TypeAlias = Any
eb: TypeAlias = Any
large: TypeAlias = Any
levelWeights: TypeAlias = Any
structured: TypeAlias = Any
positions: TypeAlias = Any
prec: TypeAlias = Any
boundingBox: TypeAlias = Any
rejectLevels: TypeAlias = Any
orgpt: TypeAlias = Any
foundWeights: TypeAlias = Any
points2f: TypeAlias = Any
nextPts: TypeAlias = Any
image1: TypeAlias = Any
weights: TypeAlias = Any
facetList: TypeAlias = Any
flow_u: TypeAlias = Any
flow: TypeAlias = Any
output: TypeAlias = Any
Params: TypeAlias = Any
costMatrix: TypeAlias = Any
triangleList: TypeAlias = Any
status: TypeAlias = Any
wechat: TypeAlias = Any
edgeList: TypeAlias = Any
firstEdge: TypeAlias = Any
keypoints: TypeAlias = Any
aligned_img: TypeAlias = Any
vertex: TypeAlias = Any
backgroundImage: TypeAlias = Any
width: TypeAlias = Any
readyIndex: TypeAlias = Any
facetCenters: TypeAlias = Any
faces: TypeAlias = Any
tb: TypeAlias = Any
line: TypeAlias = Any
img: TypeAlias = Any
matches: TypeAlias = Any
err: TypeAlias = Any
lines: TypeAlias = Any
xmap: TypeAlias = Any
dstpt: TypeAlias = Any
pano: TypeAlias = Any
decoded_info: TypeAlias = Any
grad: TypeAlias = Any
image2: TypeAlias = Any
colored: TypeAlias = Any
ppf: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any
InputArrayOfArrays: TypeAlias = npt.NDArray[Any]

class AKAZE(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getDescriptorChannels(self) -> retval:
        """"""

    def getDescriptorSize(self) -> retval:
        """"""

    def getDescriptorType(self) -> retval:
        """"""

    def getDiffusivity(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setDescriptorChannels(self, dch) -> None:
        """"""

    def setDescriptorSize(self, dsize) -> None:
        """"""

    def setDescriptorType(self, dtype) -> None:
        """"""

    def setDiffusivity(self, diff) -> None:
        """"""

    def setNOctaveLayers(self, octaveLayers) -> None:
        """"""

    def setNOctaves(self, octaves) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def create(self, descriptor_type=..., descriptor_size=..., descriptor_channels=..., threshold=..., nOctaves=..., nOctaveLayers=..., diffusivity=...) -> retval:
        """
        @brief The AKAZE constructor

        @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        @param descriptor_size Size of the descriptor in bits. 0 -\> Full size
        @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
        @param threshold Detector response threshold to accept point
        @param nOctaves Maximum octave evolution of the image
        @param nOctaveLayers Default number of sublevels per scale level
        @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        """

class AffineFeature(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getViewParams(self, tilts, rolls) -> None:
        """"""

    def setViewParams(self, tilts, rolls) -> None:
        """"""

    def create(self, backend, maxTilt=..., minTilt=..., tiltStep=..., rotateStepBase=...) -> retval:
        """
        @param backend The detector/extractor you want to use as backend.
        @param maxTilt The highest power index of tilt factor. 5 is used in the paper as tilt sampling range n.
        @param minTilt The lowest power index of tilt factor. 0 is used in the paper.
        @param tiltStep Tilt sampling step \f$\delta_t\f$ in Algorithm 1 in the paper.
        @param rotateStepBase Rotation sampling step factor b in Algorithm 1 in the paper.
        """

class AffineTransformer(ShapeTransformer):
    def getFullAffine(self) -> retval:
        """"""

    def setFullAffine(self, fullAffine) -> None:
        """"""

class AgastFeatureDetector(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getNonmaxSuppression(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getType(self) -> retval:
        """"""

    def setNonmaxSuppression(self, f) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setType(self, type) -> None:
        """"""

    def create(self, threshold=..., nonmaxSuppression=..., type=...) -> retval:
        """"""

class Algorithm(builtins.object):
    @abstractmethod
    def clear(self) -> None:
        """
        @brief Clears the algorithm state
        """

    @abstractmethod
    def empty(self) -> bool:
        """
        @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
        """

    @abstractmethod
    def getDefaultName(self) -> str:
        """
        Returns the algorithm string identifier.
        This string is used as top level xml/yml node tag when the object is saved to a file or string.
        """

    def read(self, fn: FileNode) -> None:
        """
        @brief Reads algorithm parameters from a file storage
        """

    @abstractmethod
    def save(self, filename: str) -> None:
        """
        Saves the algorithm to a file.
        In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs).
        """

    @overload
    def write(self, fs: FileStorage) -> None:
        """
        @brief Stores algorithm parameters in a file storage
        """

    @overload
    def write(self, fs: FileStorage, name: str) -> None:
        """"""

class AlignExposures(Algorithm):
    def process(self, src, dst, times, response) -> None:
        """
        @brief Aligns images

        @param src vector of input images
        @param dst vector of aligned images
        @param times vector of exposure time values for each image
        @param response 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.
        """

class AlignMTB(AlignExposures):
    def calculateShift(self, img0, img1) -> retval:
        """
        @brief Calculates shift between two images, i. e. how to shift the second image to correspond it with the
        first.

        @param img0 first image
        @param img1 second image
        """

    def computeBitmaps(self, img, tb=..., eb=...) -> tuple[tb, eb]:
        """
        @brief Computes median threshold and exclude bitmaps of given image.

        @param img input image
        @param tb median threshold bitmap
        @param eb exclude bitmap
        """

    def getCut(self) -> retval:
        """"""

    def getExcludeRange(self) -> retval:
        """"""

    def getMaxBits(self) -> retval:
        """"""

    @overload
    def process(self, src, dst, times, response) -> None:
        """"""

    @overload
    def process(self, src, dst) -> None:
        """
        @brief Short version of process, that doesn't take extra arguments.

        @param src vector of input images
        @param dst vector of aligned images
        """

    def setCut(self, value) -> None:
        """"""

    def setExcludeRange(self, exclude_range) -> None:
        """"""

    def setMaxBits(self, max_bits) -> None:
        """"""

    def shiftMat(self, src, shift, dst=...) -> dst:
        """
        @brief Helper function, that shift Mat filling new regions with zeros.

        @param src input image
        @param dst result image
        @param shift shift value
        """

class AsyncArray(builtins.object):
    @overload
    def get(self, dst=...) -> dst:
        """
        Fetch the result.
        @param[out] dst destination array  Waits for result until container has valid result. Throws exception if exception was stored as a result.  Throws exception on invalid container state.  @note Result or stored exception can be fetched only once.
        """

    @overload
    def get(self, timeoutNs, dst=...) -> tuple[retval, dst]:
        """
        Retrieving the result with timeout
        @param[out] dst destination array
        @param[in] timeoutNs timeout in nanoseconds, -1 for infinite wait  @returns true if result is ready, false if the timeout has expired  @note Result or stored exception can be fetched only once.
        """

    def release(self) -> None:
        """"""

    def valid(self) -> retval:
        """"""

    def wait_for(self, timeoutNs) -> retval:
        """"""

class BFMatcher(DescriptorMatcher):
    def create(self, normType=..., crossCheck=...) -> retval:
        """
        @brief Brute-force matcher create method.
        @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description).
        @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
        """

class BOWImgDescriptorExtractor(builtins.object):
    def compute(self, image, keypoints, imgDescriptor=...) -> imgDescriptor:
        """
        @overload
        @param keypointDescriptors Computed descriptors to match with vocabulary.
        @param imgDescriptor Computed output image descriptor.
        @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary) returned if it is non-zero.
        """

    def descriptorSize(self) -> retval:
        """
        @brief Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
        """

    def descriptorType(self) -> retval:
        """
        @brief Returns an image descriptor type.
        """

    def getVocabulary(self) -> retval:
        """
        @brief Returns the set vocabulary.
        """

    def setVocabulary(self, vocabulary) -> None:
        """
        @brief Sets a visual vocabulary.

        @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the vocabulary is a visual word (cluster center).
        """

class BOWKMeansTrainer(BOWTrainer):
    @overload
    def cluster(self) -> retval:
        """"""

    @overload
    def cluster(self, descriptors) -> retval:
        """"""

class BOWTrainer(builtins.object):
    def add(self, descriptors) -> None:
        """
        @brief Adds descriptors to a training set.

        @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a descriptor.  The training set is clustered using clustermethod to construct the vocabulary.
        """

    def clear(self) -> None:
        """"""

    @overload
    def cluster(self) -> retval:
        """
        @overload
        """

    @overload
    def cluster(self, descriptors) -> retval:
        """
        @brief Clusters train descriptors.

        @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor. Descriptors are not added to the inner train descriptor set.  The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered.
        """

    def descriptorsCount(self) -> retval:
        """
        @brief Returns the count of all descriptors stored in the training set.
        """

    def getDescriptors(self) -> retval:
        """
        @brief Returns a training set of descriptors.
        """

class BRISK(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getOctaves(self) -> retval:
        """"""

    def getPatternScale(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setOctaves(self, octaves) -> None:
        """
        @brief Set detection octaves.
        @param octaves detection octaves. Use 0 to do single scale.
        """

    def setPatternScale(self, patternScale) -> None:
        """
        @brief Set detection patternScale.
        @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        """

    def setThreshold(self, threshold) -> None:
        """
        @brief Set detection threshold.
        @param threshold AGAST detection threshold score.
        """

    @overload
    def create(self, thresh=..., octaves=..., patternScale=...) -> retval:
        """
        @brief The BRISK constructor

        @param thresh AGAST detection threshold score.
        @param octaves detection octaves. Use 0 to do single scale.
        @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        """

    @overload
    def create(self, radiusList, numberList, dMax=..., dMin=..., indexChange=...) -> retval:
        """
        @brief The BRISK constructor for a custom pattern

        @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).
        @param numberList defines the number of sampling points on the sampling circle. Must be the same size as radiusList..
        @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).
        @param dMin threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).
        @param indexChange index remapping of the bits.
        """

    @overload
    def create(self, thresh, octaves, radiusList, numberList, dMax=..., dMin=..., indexChange=...) -> retval:
        """
        @brief The BRISK constructor for a custom pattern, detection threshold and octaves

        @param thresh AGAST detection threshold score.
        @param octaves detection octaves. Use 0 to do single scale.
        @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).
        @param numberList defines the number of sampling points on the sampling circle. Must be the same size as radiusList..
        @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).
        @param dMin threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).
        @param indexChange index remapping of the bits.
        """

class BackgroundSubtractor(Algorithm):
    def apply(self, image, fgmask=..., learningRate=...) -> fgmask:
        """
        @brief Computes a foreground mask.

        @param image Next video frame.
        @param fgmask The output foreground mask as an 8-bit binary image.
        @param learningRate The value between 0 and 1 that indicates how fast the background model is learnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
        """

    def getBackgroundImage(self, backgroundImage=...) -> backgroundImage:
        """
        @brief Computes a background image.

        @param backgroundImage The output background image.  @note Sometimes the background image can be very blurry, as it contain the average background statistics.
        """

class BackgroundSubtractorKNN(BackgroundSubtractor):
    def getDetectShadows(self) -> retval:
        """
        @brief Returns the shadow detection flag

        If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorKNN for
        details.
        """

    def getDist2Threshold(self) -> retval:
        """
        @brief Returns the threshold on the squared distance between the pixel and the sample

        The threshold on the squared distance between the pixel and the sample to decide whether a pixel is
        close to a data sample.
        """

    def getHistory(self) -> retval:
        """
        @brief Returns the number of last frames that affect the background model
        """

    def getNSamples(self) -> retval:
        """
        @brief Returns the number of data samples in the background model
        """

    def getShadowThreshold(self) -> retval:
        """
        @brief Returns the shadow threshold

        A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
        the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
        is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara,
        *Detecting Moving Shadows...*, IEEE PAMI,2003.
        """

    def getShadowValue(self) -> retval:
        """
        @brief Returns the shadow value

        Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
        in the mask always means background, 255 means foreground.
        """

    def getkNNSamples(self) -> retval:
        """
        @brief Returns the number of neighbours, the k in the kNN.

        K is the number of samples that need to be within dist2Threshold in order to decide that that
        pixel is matching the kNN background model.
        """

    def setDetectShadows(self, detectShadows) -> None:
        """
        @brief Enables or disables shadow detection
        """

    def setDist2Threshold(self, _dist2Threshold) -> None:
        """
        @brief Sets the threshold on the squared distance
        """

    def setHistory(self, history) -> None:
        """
        @brief Sets the number of last frames that affect the background model
        """

    def setNSamples(self, _nN) -> None:
        """
        @brief Sets the number of data samples in the background model.

        The model needs to be reinitalized to reserve memory.
        """

    def setShadowThreshold(self, threshold) -> None:
        """
        @brief Sets the shadow threshold
        """

    def setShadowValue(self, value) -> None:
        """
        @brief Sets the shadow value
        """

    def setkNNSamples(self, _nkNN) -> None:
        """
        @brief Sets the k in the kNN. How many nearest neighbours need to match.
        """

class BackgroundSubtractorMOG2(BackgroundSubtractor):
    def apply(self, image, fgmask=..., learningRate=...) -> fgmask:
        """
        @brief Computes a foreground mask.

        @param image Next video frame. Floating point frame will be used without scaling and should be in range \f$[0,255]\f$.
        @param fgmask The output foreground mask as an 8-bit binary image.
        @param learningRate The value between 0 and 1 that indicates how fast the background model is learnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
        """

    def getBackgroundRatio(self) -> retval:
        """
        @brief Returns the "background ratio" parameter of the algorithm

        If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's
        considered background and added to the model as a center of a new component. It corresponds to TB
        parameter in the paper.
        """

    def getComplexityReductionThreshold(self) -> retval:
        """
        @brief Returns the complexity reduction threshold

        This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05
        is a default value for all the samples. By setting CT=0 you get an algorithm very similar to the
        standard Stauffer&Grimson algorithm.
        """

    def getDetectShadows(self) -> retval:
        """
        @brief Returns the shadow detection flag

        If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for
        details.
        """

    def getHistory(self) -> retval:
        """
        @brief Returns the number of last frames that affect the background model
        """

    def getNMixtures(self) -> retval:
        """
        @brief Returns the number of gaussian components in the background model
        """

    def getShadowThreshold(self) -> retval:
        """
        @brief Returns the shadow threshold

        A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
        the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
        is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara,
        *Detecting Moving Shadows...*, IEEE PAMI,2003.
        """

    def getShadowValue(self) -> retval:
        """
        @brief Returns the shadow value

        Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
        in the mask always means background, 255 means foreground.
        """

    def getVarInit(self) -> retval:
        """
        @brief Returns the initial variance of each gaussian component
        """

    def getVarMax(self) -> retval:
        """"""

    def getVarMin(self) -> retval:
        """"""

    def getVarThreshold(self) -> retval:
        """
        @brief Returns the variance threshold for the pixel-model match

        The main threshold on the squared Mahalanobis distance to decide if the sample is well described by
        the background model or not. Related to Cthr from the paper.
        """

    def getVarThresholdGen(self) -> retval:
        """
        @brief Returns the variance threshold for the pixel-model match used for new mixture component generation

        Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the
        existing components (corresponds to Tg in the paper). If a pixel is not close to any component, it
        is considered foreground or added as a new component. 3 sigma =\> Tg=3\*3=9 is default. A smaller Tg
        value generates more components. A higher Tg value may result in a small number of components but
        they can grow too large.
        """

    def setBackgroundRatio(self, ratio) -> None:
        """
        @brief Sets the "background ratio" parameter of the algorithm
        """

    def setComplexityReductionThreshold(self, ct) -> None:
        """
        @brief Sets the complexity reduction threshold
        """

    def setDetectShadows(self, detectShadows) -> None:
        """
        @brief Enables or disables shadow detection
        """

    def setHistory(self, history) -> None:
        """
        @brief Sets the number of last frames that affect the background model
        """

    def setNMixtures(self, nmixtures) -> None:
        """
        @brief Sets the number of gaussian components in the background model.

        The model needs to be reinitalized to reserve memory.
        """

    def setShadowThreshold(self, threshold) -> None:
        """
        @brief Sets the shadow threshold
        """

    def setShadowValue(self, value) -> None:
        """
        @brief Sets the shadow value
        """

    def setVarInit(self, varInit) -> None:
        """
        @brief Sets the initial variance of each gaussian component
        """

    def setVarMax(self, varMax) -> None:
        """"""

    def setVarMin(self, varMin) -> None:
        """"""

    def setVarThreshold(self, varThreshold) -> None:
        """
        @brief Sets the variance threshold for the pixel-model match
        """

    def setVarThresholdGen(self, varThresholdGen) -> None:
        """
        @brief Sets the variance threshold for the pixel-model match used for new mixture component generation
        """

class BaseCascadeClassifier(Algorithm): ...

class CLAHE(Algorithm):
    def apply(self, src, dst=...) -> dst:
        """
        @brief Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.

        @param src Source image of type CV_8UC1 or CV_16UC1.
        @param dst Destination image.
        """

    def collectGarbage(self) -> None:
        """"""

    def getClipLimit(self) -> retval:
        """"""

    def getTilesGridSize(self) -> retval:
        """"""

    def setClipLimit(self, clipLimit) -> None:
        """
        @brief Sets threshold for contrast limiting.

        @param clipLimit threshold value.
        """

    def setTilesGridSize(self, tileGridSize) -> None:
        """
        @brief Sets size of grid for histogram equalization. Input image will be divided into
        equally sized rectangular tiles.

        @param tileGridSize defines the number of tiles in row and column.
        """

class CalibrateCRF(Algorithm):
    def process(self, src, times, dst=...) -> dst:
        """
        @brief Recovers inverse camera response.

        @param src vector of input images
        @param dst 256x1 matrix with inverse camera response function
        @param times vector of exposure time values for each image
        """

class CalibrateDebevec(CalibrateCRF):
    def getLambda(self) -> retval:
        """"""

    def getRandom(self) -> retval:
        """"""

    def getSamples(self) -> retval:
        """"""

    def setLambda(self, lambda_) -> None:
        """"""

    def setRandom(self, random) -> None:
        """"""

    def setSamples(self, samples) -> None:
        """"""

class CalibrateRobertson(CalibrateCRF):
    def getMaxIter(self) -> retval:
        """"""

    def getRadiance(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setMaxIter(self, max_iter) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

class CascadeClassifier(builtins.object):
    def detectMultiScale(self, image, scaleFactor=..., minNeighbors=..., flags=..., minSize=..., maxSize=...) -> objects:
        """
        @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
        of rectangles.

        @param image Matrix of the type CV_8U containing an image where objects are detected.
        @param objects Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
        @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        @param flags Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
        """

    def detectMultiScale2(self, image, scaleFactor=..., minNeighbors=..., flags=..., minSize=..., maxSize=...) -> tuple[objects, numDetections]:
        """
        @overload
        @param image Matrix of the type CV_8U containing an image where objects are detected.
        @param objects Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
        @param numDetections Vector of detection numbers for the corresponding objects. An object's number of detections is the number of neighboring positively classified rectangles that were joined together to form the object.
        @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        @param flags Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
        """

    def detectMultiScale3(self, image, scaleFactor=..., minNeighbors=..., flags=..., minSize=..., maxSize=..., outputRejectLevels=...) -> tuple[objects, rejectLevels, levelWeights]:
        """
        @overload
        This function allows you to retrieve the final stage decision certainty of classification.
        For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
        For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
        This value can then be used to separate strong from weaker classifications.

        A code sample on how to use it efficiently can be found below:
        @code
        Mat img;
        vector<double> weights;
        vector<int> levels;
        vector<Rect> detections;
        CascadeClassifier model("/path/to/your/model.xml");
        model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
        cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
        @endcode
        """

    def empty(self) -> retval:
        """
        @brief Checks whether the classifier has been loaded.
        """

    def getFeatureType(self) -> retval:
        """"""

    def getOriginalWindowSize(self) -> retval:
        """"""

    def isOldFormatCascade(self) -> retval:
        """"""

    def load(self, filename) -> retval:
        """
        @brief Loads a classifier from a file.

        @param filename Name of the file from which the classifier is loaded. The file may contain an old HAAR classifier trained by the haartraining application or a new cascade classifier trained by the traincascade application.
        """

    def read(self, node) -> retval:
        """
        @brief Reads a classifier from a FileStorage node.

        @note The file may contain a new cascade classifier (trained by the traincascade application) only.
        """

    def convert(self, oldcascade, newcascade) -> retval:
        """"""

class ChiHistogramCostExtractor(HistogramCostExtractor): ...
class CirclesGridFinderParameters(builtins.object): ...

class DISOpticalFlow(DenseOpticalFlow):
    def getFinestScale(self) -> retval:
        """
        @brief Finest level of the Gaussian pyramid on which the flow is computed (zero level
        corresponds to the original image resolution). The final flow is obtained by bilinear upscaling.
        @see setFinestScale
        """

    def getGradientDescentIterations(self) -> retval:
        """
        @brief Maximum number of gradient descent iterations in the patch inverse search stage. Higher values
        may improve quality in some cases.
        @see setGradientDescentIterations
        """

    def getPatchSize(self) -> retval:
        """
        @brief Size of an image patch for matching (in pixels). Normally, default 8x8 patches work well
        enough in most cases.
        @see setPatchSize
        """

    def getPatchStride(self) -> retval:
        """
        @brief Stride between neighbor patches. Must be less than patch size. Lower values correspond
        to higher flow quality.
        @see setPatchStride
        """

    def getUseMeanNormalization(self) -> retval:
        """
        @brief Whether to use mean-normalization of patches when computing patch distance. It is turned on
        by default as it typically provides a noticeable quality boost because of increased robustness to
        illumination variations. Turn it off if you are certain that your sequence doesn't contain any changes
        in illumination.
        @see setUseMeanNormalization
        """

    def getUseSpatialPropagation(self) -> retval:
        """
        @brief Whether to use spatial propagation of good optical flow vectors. This option is turned on by
        default, as it tends to work better on average and can sometimes help recover from major errors
        introduced by the coarse-to-fine scheme employed by the DIS optical flow algorithm. Turning this
        option off can make the output flow field a bit smoother, however.
        @see setUseSpatialPropagation
        """

    def getVariationalRefinementAlpha(self) -> retval:
        """
        @brief Weight of the smoothness term
        @see setVariationalRefinementAlpha
        """

    def getVariationalRefinementDelta(self) -> retval:
        """
        @brief Weight of the color constancy term
        @see setVariationalRefinementDelta
        """

    def getVariationalRefinementGamma(self) -> retval:
        """
        @brief Weight of the gradient constancy term
        @see setVariationalRefinementGamma
        """

    def getVariationalRefinementIterations(self) -> retval:
        """
        @brief Number of fixed point iterations of variational refinement per scale. Set to zero to
        disable variational refinement completely. Higher values will typically result in more smooth and
        high-quality flow.
        @see setGradientDescentIterations
        """

    def setFinestScale(self, val) -> None:
        """
        @copybrief getFinestScale @see getFinestScale
        """

    def setGradientDescentIterations(self, val) -> None:
        """
        @copybrief getGradientDescentIterations @see getGradientDescentIterations
        """

    def setPatchSize(self, val) -> None:
        """
        @copybrief getPatchSize @see getPatchSize
        """

    def setPatchStride(self, val) -> None:
        """
        @copybrief getPatchStride @see getPatchStride
        """

    def setUseMeanNormalization(self, val) -> None:
        """
        @copybrief getUseMeanNormalization @see getUseMeanNormalization
        """

    def setUseSpatialPropagation(self, val) -> None:
        """
        @copybrief getUseSpatialPropagation @see getUseSpatialPropagation
        """

    def setVariationalRefinementAlpha(self, val) -> None:
        """
        @copybrief getVariationalRefinementAlpha @see getVariationalRefinementAlpha
        """

    def setVariationalRefinementDelta(self, val) -> None:
        """
        @copybrief getVariationalRefinementDelta @see getVariationalRefinementDelta
        """

    def setVariationalRefinementGamma(self, val) -> None:
        """
        @copybrief getVariationalRefinementGamma @see getVariationalRefinementGamma
        """

    def setVariationalRefinementIterations(self, val) -> None:
        """
        @copybrief getGradientDescentIterations @see getGradientDescentIterations
        """

    def create(self, preset=...) -> retval:
        """
        @brief Creates an instance of DISOpticalFlow

        @param preset one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
        """

class DMatch(builtins.object): ...

class DenseOpticalFlow(Algorithm):
    def calc(self, I0, I1, flow) -> flow:
        """
        @brief Calculates an optical flow.

        @param I0 first 8-bit single-channel input image.
        @param I1 second input image of the same size and the same type as prev.
        @param flow computed flow image that has the same size as prev and type CV_32FC2.
        """

    def collectGarbage(self) -> None:
        """
        @brief Releases all inner buffers.
        """

class DescriptorMatcher(Algorithm, ABC):
    def add(self, descriptors: InputArrayOfArrays) -> None:
        """
        @brief Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
        collection.

        If the collection is not empty, the new descriptors are added to existing train descriptors.

        @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same train image.
        """

    def clear(self) -> None:
        """
        @brief Clears the train descriptor collections.
        """

    def clone(self, emptyTrainData=...) -> retval:
        """
        @brief Clones the matcher.

        @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object, that is, copies both parameters and train data. If emptyTrainData is true, the method creates an object copy with the current parameters but with empty train data.
        """

    def empty(self) -> retval:
        """
        @brief Returns true if there are no train descriptors in the both collections.
        """

    def getTrainDescriptors(self) -> retval:
        """
        @brief Returns a constant link to the train descriptor collection trainDescCollection .
        """

    def isMaskSupported(self) -> retval:
        """
        @brief Returns true if the descriptor matcher supports masking permissible matches.
        """

    @overload
    def knnMatch(self, queryDescriptors, trainDescriptors, k, mask=..., compactResult=...) -> matches:
        """
        @brief Finds the k best matches for each descriptor from a query set.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.
        @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
        @param k Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.  These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors.
        """

    @overload
    def knnMatch(self, queryDescriptors, k, masks=..., compactResult=...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
        @param k Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        """

    @overload
    def match(self, queryDescriptors, trainDescriptors, mask=...) -> matches:
        """
        @brief Finds the best match for each descriptor from a query set.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param matches Matches. If a query descriptor is masked out in mask , no match is added for this descriptor. So, matches size may be smaller than the query descriptors count.
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.  In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero.
        """

    @overload
    def match(self, queryDescriptors, masks=...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Matches. If a query descriptor is masked out in mask , no match is added for this descriptor. So, matches size may be smaller than the query descriptors count.
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        """

    @overload
    def radiusMatch(self, queryDescriptors, trainDescriptors, maxDistance, mask=..., compactResult=...) -> matches:
        """
        @brief For each query descriptor, finds the training descriptors not farther than the specified distance.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param matches Found matches.
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        @param maxDistance Threshold for the distance between matched descriptors. Distance means here metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.  For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order.
        """

    @overload
    def radiusMatch(self, queryDescriptors, maxDistance, masks=..., compactResult=...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Found matches.
        @param maxDistance Threshold for the distance between matched descriptors. Distance means here metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        """

    @overload
    def read(self, fileName) -> None:
        """"""

    @overload
    def read(self, arg1) -> None:
        """"""

    def train(self) -> None:
        """
        @brief Trains a descriptor matcher

        Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
        train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
        have an empty implementation of this method. Other matchers really train their inner structures (for
        example, FlannBasedMatcher trains flann::Index ).
        """

    @overload
    def write(self, fileName) -> None:
        """"""

    @overload
    def write(self, fs, name) -> None:
        """"""

    @overload
    def create(self, descriptorMatcherType) -> retval:
        """
        @brief Creates a descriptor matcher of a given type with the default parameters (using default
        constructor).

        @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are supported: -   `BruteForce` (it uses L2 ) -   `BruteForce-L1` -   `BruteForce-Hamming` -   `BruteForce-Hamming(2)` -   `FlannBased`
        """

    @overload
    def create(self, matcherType) -> retval:
        """"""

class EMDHistogramCostExtractor(HistogramCostExtractor):
    def getNormFlag(self) -> retval:
        """"""

    def setNormFlag(self, flag) -> None:
        """"""

class EMDL1HistogramCostExtractor(HistogramCostExtractor): ...

class FaceDetectorYN(builtins.object):
    def detect(self, image, faces=...) -> tuple[retval, faces]:
        """
        @brief A simple interface to detect face from given image
        *
        *  @param image an image to detect
        *  @param faces detection results stored in a cv::Mat
        """

    def getInputSize(self) -> retval:
        """"""

    def getNMSThreshold(self) -> retval:
        """"""

    def getScoreThreshold(self) -> retval:
        """"""

    def getTopK(self) -> retval:
        """"""

    def setInputSize(self, input_size) -> None:
        """
        @brief Set the size for the network input, which overwrites the input size of creating model. Call this method when the size of input image does not match the input size when creating model
        *
        * @param input_size the size of the input image
        """

    def setNMSThreshold(self, nms_threshold) -> None:
        """
        @brief Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value
        *
        * @param nms_threshold threshold for NMS operation
        """

    def setScoreThreshold(self, score_threshold) -> None:
        """
        @brief Set the score threshold to filter out bounding boxes of score less than the given value
        *
        * @param score_threshold threshold for filtering out bounding boxes
        """

    def setTopK(self, top_k) -> None:
        """
        @brief Set the number of bounding boxes preserved before NMS
        *
        * @param top_k the number of bounding boxes to preserve from top rank based on score
        """

    def create(self, model, config, input_size, score_threshold=..., nms_threshold=..., top_k=..., backend_id=..., target_id=...) -> retval:
        """
        @brief Creates an instance of this class with given parameters
        *
        *  @param model the path to the requested model
        *  @param config the path to the config file for compability, which is not requested for ONNX models
        *  @param input_size the size of the input image
        *  @param score_threshold the threshold to filter out bounding boxes of score smaller than the given value
        *  @param nms_threshold the threshold to suppress bounding boxes of IoU bigger than the given value
        *  @param top_k keep top K bboxes before NMS
        *  @param backend_id the id of backend
        *  @param target_id the id of target device
        """

class FaceRecognizerSF(builtins.object):
    def alignCrop(self, src_img, face_box, aligned_img=...) -> aligned_img:
        """
        @brief Aligning image to put face on the standard position
        *  @param src_img input image
        *  @param face_box the detection result used for indicate face in input image
        *  @param aligned_img output aligned image
        """

    def feature(self, aligned_img, face_feature=...) -> face_feature:
        """
        @brief Extracting face feature from aligned image
        *  @param aligned_img input aligned image
        *  @param face_feature output face feature
        """

    def match(self, face_feature1, face_feature2, dis_type=...) -> retval:
        """
        @brief Calculating the distance between two face features
        *  @param face_feature1 the first input feature
        *  @param face_feature2 the second input feature of the same size and the same type as face_feature1
        *  @param dis_type defining the similarity with optional values "FR_OSINE" or "FR_NORM_L2"
        """

    def create(self, model, config, backend_id=..., target_id=...) -> retval:
        """
        @brief Creates an instance of this class with given parameters
        *  @param model the path of the onnx model used for face recognition
        *  @param config the path to the config file for compability, which is not requested for ONNX models
        *  @param backend_id the id of backend
        *  @param target_id the id of target device
        """

class FarnebackOpticalFlow(DenseOpticalFlow):
    def getFastPyramids(self) -> retval:
        """"""

    def getFlags(self) -> retval:
        """"""

    def getNumIters(self) -> retval:
        """"""

    def getNumLevels(self) -> retval:
        """"""

    def getPolyN(self) -> retval:
        """"""

    def getPolySigma(self) -> retval:
        """"""

    def getPyrScale(self) -> retval:
        """"""

    def getWinSize(self) -> retval:
        """"""

    def setFastPyramids(self, fastPyramids) -> None:
        """"""

    def setFlags(self, flags) -> None:
        """"""

    def setNumIters(self, numIters) -> None:
        """"""

    def setNumLevels(self, numLevels) -> None:
        """"""

    def setPolyN(self, polyN) -> None:
        """"""

    def setPolySigma(self, polySigma) -> None:
        """"""

    def setPyrScale(self, pyrScale) -> None:
        """"""

    def setWinSize(self, winSize) -> None:
        """"""

    def create(self, numLevels=..., pyrScale=..., fastPyramids=..., winSize=..., numIters=..., polyN=..., polySigma=..., flags=...) -> retval:
        """"""

class FastFeatureDetector(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getNonmaxSuppression(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getType(self) -> retval:
        """"""

    def setNonmaxSuppression(self, f) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setType(self, type) -> None:
        """"""

    def create(self, threshold=..., nonmaxSuppression=..., type=...) -> retval:
        """"""

class Feature2D(ABC, builtins.object):
    @overload
    def compute(self, image, keypoints, descriptors=...) -> tuple[keypoints, descriptors]:
        """
        @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
        (second variant).

        @param image Image.
        @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
        @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
        """

    @overload
    def compute(self, images, keypoints, descriptors=...) -> tuple[keypoints, descriptors]:
        """
        @overload

        @param images Image set.
        @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
        @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
        """

    def defaultNorm(self) -> retval:
        """"""

    def descriptorSize(self) -> retval:
        """"""

    def descriptorType(self) -> retval:
        """"""

    @overload
    def detect(self, image, mask=...) -> keypoints:
        """
        @brief Detects keypoints in an image (first variant) or image set (second variant).

        @param image Image.
        @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
        @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.
        """

    @overload
    def detect(self, images, masks=...) -> keypoints:
        """
        @overload
        @param images Image set.
        @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
        @param masks Masks for each input image specifying where to look for keypoints (optional). masks[i] is a mask for images[i].
        """

    def detectAndCompute(self, image, mask, descriptors=..., useProvidedKeypoints=...) -> tuple[keypoints, descriptors]:
        """
        Detects keypoints and computes the descriptors
        """

    def empty(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    @overload
    def read(self, fileName) -> None:
        """"""

    @overload
    def read(self, arg1) -> None:
        """"""

    @overload
    def write(self, fileName) -> None:
        """"""

    @overload
    def write(self, fs, name) -> None:
        """"""

class FileNode(builtins.object):
    def at(self, i) -> retval:
        """
        @overload
        @param i Index of an element in the sequence node.
        """

    def empty(self) -> retval:
        """"""

    def getNode(self, nodename) -> retval:
        """
        @overload
        @param nodename Name of an element in the mapping node.
        """

    def isInt(self) -> retval:
        """"""

    def isMap(self) -> retval:
        """"""

    def isNamed(self) -> retval:
        """"""

    def isNone(self) -> retval:
        """"""

    def isReal(self) -> retval:
        """"""

    def isSeq(self) -> retval:
        """"""

    def isString(self) -> retval:
        """"""

    def keys(self) -> retval:
        """
        @brief Returns keys of a mapping node.
        @returns Keys of a mapping node.
        """

    def mat(self) -> retval:
        """"""

    def name(self) -> retval:
        """"""

    def rawSize(self) -> retval:
        """"""

    def real(self) -> retval:
        """
        Internal method used when reading FileStorage.
        Sets the type (int, real or string) and value of the previously created node.
        """

    def size(self) -> retval:
        """"""

    def string(self) -> retval:
        """"""

    def type(self) -> retval:
        """
        @brief Returns type of the node.
        @returns Type of the node. See FileNode::Type
        """

class FileStorage(builtins.object):
    def endWriteStruct(self) -> None:
        """
        @brief Finishes writing nested structure (should pair startWriteStruct())
        """

    def getFirstTopLevelNode(self) -> FileNode:
        """
        @brief Returns the first element of the top-level mapping.
        @returns The first element of the top-level mapping.
        """

    def getFormat(self) -> int:
        """
        @brief Returns the current format.
        * @returns The current format, see FileStorage::Mode
        """

    def getNode(self, nodename) -> retval:
        """
        @overload
        """

    def isOpened(self) -> bool:
        """
        @brief Checks whether the file is opened.

        @returns true if the object is associated with the current file and false otherwise. It is a
        good practice to call this method after you tried to open a file.
        """

    def open(self, filename: str, flags: int, encoding: str = ...) -> bool:
        """
        @brief Opens a file.

        See description of parameters in FileStorage::FileStorage. The method calls FileStorage::release
        before opening the file.
        @param filename Name of the file to open or the text string to read the data from. Extension of the file (.xml, .yml/.yaml or .json) determines its format (XML, YAML or JSON respectively). Also you can append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both FileStorage::WRITE and FileStorage::MEMORY flags are specified, source is used just to specify the output file format (e.g. mydata.xml, .yml etc.). A file name can also contain parameters. You can use this format, "*?base64" (e.g. "file.json?base64" (case sensitive)), as an alternative to FileStorage::BASE64 flag.
        @param flags Mode of operation. One of FileStorage::Mode
        @param encoding Encoding of the file. Note that UTF-16 XML encoding is not supported currently and you should use 8-bit encoding instead of it.
        """

    def release(self) -> None:
        """
        @brief Closes the file and releases all the memory buffers.

        Call this method after all I/O operations with the storage are finished.
        """

    def releaseAndGetString(self) -> retval:
        """
        @brief Closes the file and releases all the memory buffers.

        Call this method after all I/O operations with the storage are finished. If the storage was
        opened for writing data and FileStorage::WRITE was specified
        """

    def root(self, streamidx: int = ...) -> FileNode:
        """
        @brief Returns the top-level mapping
        @param streamidx Zero-based index of the stream. In most cases there is only one stream in the file. However, YAML supports multiple streams and so there can be several. @returns The top-level mapping.
        """

    @overload
    def startWriteStruct(self, name: str, flags: int, typeName: str = ...) -> None:
        """
        @brief Starts to write a nested structure (sequence or a mapping).
        @param name name of the structure. When writing to sequences (a.k.a. "arrays"), pass an empty string.
        @param flags type of the structure (FileNode::MAP or FileNode::SEQ (both with optional FileNode::FLOW)).
        @param typeName optional name of the type you store. The effect of setting this depends on the storage format. I.e. if the format has a specification for storing type information, this parameter is used.
        """

    @overload
    def startWriteStruct(self, name, flags, typeName=...) -> None:
        """
        * @brief Simplified writing API to use with bindings.
        * @param name Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
        * @param val Value of the written object.
        """

    def writeComment(self, comment: str, append: bool = ...) -> None:
        """
        @brief Writes a comment.

        The function writes a comment into file storage. The comments are skipped when the storage is read.
        @param comment The written comment, single-line or multi-line
        @param append If true, the function tries to put the comment at the end of current line. Else if the comment is multi-line, or if it does not fit at the end of the current line, the comment starts a new line.
        """

class FlannBasedMatcher(DescriptorMatcher):
    def create(self) -> retval:
        """"""

class GArray(builtins.object): ...
class GArrayDesc(builtins.object): ...

class GArrayT(builtins.object):
    def type(self) -> retval:
        """"""

class GCompileArg(builtins.object): ...

class GComputation(builtins.object):
    def apply(self, callback, args=...) -> retval:
        """
        * @brief Compile graph on-the-fly and immediately execute it on
        * the inputs data vectors.
        *
        * Number of input/output data objects must match GComputation's
        * protocol, also types of host data objects (cv::Mat, cv::Scalar)
        * must match the shapes of data objects from protocol (cv::GMat,
        * cv::GScalar). If there's a mismatch, a run-time exception will
        * be generated.
        *
        * Internally, a cv::GCompiled object is created for the given
        * input format configuration, which then is executed on the input
        * data immediately. cv::GComputation caches compiled objects
        * produced within apply() -- if this method would be called next
        * time with the same input parameters (image formats, image
        * resolution, etc), the underlying compiled graph will be reused
        * without recompilation. If new metadata doesn't match the cached
        * one, the underlying compiled graph is regenerated.
        *
        * @note compile() always triggers a compilation process and
        * produces a new GCompiled object regardless if a similar one has
        * been cached via apply() or not.
        *
        * @param ins vector of input data to process. Don't create * GRunArgs object manually, use cv::gin() wrapper instead.
        * @param outs vector of output data to fill results in. cv::Mat * objects may be empty in this vector, G-API will automatically * initialize it with the required format & dimensions. Don't * create GRunArgsP object manually, use cv::gout() wrapper instead.
        * @param args a list of compilation arguments to pass to the * underlying compilation process. Don't create GCompileArgs * object manually, use cv::compile_args() wrapper instead. * * @sa @ref gapi_data_objects, @ref gapi_compile_args
        """

    @overload
    def compileStreaming(self, in_metas, args=...) -> retval:
        """
        * @brief Compile the computation for streaming mode.
        *
        * This method triggers compilation process and produces a new
        * GStreamingCompiled object which then can process video stream
        * data of the given format. Passing a stream in a different
        * format to the compiled computation will generate a run-time
        * exception.
        *
        * @param in_metas vector of input metadata configuration. Grab * metadata from real data objects (like cv::Mat or cv::Scalar) * using cv::descr_of(), or create it on your own. *
        * @param args compilation arguments for this compilation * process. Compilation arguments directly affect what kind of * executable object would be produced, e.g. which kernels (and * thus, devices) would be used to execute computation. * * @return GStreamingCompiled, a streaming-oriented executable * computation compiled specifically for the given input * parameters. * * @sa @ref gapi_compile_args
        """

    @overload
    def compileStreaming(self, args=...) -> retval:
        """
        * @brief Compile the computation for streaming mode.
        *
        * This method triggers compilation process and produces a new
        * GStreamingCompiled object which then can process video stream
        * data in any format. Underlying mechanisms will be adjusted to
        * every new input video stream automatically, but please note that
        * _not all_ existing backends support this (see reshape()).
        *
        * @param args compilation arguments for this compilation * process. Compilation arguments directly affect what kind of * executable object would be produced, e.g. which kernels (and * thus, devices) would be used to execute computation. * * @return GStreamingCompiled, a streaming-oriented executable * computation compiled for any input image format. * * @sa @ref gapi_compile_args
        """

    @overload
    def compileStreaming(self, callback, args=...) -> retval:
        """"""

class GFTTDetector(Feature2D):
    def getBlockSize(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getGradientSize(self) -> retval:
        """"""

    def getHarrisDetector(self) -> retval:
        """"""

    def getK(self) -> retval:
        """"""

    def getMaxFeatures(self) -> retval:
        """"""

    def getMinDistance(self) -> retval:
        """"""

    def getQualityLevel(self) -> retval:
        """"""

    def setBlockSize(self, blockSize) -> None:
        """"""

    def setGradientSize(self, gradientSize_) -> None:
        """"""

    def setHarrisDetector(self, val) -> None:
        """"""

    def setK(self, k) -> None:
        """"""

    def setMaxFeatures(self, maxFeatures) -> None:
        """"""

    def setMinDistance(self, minDistance) -> None:
        """"""

    def setQualityLevel(self, qlevel) -> None:
        """"""

    @overload
    def create(self, maxCorners=..., qualityLevel=..., minDistance=..., blockSize=..., useHarrisDetector=..., k=...) -> retval:
        """"""

    @overload
    def create(self, maxCorners, qualityLevel, minDistance, blockSize, gradiantSize, useHarrisDetector=..., k=...) -> retval:
        """"""

class GFrame(builtins.object): ...

class GInferInputs(builtins.object):
    def setInput(self, name, value) -> retval:
        """"""

class GInferListInputs(builtins.object):
    def setInput(self, name, value) -> retval:
        """"""

class GInferListOutputs(builtins.object):
    def at(self, name) -> retval:
        """"""

class GInferOutputs(builtins.object):
    def at(self, name) -> retval:
        """"""

class GKernelPackage(builtins.object): ...
class GMat(builtins.object): ...

class GMatDesc(builtins.object):
    def asInterleaved(self) -> retval:
        """"""

    @overload
    def asPlanar(self) -> retval:
        """"""

    @overload
    def asPlanar(self, planes) -> retval:
        """"""

    def withDepth(self, ddepth) -> retval:
        """"""

    def withSize(self, sz) -> retval:
        """"""

    @overload
    def withSizeDelta(self, delta) -> retval:
        """"""

    @overload
    def withSizeDelta(self, dx, dy) -> retval:
        """"""

    def withType(self, ddepth, dchan) -> retval:
        """"""

class GOpaque(builtins.object): ...
class GOpaqueDesc(builtins.object): ...

class GOpaqueT(builtins.object):
    def type(self) -> retval:
        """"""

class GScalar(builtins.object): ...
class GScalarDesc(builtins.object): ...

class GStreamingCompiled(builtins.object):
    def pull(self) -> retval:
        """
        * @brief Get the next processed frame from the pipeline.
        *
        * Use gout() to create an output parameter vector.
        *
        * Output vectors must have the same number of elements as defined
        * in the cv::GComputation protocol (at the moment of its
        * construction). Shapes of elements also must conform to protocol
        * (e.g. cv::Mat needs to be passed where cv::GMat has been
        * declared as output, and so on). Run-time exception is generated
        * on type mismatch.
        *
        * This method writes new data into objects passed via output
        * vector.  If there is no data ready yet, this method blocks. Use
        * try_pull() if you need a non-blocking version.
        *
        * @param outs vector of output parameters to obtain. * @return true if next result has been obtained, *    false marks end of the stream.
        """

    def running(self) -> retval:
        """
        * @brief Test if the pipeline is running.
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        *
        * @return true if the current stream is not over yet.
        """

    def setSource(self, callback) -> None:
        """
        * @brief Specify the input data to GStreamingCompiled for
        * processing, a generic version.
        *
        * Use gin() to create an input parameter vector.
        *
        * Input vectors must have the same number of elements as defined
        * in the cv::GComputation protocol (at the moment of its
        * construction). Shapes of elements also must conform to protocol
        * (e.g. cv::Mat needs to be passed where cv::GMat has been
        * declared as input, and so on). Run-time exception is generated
        * on type mismatch.
        *
        * In contrast with regular GCompiled, user can also pass an
        * object of type GVideoCapture for a GMat parameter of the parent
        * GComputation.  The compiled pipeline will start fetching data
        * from that GVideoCapture and feeding it into the
        * pipeline. Pipeline stops when a GVideoCapture marks end of the
        * stream (or when stop() is called).
        *
        * Passing a regular Mat for a GMat parameter makes it "infinite"
        * source -- pipeline may run forever feeding with this Mat until
        * stopped explicitly.
        *
        * Currently only a single GVideoCapture is supported as input. If
        * the parent GComputation is declared with multiple input GMat's,
        * one of those can be specified as GVideoCapture but all others
        * must be regular Mat objects.
        *
        * Throws if pipeline is already running. Use stop() and then
        * setSource() to run the graph on a new video stream.
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        *
        * @param ins vector of inputs to process. * @sa gin
        """

    def start(self) -> None:
        """
        * @brief Start the pipeline execution.
        *
        * Use pull()/try_pull() to obtain data. Throws an exception if
        * a video source was not specified.
        *
        * setSource() must be called first, even if the pipeline has been
        * working already and then stopped (explicitly via stop() or due
        * stream completion)
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        """

    def stop(self) -> None:
        """
        * @brief Stop (abort) processing the pipeline.
        *
        * Note - it is not pause but a complete stop. Calling start()
        * will cause G-API to start processing the stream from the early beginning.
        *
        * Throws if the pipeline is not running.
        """

class GeneralizedHough(Algorithm):
    @overload
    def detect(self, image, positions=..., votes=...) -> tuple[positions, votes]:
        """"""

    @overload
    def detect(self, edges, dx, dy, positions=..., votes=...) -> tuple[positions, votes]:
        """"""

    def getCannyHighThresh(self) -> retval:
        """"""

    def getCannyLowThresh(self) -> retval:
        """"""

    def getDp(self) -> retval:
        """"""

    def getMaxBufferSize(self) -> retval:
        """"""

    def getMinDist(self) -> retval:
        """"""

    def setCannyHighThresh(self, cannyHighThresh) -> None:
        """"""

    def setCannyLowThresh(self, cannyLowThresh) -> None:
        """"""

    def setDp(self, dp) -> None:
        """"""

    def setMaxBufferSize(self, maxBufferSize) -> None:
        """"""

    def setMinDist(self, minDist) -> None:
        """"""

    @overload
    def setTemplate(self, templ, templCenter=...) -> None:
        """"""

    @overload
    def setTemplate(self, edges, dx, dy, templCenter=...) -> None:
        """"""

class GeneralizedHoughBallard(GeneralizedHough):
    def getLevels(self) -> retval:
        """"""

    def getVotesThreshold(self) -> retval:
        """"""

    def setLevels(self, levels) -> None:
        """"""

    def setVotesThreshold(self, votesThreshold) -> None:
        """"""

class GeneralizedHoughGuil(GeneralizedHough):
    def getAngleEpsilon(self) -> retval:
        """"""

    def getAngleStep(self) -> retval:
        """"""

    def getAngleThresh(self) -> retval:
        """"""

    def getLevels(self) -> retval:
        """"""

    def getMaxAngle(self) -> retval:
        """"""

    def getMaxScale(self) -> retval:
        """"""

    def getMinAngle(self) -> retval:
        """"""

    def getMinScale(self) -> retval:
        """"""

    def getPosThresh(self) -> retval:
        """"""

    def getScaleStep(self) -> retval:
        """"""

    def getScaleThresh(self) -> retval:
        """"""

    def getXi(self) -> retval:
        """"""

    def setAngleEpsilon(self, angleEpsilon) -> None:
        """"""

    def setAngleStep(self, angleStep) -> None:
        """"""

    def setAngleThresh(self, angleThresh) -> None:
        """"""

    def setLevels(self, levels) -> None:
        """"""

    def setMaxAngle(self, maxAngle) -> None:
        """"""

    def setMaxScale(self, maxScale) -> None:
        """"""

    def setMinAngle(self, minAngle) -> None:
        """"""

    def setMinScale(self, minScale) -> None:
        """"""

    def setPosThresh(self, posThresh) -> None:
        """"""

    def setScaleStep(self, scaleStep) -> None:
        """"""

    def setScaleThresh(self, scaleThresh) -> None:
        """"""

    def setXi(self, xi) -> None:
        """"""

class HOGDescriptor(builtins.object):
    def checkDetectorSize(self) -> retval:
        """
        @brief Checks if detector size equal to descriptor size.
        """

    def compute(self, img, winStride=..., padding=..., locations=...) -> descriptors:
        """
        @brief Computes HOG descriptors of given image.
        @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
        @param descriptors Matrix of the type CV_32F
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param locations Vector of Point
        """

    def computeGradient(self, img, grad, angleOfs, paddingTL=..., paddingBR=...) -> tuple[grad, angleOfs]:
        """
        @brief  Computes gradients and quantized gradient orientations.
        @param img Matrix contains the image to be computed
        @param grad Matrix of type CV_32FC2 contains computed gradients
        @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
        @param paddingTL Padding from top-left
        @param paddingBR Padding from bottom-right
        """

    def detect(self, img, hitThreshold=..., winStride=..., padding=..., searchLocations=...) -> tuple[foundLocations, weights]:
        """
        @brief Performs object detection without a multi-scale window.
        @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
        @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
        @param weights Vector that will contain confidence values for each detected object.
        @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param searchLocations Vector of Point includes set of requested locations to be evaluated.
        """

    def detectMultiScale(self, img, hitThreshold=..., winStride=..., padding=..., scale=..., groupThreshold=..., useMeanshiftGrouping=...) -> tuple[foundLocations, foundWeights]:
        """
        @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
        of rectangles.
        @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
        @param foundLocations Vector of rectangles where each rectangle contains the detected object.
        @param foundWeights Vector that will contain confidence values for each detected object.
        @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param scale Coefficient of the detection window increase.
        @param groupThreshold Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping.
        @param useMeanshiftGrouping indicates grouping algorithm
        """

    def getDescriptorSize(self) -> retval:
        """
        @brief Returns the number of coefficients required for the classification.
        """

    def getWinSigma(self) -> retval:
        """
        @brief Returns winSigma value
        """

    def load(self, filename, objname=...) -> retval:
        """
        @brief loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file
        @param filename Name of the file to read.
        @param objname The optional name of the node to read (if empty, the first top-level node will be used).
        """

    def save(self, filename, objname=...) -> None:
        """
        @brief saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
        @param filename File name
        @param objname Object name
        """

    def setSVMDetector(self, svmdetector) -> None:
        """
        @brief Sets coefficients for the linear SVM classifier.
        @param svmdetector coefficients for the linear SVM classifier.
        """

    def getDaimlerPeopleDetector(self) -> retval:
        """
        @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
        """

    def getDefaultPeopleDetector(self) -> retval:
        """
        @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
        """

class HausdorffDistanceExtractor(ShapeDistanceExtractor):
    def getDistanceFlag(self) -> retval:
        """"""

    def getRankProportion(self) -> retval:
        """"""

    def setDistanceFlag(self, distanceFlag) -> None:
        """
        @brief Set the norm used to compute the Hausdorff value between two shapes. It can be L1 or L2 norm.

        @param distanceFlag Flag indicating which norm is used to compute the Hausdorff distance (NORM_L1, NORM_L2).
        """

    def setRankProportion(self, rankProportion) -> None:
        """
        @brief This method sets the rank proportion (or fractional value) that establish the Kth ranked value of
        the partial Hausdorff distance. Experimentally had been shown that 0.6 is a good value to compare
        shapes.

        @param rankProportion fractional value (between 0 and 1).
        """

class HistogramCostExtractor(Algorithm):
    def buildCostMatrix(self, descriptors1, descriptors2, costMatrix=...) -> costMatrix:
        """"""

    def getDefaultCost(self) -> retval:
        """"""

    def getNDummies(self) -> retval:
        """"""

    def setDefaultCost(self, defaultCost) -> None:
        """"""

    def setNDummies(self, nDummies) -> None:
        """"""

class KAZE(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getDiffusivity(self) -> retval:
        """"""

    def getExtended(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getUpright(self) -> retval:
        """"""

    def setDiffusivity(self, diff) -> None:
        """"""

    def setExtended(self, extended) -> None:
        """"""

    def setNOctaveLayers(self, octaveLayers) -> None:
        """"""

    def setNOctaves(self, octaves) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setUpright(self, upright) -> None:
        """"""

    def create(self, extended=..., upright=..., threshold=..., nOctaves=..., nOctaveLayers=..., diffusivity=...) -> retval:
        """
        @brief The KAZE constructor

        @param extended Set to enable extraction of extended (128-byte) descriptor.
        @param upright Set to enable use of upright descriptors (non rotation-invariant).
        @param threshold Detector response threshold to accept point
        @param nOctaves Maximum octave evolution of the image
        @param nOctaveLayers Default number of sublevels per scale level
        @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        """

class KalmanFilter(builtins.object):
    def correct(self, measurement) -> retval:
        """
        @brief Updates the predicted state from the measurement.

        @param measurement The measured system parameters
        """

    def predict(self, control=...) -> retval:
        """
        @brief Computes a predicted state.

        @param control The optional input control
        """

class KeyPoint(builtins.object):
    @overload
    def convert(self, keypoints, keypointIndexes=...) -> points2f:
        """
        This method converts vector of keypoints to vector of points or the reverse, where each keypoint is
        assigned the same size and the same orientation.

        @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
        @param points2f Array of (x,y) coordinates of each keypoint
        @param keypointIndexes Array of indexes of keypoints to be converted to points. (Acts like a mask to convert only specified keypoints)
        """

    @overload
    def convert(self, points2f, size=..., response=..., octave=..., class_id=...) -> keypoints:
        """
        @overload
        @param points2f Array of (x,y) coordinates of each keypoint
        @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
        @param size keypoint diameter
        @param response keypoint detector response on the keypoint (that is, strength of the keypoint)
        @param octave pyramid octave in which the keypoint has been detected
        @param class_id object id
        """

    def overlap(self, kp1, kp2) -> retval:
        """
        This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint
        regions' intersection and area of keypoint regions' union (considering keypoint region as circle).
        If they don't overlap, we get zero. If they coincide at same location with same size, we get 1.
        @param kp1 First keypoint
        @param kp2 Second keypoint
        """

class LineSegmentDetector(Algorithm):
    def compareSegments(self, size, lines1, lines2, image=...) -> tuple[retval, image]:
        """
        @brief Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.

        @param size The size of the image, where lines1 and lines2 were found.
        @param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
        @param lines2 The second group of lines. They visualized in red color.
        @param image Optional image, where the lines will be drawn. The image should be color(3-channel) in order for lines1 and lines2 to be drawn in the above mentioned colors.
        """

    def detect(self, image, lines=..., width=..., prec=..., nfa=...) -> tuple[lines, width, prec, nfa]:
        """
        @brief Finds lines in the input image.

        This is the output of the default parameters of the algorithm on the above shown image.

        ![image](pics/building_lsd.png)

        @param image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use: `lsd_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
        @param lines A vector of Vec4f elements specifying the beginning and ending point of a line. Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly oriented depending on the gradient.
        @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
        @param prec Vector of precisions with which the lines are found.
        @param nfa Vector containing number of false alarms in the line region, with precision of 10%. The bigger the value, logarithmically better the detection. - -1 corresponds to 10 mean false alarms - 0 corresponds to 1 mean false alarm - 1 corresponds to 0.1 mean false alarms This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
        """

    def drawSegments(self, image, lines) -> image:
        """
        @brief Draws the line segments on a given image.
        @param image The image, where the lines will be drawn. Should be bigger or equal to the image, where the lines were found.
        @param lines A vector of the lines that needed to be drawn.
        """

class MSER(Feature2D):
    def detectRegions(self, image) -> tuple[msers, bboxes]:
        """
        @brief Detect %MSER regions

        @param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
        @param msers resulting list of point sets
        @param bboxes resulting bounding boxes
        """

    def getAreaThreshold(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getDelta(self) -> retval:
        """"""

    def getEdgeBlurSize(self) -> retval:
        """"""

    def getMaxArea(self) -> retval:
        """"""

    def getMaxEvolution(self) -> retval:
        """"""

    def getMaxVariation(self) -> retval:
        """"""

    def getMinArea(self) -> retval:
        """"""

    def getMinDiversity(self) -> retval:
        """"""

    def getMinMargin(self) -> retval:
        """"""

    def getPass2Only(self) -> retval:
        """"""

    def setAreaThreshold(self, areaThreshold) -> None:
        """"""

    def setDelta(self, delta) -> None:
        """"""

    def setEdgeBlurSize(self, edge_blur_size) -> None:
        """"""

    def setMaxArea(self, maxArea) -> None:
        """"""

    def setMaxEvolution(self, maxEvolution) -> None:
        """"""

    def setMaxVariation(self, maxVariation) -> None:
        """"""

    def setMinArea(self, minArea) -> None:
        """"""

    def setMinDiversity(self, minDiversity) -> None:
        """"""

    def setMinMargin(self, min_margin) -> None:
        """"""

    def setPass2Only(self, f) -> None:
        """"""

    def create(self, delta=..., min_area=..., max_area=..., max_variation=..., min_diversity=..., max_evolution=..., area_threshold=..., min_margin=..., edge_blur_size=...) -> retval:
        """
        @brief Full constructor for %MSER detector

        @param delta it compares \f$(size_{i}-size_{i-delta})/size_{i-delta}\f$
        @param min_area prune the area which smaller than minArea
        @param max_area prune the area which bigger than maxArea
        @param max_variation prune the area have similar size to its children
        @param min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
        @param max_evolution  for color image, the evolution steps
        @param area_threshold for color image, the area threshold to cause re-initialize
        @param min_margin for color image, ignore too small margin
        @param edge_blur_size for color image, the aperture size for edge blur
        """

class Mat(numpy.ndarray): ...

class MergeDebevec(MergeExposures):
    @overload
    def process(self, src, times, response, dst=...) -> dst:
        """"""

    @overload
    def process(self, src, times, dst=...) -> dst:
        """"""

class MergeExposures(Algorithm):
    def process(self, src, times, response, dst=...) -> dst:
        """
        @brief Merges images.

        @param src vector of input images
        @param dst result image
        @param times vector of exposure time values for each image
        @param response 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.
        """

class MergeMertens(MergeExposures):
    def getContrastWeight(self) -> retval:
        """"""

    def getExposureWeight(self) -> retval:
        """"""

    def getSaturationWeight(self) -> retval:
        """"""

    @overload
    def process(self, src, times, response, dst=...) -> dst:
        """"""

    @overload
    def process(self, src, dst=...) -> dst:
        """
        @brief Short version of process, that doesn't take extra arguments.

        @param src vector of input images
        @param dst result image
        """

    def setContrastWeight(self, contrast_weiht) -> None:
        """"""

    def setExposureWeight(self, exposure_weight) -> None:
        """"""

    def setSaturationWeight(self, saturation_weight) -> None:
        """"""

class MergeRobertson(MergeExposures):
    @overload
    def process(self, src, times, response, dst=...) -> dst:
        """"""

    @overload
    def process(self, src, times, dst=...) -> dst:
        """"""

class NormHistogramCostExtractor(HistogramCostExtractor):
    def getNormFlag(self) -> retval:
        """"""

    def setNormFlag(self, flag) -> None:
        """"""

class ORB(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getEdgeThreshold(self) -> retval:
        """"""

    def getFastThreshold(self) -> retval:
        """"""

    def getFirstLevel(self) -> retval:
        """"""

    def getMaxFeatures(self) -> retval:
        """"""

    def getNLevels(self) -> retval:
        """"""

    def getPatchSize(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def getScoreType(self) -> retval:
        """"""

    def getWTA_K(self) -> retval:
        """"""

    def setEdgeThreshold(self, edgeThreshold) -> None:
        """"""

    def setFastThreshold(self, fastThreshold) -> None:
        """"""

    def setFirstLevel(self, firstLevel) -> None:
        """"""

    def setMaxFeatures(self, maxFeatures) -> None:
        """"""

    def setNLevels(self, nlevels) -> None:
        """"""

    def setPatchSize(self, patchSize) -> None:
        """"""

    def setScaleFactor(self, scaleFactor) -> None:
        """"""

    def setScoreType(self, scoreType) -> None:
        """"""

    def setWTA_K(self, wta_k) -> None:
        """"""

    def create(self, nfeatures=..., scaleFactor=..., nlevels=..., edgeThreshold=..., firstLevel=..., WTA_K=..., scoreType=..., patchSize=..., fastThreshold=...) -> retval:
        """
        @brief The ORB constructor

        @param nfeatures The maximum number of features to retain.
        @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        @param nlevels The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        @param edgeThreshold This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        @param firstLevel The level of pyramid to put source image to. Previous layers are filled with upscaled source image.
        @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.
        @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        @param fastThreshold the fast threshold
        """

class PyRotationWarper(builtins.object):
    def buildMaps(self, src_size, K, R, xmap=..., ymap=...) -> tuple[retval, xmap, ymap]:
        """
        @brief Builds the projection maps according to the given camera data.

        @param src_size Source image size
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param xmap Projection map for the x axis
        @param ymap Projection map for the y axis @return Projected image minimum bounding box
        """

    def getScale(self) -> retval:
        """"""

    def setScale(self, arg1) -> None:
        """"""

    def warp(self, src, K, R, interp_mode, border_mode, dst=...) -> tuple[retval, dst]:
        """
        @brief Projects the image.

        @param src Source image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst Projected image @return Project image top-left corner
        """

    def warpBackward(self, src, K, R, interp_mode, border_mode, dst_size, dst=...) -> dst:
        """
        @brief Projects the image backward.

        @param src Projected image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst_size Backward-projected image size
        @param dst Backward-projected image
        """

    def warpPoint(self, pt, K, R) -> retval:
        """
        @brief Projects the image point.

        @param pt Source point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Projected point
        """

    def warpPointBackward(self, pt, K, R) -> retval:
        """
        @brief Projects the image point backward.

        @param pt Projected point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Backward-projected point
        """

    def warpRoi(self, src_size, K, R) -> retval:
        """
        @param src_size Source image bounding box
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Projected image minimum bounding box
        """

class QRCodeDetector(builtins.object):
    def decode(self, img, points, straight_qrcode=...) -> tuple[retval, straight_qrcode]:
        """
        @brief Decodes QR code in image once it's found by the detect() method.

        Returns UTF8-encoded output string or empty string if the code cannot be decoded.
        @param img grayscale or color (BGR) image containing QR code.
        @param points Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def decodeCurved(self, img, points, straight_qrcode=...) -> tuple[retval, straight_qrcode]:
        """
        @brief Decodes QR code on a curved surface in image once it's found by the detect() method.

        Returns UTF8-encoded output string or empty string if the code cannot be decoded.
        @param img grayscale or color (BGR) image containing QR code.
        @param points Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def decodeMulti(self, img, points, straight_qrcode=...) -> tuple[retval, decoded_info, straight_qrcode]:
        """
        @brief Decodes QR codes in image once it's found by the detect() method.
        @param img grayscale or color (BGR) image containing QR codes.
        @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
        @param points vector of Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
        """

    def detect(self, img, points=...) -> tuple[retval, points]:
        """
        @brief Detects QR code in image and returns the quadrangle containing the code.
        @param img grayscale or color (BGR) image containing (or not) QR code.
        @param points Output vector of vertices of the minimum-area quadrangle containing the code.
        """

    def detectAndDecode(self, img, points=..., straight_qrcode=...) -> tuple[retval, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR code

        @param img grayscale or color (BGR) image containing QR code.
        @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def detectAndDecodeCurved(self, img, points=..., straight_qrcode=...) -> tuple[retval, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR code on a curved surface

        @param img grayscale or color (BGR) image containing QR code.
        @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def detectAndDecodeMulti(self, img, points=..., straight_qrcode=...) -> tuple[retval, decoded_info, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR codes
        @param img grayscale or color (BGR) image containing QR codes.
        @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
        @param points optional output vector of vertices of the found QR code quadrangles. Will be empty if not found.
        @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
        """

    def detectMulti(self, img, points=...) -> tuple[retval, points]:
        """
        @brief Detects QR codes in image and returns the vector of the quadrangles containing the codes.
        @param img grayscale or color (BGR) image containing (or not) QR codes.
        @param points Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
        """

    def setEpsX(self, epsX) -> None:
        """
        @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
        @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern of the scheme 1:1:3:1:1 according to QR code standard.
        """

    def setEpsY(self, epsY) -> None:
        """
        @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
        @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern of the scheme 1:1:3:1:1 according to QR code standard.
        """

    def setUseAlignmentMarkers(self, useAlignmentMarkers) -> None:
        """
        @brief use markers to improve the position of the corners of the QR code
        *
        * alignmentMarkers using by default
        """

class QRCodeEncoder(builtins.object):
    def encode(self, encoded_info, qrcode=...) -> qrcode:
        """
        @brief Generates QR code from input string.
        @param encoded_info Input string to encode.
        @param qrcode Generated QR code.
        """

    def encodeStructuredAppend(self, encoded_info, qrcodes=...) -> qrcodes:
        """
        @brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
        @param encoded_info Input string to encode.
        @param qrcodes Vector of generated QR codes.
        """

    def create(self, parameters=...) -> retval:
        """
        @brief Constructor
        @param parameters QR code encoder parameters QRCodeEncoder::Params
        """

QRCodeEncoder_Params = Params

class SIFT(Feature2D):
    def getContrastThreshold(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getEdgeThreshold(self) -> retval:
        """"""

    def getNFeatures(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def setContrastThreshold(self, contrastThreshold) -> None:
        """"""

    def setEdgeThreshold(self, edgeThreshold) -> None:
        """"""

    def setNFeatures(self, maxFeatures) -> None:
        """"""

    def setNOctaveLayers(self, nOctaveLayers) -> None:
        """"""

    def setSigma(self, sigma) -> None:
        """"""

    @overload
    def create(self, nfeatures=..., nOctaveLayers=..., contrastThreshold=..., edgeThreshold=..., sigma=...) -> retval:
        """
        @param nfeatures The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.  @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.
        @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        """

    @overload
    def create(self, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, descriptorType) -> retval:
        """
        @brief Create SIFT with specified descriptorType.
        @param nfeatures The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast)
        @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution.
        @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.  @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09.
        @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained).
        @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        @param descriptorType The type of descriptors. Only CV_32F and CV_8U are supported.
        """

class ShapeContextDistanceExtractor(ShapeDistanceExtractor):
    def getAngularBins(self) -> retval:
        """"""

    def getBendingEnergyWeight(self) -> retval:
        """"""

    def getCostExtractor(self) -> retval:
        """"""

    def getImageAppearanceWeight(self) -> retval:
        """"""

    def getImages(self, image1=..., image2=...) -> tuple[image1, image2]:
        """"""

    def getInnerRadius(self) -> retval:
        """"""

    def getIterations(self) -> retval:
        """"""

    def getOuterRadius(self) -> retval:
        """"""

    def getRadialBins(self) -> retval:
        """"""

    def getRotationInvariant(self) -> retval:
        """"""

    def getShapeContextWeight(self) -> retval:
        """"""

    def getStdDev(self) -> retval:
        """"""

    def getTransformAlgorithm(self) -> retval:
        """"""

    def setAngularBins(self, nAngularBins) -> None:
        """
        @brief Establish the number of angular bins for the Shape Context Descriptor used in the shape matching
        pipeline.

        @param nAngularBins The number of angular bins in the shape context descriptor.
        """

    def setBendingEnergyWeight(self, bendingEnergyWeight) -> None:
        """
        @brief Set the weight of the Bending Energy in the final value of the shape distance. The bending energy
        definition depends on what transformation is being used to align the shapes. The final value of the
        shape distance is a user-defined linear combination of the shape context distance, an image
        appearance distance, and a bending energy.

        @param bendingEnergyWeight The weight of the Bending Energy in the final distance value.
        """

    def setCostExtractor(self, comparer) -> None:
        """
        @brief Set the algorithm used for building the shape context descriptor cost matrix.

        @param comparer Smart pointer to a HistogramCostExtractor, an algorithm that defines the cost matrix between descriptors.
        """

    def setImageAppearanceWeight(self, imageAppearanceWeight) -> None:
        """
        @brief Set the weight of the Image Appearance cost in the final value of the shape distance. The image
        appearance cost is defined as the sum of squared brightness differences in Gaussian windows around
        corresponding image points. The final value of the shape distance is a user-defined linear
        combination of the shape context distance, an image appearance distance, and a bending energy. If
        this value is set to a number different from 0, is mandatory to set the images that correspond to
        each shape.

        @param imageAppearanceWeight The weight of the appearance cost in the final distance value.
        """

    def setImages(self, image1, image2) -> None:
        """
        @brief Set the images that correspond to each shape. This images are used in the calculation of the Image
        Appearance cost.

        @param image1 Image corresponding to the shape defined by contours1.
        @param image2 Image corresponding to the shape defined by contours2.
        """

    def setInnerRadius(self, innerRadius) -> None:
        """
        @brief Set the inner radius of the shape context descriptor.

        @param innerRadius The value of the inner radius.
        """

    def setIterations(self, iterations) -> None:
        """"""

    def setOuterRadius(self, outerRadius) -> None:
        """
        @brief Set the outer radius of the shape context descriptor.

        @param outerRadius The value of the outer radius.
        """

    def setRadialBins(self, nRadialBins) -> None:
        """
        @brief Establish the number of radial bins for the Shape Context Descriptor used in the shape matching
        pipeline.

        @param nRadialBins The number of radial bins in the shape context descriptor.
        """

    def setRotationInvariant(self, rotationInvariant) -> None:
        """"""

    def setShapeContextWeight(self, shapeContextWeight) -> None:
        """
        @brief Set the weight of the shape context distance in the final value of the shape distance. The shape
        context distance between two shapes is defined as the symmetric sum of shape context matching costs
        over best matching points. The final value of the shape distance is a user-defined linear
        combination of the shape context distance, an image appearance distance, and a bending energy.

        @param shapeContextWeight The weight of the shape context distance in the final distance value.
        """

    def setStdDev(self, sigma) -> None:
        """
        @brief Set the value of the standard deviation for the Gaussian window for the image appearance cost.

        @param sigma Standard Deviation.
        """

    def setTransformAlgorithm(self, transformer) -> None:
        """
        @brief Set the algorithm used for aligning the shapes.

        @param transformer Smart pointer to a ShapeTransformer, an algorithm that defines the aligning transformation.
        """

class ShapeDistanceExtractor(Algorithm):
    def computeDistance(self, contour1, contour2) -> retval:
        """
        @brief Compute the shape distance between two shapes defined by its contours.

        @param contour1 Contour defining first shape.
        @param contour2 Contour defining second shape.
        """

class ShapeTransformer(Algorithm):
    def applyTransformation(self, input, output=...) -> tuple[retval, output]:
        """
        @brief Apply a transformation, given a pre-estimated transformation parameters.

        @param input Contour (set of points) to apply the transformation.
        @param output Output contour.
        """

    def estimateTransformation(self, transformingShape, targetShape, matches) -> None:
        """
        @brief Estimate the transformation parameters of the current transformer algorithm, based on point matches.

        @param transformingShape Contour defining first shape.
        @param targetShape Contour defining second shape (Target).
        @param matches Standard vector of Matches between points.
        """

    def warpImage(self, transformingImage, output=..., flags=..., borderMode=..., borderValue=...) -> output:
        """
        @brief Apply a transformation, given a pre-estimated transformation parameters, to an Image.

        @param transformingImage Input image.
        @param output Output image.
        @param flags Image interpolation method.
        @param borderMode border style.
        @param borderValue border value.
        """

class SimpleBlobDetector(Feature2D):
    def getBlobContours(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getParams(self) -> retval:
        """"""

    def setParams(self, params) -> None:
        """"""

    def create(self, parameters=...) -> retval:
        """"""

SimpleBlobDetector_Params = Params

class SparseOpticalFlow(Algorithm):
    def calc(self, prevImg, nextImg, prevPts, nextPts, status=..., err=...) -> tuple[nextPts, status, err]:
        """
        @brief Calculates a sparse optical flow.

        @param prevImg First input image.
        @param nextImg Second input image of the same size and the same type as prevImg.
        @param prevPts Vector of 2D points for which the flow needs to be found.
        @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
        @param status Output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.
        @param err Optional output vector that contains error response for each point (inverse confidence).
        """

class SparsePyrLKOpticalFlow(SparseOpticalFlow):
    def getFlags(self) -> retval:
        """"""

    def getMaxLevel(self) -> retval:
        """"""

    def getMinEigThreshold(self) -> retval:
        """"""

    def getTermCriteria(self) -> retval:
        """"""

    def getWinSize(self) -> retval:
        """"""

    def setFlags(self, flags) -> None:
        """"""

    def setMaxLevel(self, maxLevel) -> None:
        """"""

    def setMinEigThreshold(self, minEigThreshold) -> None:
        """"""

    def setTermCriteria(self, crit) -> None:
        """"""

    def setWinSize(self, winSize) -> None:
        """"""

    def create(self, winSize=..., maxLevel=..., crit=..., flags=..., minEigThreshold=...) -> retval:
        """"""

class StereoBM(StereoMatcher):
    def getPreFilterCap(self) -> retval:
        """"""

    def getPreFilterSize(self) -> retval:
        """"""

    def getPreFilterType(self) -> retval:
        """"""

    def getROI1(self) -> retval:
        """"""

    def getROI2(self) -> retval:
        """"""

    def getSmallerBlockSize(self) -> retval:
        """"""

    def getTextureThreshold(self) -> retval:
        """"""

    def getUniquenessRatio(self) -> retval:
        """"""

    def setPreFilterCap(self, preFilterCap) -> None:
        """"""

    def setPreFilterSize(self, preFilterSize) -> None:
        """"""

    def setPreFilterType(self, preFilterType) -> None:
        """"""

    def setROI1(self, roi1) -> None:
        """"""

    def setROI2(self, roi2) -> None:
        """"""

    def setSmallerBlockSize(self, blockSize) -> None:
        """"""

    def setTextureThreshold(self, textureThreshold) -> None:
        """"""

    def setUniquenessRatio(self, uniquenessRatio) -> None:
        """"""

    def create(self, numDisparities=..., blockSize=...) -> retval:
        """
        @brief Creates StereoBM object

        @param numDisparities the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity.
        @param blockSize the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.  The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for a specific stereo pair.
        """

class StereoMatcher(Algorithm):
    def compute(self, left, right, disparity=...) -> disparity:
        """
        @brief Computes disparity map for the specified stereo pair

        @param left Left 8-bit single-channel image.
        @param right Right image of the same size and the same type as the left one.
        @param disparity Output disparity map. It has the same size as the input images. Some algorithms, like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
        """

    def getBlockSize(self) -> retval:
        """"""

    def getDisp12MaxDiff(self) -> retval:
        """"""

    def getMinDisparity(self) -> retval:
        """"""

    def getNumDisparities(self) -> retval:
        """"""

    def getSpeckleRange(self) -> retval:
        """"""

    def getSpeckleWindowSize(self) -> retval:
        """"""

    def setBlockSize(self, blockSize) -> None:
        """"""

    def setDisp12MaxDiff(self, disp12MaxDiff) -> None:
        """"""

    def setMinDisparity(self, minDisparity) -> None:
        """"""

    def setNumDisparities(self, numDisparities) -> None:
        """"""

    def setSpeckleRange(self, speckleRange) -> None:
        """"""

    def setSpeckleWindowSize(self, speckleWindowSize) -> None:
        """"""

class StereoSGBM(StereoMatcher):
    def getMode(self) -> retval:
        """"""

    def getP1(self) -> retval:
        """"""

    def getP2(self) -> retval:
        """"""

    def getPreFilterCap(self) -> retval:
        """"""

    def getUniquenessRatio(self) -> retval:
        """"""

    def setMode(self, mode) -> None:
        """"""

    def setP1(self, P1) -> None:
        """"""

    def setP2(self, P2) -> None:
        """"""

    def setPreFilterCap(self, preFilterCap) -> None:
        """"""

    def setUniquenessRatio(self, uniquenessRatio) -> None:
        """"""

    def create(self, minDisparity=..., numDisparities=..., blockSize=..., P1=..., P2=..., disp12MaxDiff=..., preFilterCap=..., uniquenessRatio=..., speckleWindowSize=..., speckleRange=..., mode=...) -> retval:
        """
        @brief Creates StereoSGBM object

        @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be somewhere in the 3..11 range.
        @param P1 The first parameter controlling the disparity smoothness. See below.
        @param P2 The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and 32\*number_of_image_channels\*blockSize\*blockSize , respectively).
        @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
        @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        @param speckleRange Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .  The first constructor initializes StereoSGBM with all the default parameters. So, you only have to set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter to a custom value.
        """

class Stitcher(builtins.object):
    @overload
    @overload
    def composePanorama(self, pano=...) -> tuple[retval, pano]:
        """
        @overload
        """

    @overload
    def composePanorama(self, images, pano=...) -> tuple[retval, pano]:
        """
        @brief These functions try to compose the given images (or images stored internally from the other function
        calls) into the final pano under the assumption that the image transformations were estimated
        before.

        @note Use the functions only if you're aware of the stitching pipeline, otherwise use
        Stitcher::stitch.

        @param images Input images.
        @param pano Final pano. @return Status code.
        """

    def compositingResol(self) -> retval:
        """"""

    def estimateTransform(self, images, masks=...) -> retval:
        """
        @brief These functions try to match the given images and to estimate rotations of each camera.

        @note Use the functions only if you're aware of the stitching pipeline, otherwise use
        Stitcher::stitch.

        @param images Input images.
        @param masks Masks for each input image specifying where to look for keypoints (optional). @return Status code.
        """

    def interpolationFlags(self) -> retval:
        """"""

    def panoConfidenceThresh(self) -> retval:
        """"""

    def registrationResol(self) -> retval:
        """"""

    def seamEstimationResol(self) -> retval:
        """"""

    def setCompositingResol(self, resol_mpx) -> None:
        """"""

    def setInterpolationFlags(self, interp_flags) -> None:
        """"""

    def setPanoConfidenceThresh(self, conf_thresh) -> None:
        """"""

    def setRegistrationResol(self, resol_mpx) -> None:
        """"""

    def setSeamEstimationResol(self, resol_mpx) -> None:
        """"""

    def setWaveCorrection(self, flag) -> None:
        """"""

    @overload
    def stitch(self, images, pano=...) -> tuple[retval, pano]:
        """
        @overload
        """

    @overload
    def stitch(self, images, masks, pano=...) -> tuple[retval, pano]:
        """
        @brief These functions try to stitch the given images.

        @param images Input images.
        @param masks Masks for each input image specifying where to look for keypoints (optional).
        @param pano Final pano. @return Status code.
        """

    def waveCorrection(self) -> retval:
        """"""

    def workScale(self) -> retval:
        """"""

    def create(self, mode=...) -> retval:
        """
        @brief Creates a Stitcher configured in one of the stitching modes.

        @param mode Scenario for stitcher operation. This is usually determined by source of images to stitch and their transformation. Default parameters will be chosen for operation in given scenario. @return Stitcher class instance.
        """

class Subdiv2D(builtins.object):
    def edgeDst(self, edge) -> tuple[retval, dstpt]:
        """
        @brief Returns the edge destination.

        @param edge Subdivision edge ID.
        @param dstpt Output vertex location.  @returns vertex ID.
        """

    def edgeOrg(self, edge) -> tuple[retval, orgpt]:
        """
        @brief Returns the edge origin.

        @param edge Subdivision edge ID.
        @param orgpt Output vertex location.  @returns vertex ID.
        """

    def findNearest(self, pt) -> tuple[retval, nearestPt]:
        """
        @brief Finds the subdivision vertex closest to the given point.

        @param pt Input point.
        @param nearestPt Output subdivision vertex point.  The function is another function that locates the input point within the subdivision. It finds the subdivision vertex that is the closest to the input point. It is not necessarily one of vertices of the facet containing the input point, though the facet (located using locate() ) is used as a starting point.  @returns vertex ID.
        """

    def getEdge(self, edge, nextEdgeType) -> retval:
        """
        @brief Returns one of the edges related to the given edge.

        @param edge Subdivision edge ID.
        @param nextEdgeType Parameter specifying which of the related edges to return. The following values are possible: -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge) -   NEXT_AROUND_DST next around the edge vertex ( eDnext ) -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext ) -   PREV_AROUND_DST previous around the edge destination (reversed eLnext ) -   NEXT_AROUND_LEFT next around the left facet ( eLnext ) -   NEXT_AROUND_RIGHT next around the right facet ( eRnext ) -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext ) -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )  ![sample output](pics/quadedge.png)  @returns edge ID related to the input edge.
        """

    def getEdgeList(self) -> edgeList:
        """
        @brief Returns a list of all edges.

        @param edgeList Output vector.  The function gives each edge as a 4 numbers vector, where each two are one of the edge vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
        """

    def getLeadingEdgeList(self) -> leadingEdgeList:
        """
        @brief Returns a list of the leading edge ID connected to each triangle.

        @param leadingEdgeList Output vector.  The function gives one edge ID for each triangle.
        """

    def getTriangleList(self) -> triangleList:
        """
        @brief Returns a list of all triangles.

        @param triangleList Output vector.  The function gives each triangle as a 6 numbers vector, where each two are one of the triangle vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
        """

    def getVertex(self, vertex) -> tuple[retval, firstEdge]:
        """
        @brief Returns vertex location from vertex ID.

        @param vertex vertex ID.
        @param firstEdge Optional. The first edge ID which is connected to the vertex. @returns vertex (x,y)
        """

    def getVoronoiFacetList(self, idx) -> tuple[facetList, facetCenters]:
        """
        @brief Returns a list of all Voronoi facets.

        @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
        @param facetList Output vector of the Voronoi facets.
        @param facetCenters Output vector of the Voronoi facets center points.
        """

    def initDelaunay(self, rect) -> None:
        """
        @brief Creates a new empty Delaunay subdivision

        @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
        """

    @overload
    def insert(self, pt) -> retval:
        """
        @brief Insert a single point into a Delaunay triangulation.

        @param pt Point to insert.  The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. @returns the ID of the point.  @note If the point is outside of the triangulation specified rect a runtime error is raised.
        """

    @overload
    def insert(self, ptvec) -> None:
        """
        @brief Insert multiple points into a Delaunay triangulation.

        @param ptvec Points to insert.  The function inserts a vector of points into a subdivision and modifies the subdivision topology appropriately.
        """

    def locate(self, pt) -> tuple[retval, edge, vertex]:
        """
        @brief Returns the location of a point within a Delaunay triangulation.

        @param pt Point to locate.
        @param edge Output edge that the point belongs to or is located to the right of it.
        @param vertex Optional output vertex the input point coincides with.  The function locates the input point within the subdivision and gives one of the triangle edges or vertices.  @returns an integer which specify one of the following five cases for point location: -  The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of edges of the facet. -  The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge. -  The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and vertex will contain a pointer to the vertex. -  The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT and no pointers are filled. -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error processing mode is selected, #PTLOC_ERROR is returned.
        """

    def nextEdge(self, edge) -> retval:
        """
        @brief Returns next edge around the edge origin.

        @param edge Subdivision edge ID.  @returns an integer which is next edge ID around the edge origin: eOnext on the picture above if e is the input edge).
        """

    def rotateEdge(self, edge, rotate) -> retval:
        """
        @brief Returns another edge of the same quad-edge.

        @param edge Subdivision edge ID.
        @param rotate Parameter specifying which of the edges of the same quad-edge as the input one to return. The following values are possible: -   0 - the input edge ( e on the picture below if e is the input edge) -   1 - the rotated edge ( eRot ) -   2 - the reversed edge (reversed e (in green)) -   3 - the reversed rotated edge (reversed eRot (in green))  @returns one of the edges ID of the same quad-edge as the input edge.
        """

    def symEdge(self, edge) -> retval:
        """"""

class ThinPlateSplineShapeTransformer(ShapeTransformer):
    def getRegularizationParameter(self) -> retval:
        """"""

    def setRegularizationParameter(self, beta) -> None:
        """
        @brief Set the regularization parameter for relaxing the exact interpolation requirements of the TPS
        algorithm.

        @param beta value of the regularization parameter.
        """

class TickMeter(builtins.object):
    def getAvgTimeMilli(self) -> retval:
        """"""

    def getAvgTimeSec(self) -> retval:
        """"""

    def getCounter(self) -> retval:
        """"""

    def getFPS(self) -> retval:
        """"""

    def getTimeMicro(self) -> retval:
        """"""

    def getTimeMilli(self) -> retval:
        """"""

    def getTimeSec(self) -> retval:
        """"""

    def getTimeTicks(self) -> retval:
        """"""

    def reset(self) -> None:
        """"""

    def start(self) -> None:
        """"""

    def stop(self) -> None:
        """"""

class Tonemap(Algorithm):
    def getGamma(self) -> retval:
        """"""

    def process(self, src, dst=...) -> dst:
        """
        @brief Tonemaps image

        @param src source image - CV_32FC3 Mat (float 32 bits 3 channels)
        @param dst destination image - CV_32FC3 Mat with values in [0, 1] range
        """

    def setGamma(self, gamma) -> None:
        """"""

class TonemapDrago(Tonemap):
    def getBias(self) -> retval:
        """"""

    def getSaturation(self) -> retval:
        """"""

    def setBias(self, bias) -> None:
        """"""

    def setSaturation(self, saturation) -> None:
        """"""

class TonemapMantiuk(Tonemap):
    def getSaturation(self) -> retval:
        """"""

    def getScale(self) -> retval:
        """"""

    def setSaturation(self, saturation) -> None:
        """"""

    def setScale(self, scale) -> None:
        """"""

class TonemapReinhard(Tonemap):
    def getColorAdaptation(self) -> retval:
        """"""

    def getIntensity(self) -> retval:
        """"""

    def getLightAdaptation(self) -> retval:
        """"""

    def setColorAdaptation(self, color_adapt) -> None:
        """"""

    def setIntensity(self, intensity) -> None:
        """"""

    def setLightAdaptation(self, light_adapt) -> None:
        """"""

class Tracker(builtins.object):
    def init(self, image, boundingBox) -> None:
        """
        @brief Initialize the tracker with a known bounding box that surrounded the target
        @param image The initial frame
        @param boundingBox The initial bounding box
        """

    def update(self, image) -> tuple[retval, boundingBox]:
        """
        @brief Update the tracker, find the new most likely bounding box for the target
        @param image The current frame
        @param boundingBox The bounding box that represent the new target location, if true was returned, not modified otherwise  @return True means that target was located and false means that tracker cannot locate target in current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight)
        """

class TrackerCSRT(Tracker):
    def setInitialMask(self, mask) -> None:
        """"""

    def create(self, parameters=...) -> retval:
        """
        @brief Create CSRT tracker instance
        @param parameters CSRT parameters TrackerCSRT::Params
        """

TrackerCSRT_Params = Params

class TrackerDaSiamRPN(Tracker):
    def getTrackingScore(self) -> retval:
        """
        @brief Return tracking score
        """

    def create(self, parameters=...) -> retval:
        """
        @brief Constructor
        @param parameters DaSiamRPN parameters TrackerDaSiamRPN::Params
        """

TrackerDaSiamRPN_Params = Params

class TrackerGOTURN(Tracker):
    def create(self, parameters=...) -> retval:
        """
        @brief Constructor
        @param parameters GOTURN parameters TrackerGOTURN::Params
        """

TrackerGOTURN_Params = Params

class TrackerKCF(Tracker):
    def create(self, parameters=...) -> retval:
        """
        @brief Create KCF tracker instance
        @param parameters KCF parameters TrackerKCF::Params
        """

TrackerKCF_Params = Params

class TrackerMIL(Tracker):
    def create(self, parameters=...) -> retval:
        """
        @brief Create MIL tracker instance
        *  @param parameters MIL parameters TrackerMIL::Params
        """

TrackerMIL_Params = Params

class TrackerNano(Tracker):
    def getTrackingScore(self) -> retval:
        """
        @brief Return tracking score
        """

    def create(self, parameters=...) -> retval:
        """
        @brief Constructor
        @param parameters NanoTrack parameters TrackerNano::Params
        """

TrackerNano_Params = Params

class UMat(builtins.object):
    def get(self) -> retval:
        """"""

    def handle(self, accessFlags) -> retval:
        """"""

    def isContinuous(self) -> retval:
        """"""

    def isSubmatrix(self) -> retval:
        """"""

    def context(self) -> retval:
        """"""

    def queue(self) -> retval:
        """"""

class UsacParams(builtins.object): ...

class VariationalRefinement(DenseOpticalFlow):
    def calcUV(self, I0, I1, flow_u, flow_v) -> tuple[flow_u, flow_v]:
        """
        @brief @ref calc function overload to handle separate horizontal (u) and vertical (v) flow components
        (to avoid extra splits/merges)
        """

    def getAlpha(self) -> retval:
        """
        @brief Weight of the smoothness term
        @see setAlpha
        """

    def getDelta(self) -> retval:
        """
        @brief Weight of the color constancy term
        @see setDelta
        """

    def getFixedPointIterations(self) -> retval:
        """
        @brief Number of outer (fixed-point) iterations in the minimization procedure.
        @see setFixedPointIterations
        """

    def getGamma(self) -> retval:
        """
        @brief Weight of the gradient constancy term
        @see setGamma
        """

    def getOmega(self) -> retval:
        """
        @brief Relaxation factor in SOR
        @see setOmega
        """

    def getSorIterations(self) -> retval:
        """
        @brief Number of inner successive over-relaxation (SOR) iterations
        in the minimization procedure to solve the respective linear system.
        @see setSorIterations
        """

    def setAlpha(self, val) -> None:
        """
        @copybrief getAlpha @see getAlpha
        """

    def setDelta(self, val) -> None:
        """
        @copybrief getDelta @see getDelta
        """

    def setFixedPointIterations(self, val) -> None:
        """
        @copybrief getFixedPointIterations @see getFixedPointIterations
        """

    def setGamma(self, val) -> None:
        """
        @copybrief getGamma @see getGamma
        """

    def setOmega(self, val) -> None:
        """
        @copybrief getOmega @see getOmega
        """

    def setSorIterations(self, val) -> None:
        """
        @copybrief getSorIterations @see getSorIterations
        """

    def create(self) -> retval:
        """
        @brief Creates an instance of VariationalRefinement
        """

class VideoCapture(builtins.object):
    def get(self, propId) -> retval:
        """
        @brief Returns the specified VideoCapture property

        @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) or one from @ref videoio_flags_others @return Value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoCapture instance.  @note Reading / writing properties involves many layers. Some unexpected result might happens along this chain. @code{.txt} VideoCapture -> API Backend -> Operating System -> Device Driver -> Device Hardware @endcode The returned value might be different from what really used by the device or it could be encoded using device dependent rules (eg. steps or percentage). Effective behaviour depends from device driver and API Backend
        """

    def getBackendName(self) -> retval:
        """
        @brief Returns used backend API name

        @note Stream should be opened.
        """

    def getExceptionMode(self) -> retval:
        """"""

    def grab(self) -> retval:
        """
        @brief Grabs the next frame from video file or capturing device.

        @return `true` (non-zero) in the case of success.

        The method/function grabs the next frame from video file or camera and returns true (non-zero) in
        the case of success.

        The primary use of the function is in multi-camera environments, especially when the cameras do not
        have hardware synchronization. That is, you call VideoCapture::grab() for each camera and after that
        call the slower method VideoCapture::retrieve() to decode and get frame from each camera. This way
        the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames
        from different cameras will be closer in time.

        Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the
        correct way of retrieving data from it is to call VideoCapture::grab() first and then call
        VideoCapture::retrieve() one or more times with different values of the channel parameter.

        @ref tutorial_kinect_openni
        """

    def isOpened(self) -> retval:
        """
        @brief Returns true if video capturing has been initialized already.

        If the previous call to VideoCapture constructor or VideoCapture::open() succeeded, the method returns
        true.
        """

    @overload
    def open(self, filename, apiPreference=...) -> retval:
        """
        @brief  Opens a video file or a capturing device or an IP video stream for video capturing.

        @overload

        Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY)
        @return `true` if the file has been successfully opened

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    @overload
    def open(self, filename, apiPreference, params) -> retval:
        """
        @brief  Opens a video file or a capturing device or an IP video stream for video capturing with API Preference and parameters

        @overload

        The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`.
        See cv::VideoCaptureProperties

        @return `true` if the file has been successfully opened

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    @overload
    def open(self, index, apiPreference=...) -> retval:
        """
        @brief  Opens a camera for video capturing

        @overload

        Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY)
        @return `true` if the camera has been successfully opened.

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    @overload
    def open(self, index, apiPreference, params) -> retval:
        """
        @brief  Opens a camera for video capturing with API Preference and parameters

        @overload

        The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`.
        See cv::VideoCaptureProperties

        @return `true` if the camera has been successfully opened.

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    def read(self, image=...) -> tuple[retval, image]:
        """
        @brief Grabs, decodes and returns the next video frame.

        @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty. @return `false` if no frames has been grabbed  The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the most convenient method for reading video files or capturing data from decode and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns empty image (with %cv::Mat, test it with Mat::empty()).  @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy.
        """

    def release(self) -> None:
        """
        @brief Closes video file or capturing device.

        The method is automatically called by subsequent VideoCapture::open and by VideoCapture
        destructor.

        The C function also deallocates memory and clears \*capture pointer.
        """

    def retrieve(self, image=..., flag=...) -> tuple[retval, image]:
        """
        @brief Decodes and returns the grabbed video frame.

        @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty.
        @param flag it could be a frame index or a driver specific flag @return `false` if no frames has been grabbed  The method decodes and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns an empty image (with %cv::Mat, test it with Mat::empty()).  @sa read()  @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy.
        """

    def set(self, propId, value) -> retval:
        """
        @brief Sets a property in the VideoCapture.

        @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) or one from @ref videoio_flags_others
        @param value Value of the property. @return `true` if the property is supported by backend used by the VideoCapture instance. @note Even if it returns `true` this doesn't ensure that the property value has been accepted by the capture device. See note in VideoCapture::get()
        """

    def setExceptionMode(self, enable) -> None:
        """
        Switches exceptions mode
        *
        * methods raise exceptions if not successful instead of returning an error code
        """

    def waitAny(self, streams, timeoutNs=...) -> tuple[retval, readyIndex]:
        """
        @brief Wait for ready frames from VideoCapture.

        @param streams input video streams
        @param readyIndex stream indexes with grabbed frames (ready to use .retrieve() to fetch actual frame)
        @param timeoutNs number of nanoseconds (0 - infinite) @return `true` if streamReady is not empty  @throws Exception %Exception on stream errors (check .isOpened() to filter out malformed streams) or VideoCapture type is not supported  The primary use of the function is in multi-camera environments. The method fills the ready state vector, grabs video frame, if camera is ready.  After this call use VideoCapture::retrieve() to decode and fetch frame data.
        """

class VideoWriter(builtins.object):
    def get(self, propId) -> retval:
        """
        @brief Returns the specified VideoWriter property

        @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY) or one of @ref videoio_flags_others  @return Value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoWriter instance.
        """

    def getBackendName(self) -> retval:
        """
        @brief Returns used backend API name

        @note Stream should be opened.
        """

    def isOpened(self) -> retval:
        """
        @brief Returns true if video writer has been successfully initialized.
        """

    @overload
    def open(self, filename, fourcc, fps, frameSize, isColor=...) -> retval:
        """
        @brief Initializes or reinitializes video writer.

        The method opens video writer. Parameters are the same as in the constructor
        VideoWriter::VideoWriter.
        @return `true` if video writer has been successfully initialized

        The method first calls VideoWriter::release to close the already opened file.
        """

    @overload
    def open(self, filename, apiPreference, fourcc, fps, frameSize, isColor=...) -> retval:
        """
        @overload
        """

    @overload
    def open(self, filename, fourcc, fps, frameSize, params) -> retval:
        """
        @overload
        """

    @overload
    def open(self, filename, apiPreference, fourcc, fps, frameSize, params) -> retval:
        """
        @overload
        """

    def release(self) -> None:
        """
        @brief Closes the video writer.

        The method is automatically called by subsequent VideoWriter::open and by the VideoWriter
        destructor.
        """

    def set(self, propId, value) -> retval:
        """
        @brief Sets a property in the VideoWriter.

        @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY) or one of @ref videoio_flags_others
        @param value Value of the property. @return  `true` if the property is supported by the backend used by the VideoWriter instance.
        """

    def write(self, image) -> None:
        """
        @brief Writes the next video frame

        @param image The written frame. In general, color images are expected in BGR format.  The function/method writes the specified image to video file. It must have the same size as has been specified when opening the video writer.
        """

    def fourcc(self, c1, c2, c3, c4) -> retval:
        """
        @brief Concatenates 4 chars to a fourcc code

        @return a fourcc code

        This static method constructs the fourcc code of the codec to be used in the constructor
        VideoWriter::VideoWriter or VideoWriter::open.
        """

class WarperCreator(builtins.object): ...

aruco_ArucoDetector = aruco.ArucoDetector
aruco_Board = aruco.Board
aruco_CharucoBoard = aruco.CharucoBoard
aruco_CharucoDetector = aruco.CharucoDetector
aruco_CharucoParameters = aruco.CharucoParameters
aruco_DetectorParameters = aruco.DetectorParameters
aruco_Dictionary = aruco.Dictionary
aruco_EstimateParameters = aruco.EstimateParameters
aruco_GridBoard = aruco.GridBoard
aruco_RefineParameters = aruco.RefineParameters
barcode_BarcodeDetector = barcode.BarcodeDetector
bgsegm_BackgroundSubtractorCNT = bgsegm.BackgroundSubtractorCNT
bgsegm_BackgroundSubtractorGMG = bgsegm.BackgroundSubtractorGMG
bgsegm_BackgroundSubtractorGSOC = bgsegm.BackgroundSubtractorGSOC
bgsegm_BackgroundSubtractorLSBP = bgsegm.BackgroundSubtractorLSBP
bgsegm_BackgroundSubtractorLSBPDesc = bgsegm.BackgroundSubtractorLSBPDesc
bgsegm_BackgroundSubtractorMOG = bgsegm.BackgroundSubtractorMOG
bgsegm_SyntheticSequenceGenerator = bgsegm.SyntheticSequenceGenerator
bioinspired_Retina = bioinspired.Retina
bioinspired_RetinaFastToneMapping = bioinspired.RetinaFastToneMapping
bioinspired_TransientAreasSegmentationModule = bioinspired.TransientAreasSegmentationModule
ccm_ColorCorrectionModel = ccm.ColorCorrectionModel
colored_kinfu_ColoredKinFu = colored.ColoredKinFu
colored_kinfu_Params = colored.Params
cuda_BufferPool = cuda.BufferPool
cuda_DeviceInfo = cuda.DeviceInfo
cuda_Event = cuda.Event
cuda_GpuData = cuda.GpuData
cuda_GpuMat = cuda.GpuMat
cuda_GpuMatND = cuda.GpuMatND
cuda_GpuMat_Allocator = cuda.GpuMat.Allocator
cuda_HostMem = cuda.HostMem
cuda_SURF_CUDA = cuda.SURF_CUDA
cuda_Stream = cuda.Stream
cuda_TargetArchs = cuda.TargetArchs
detail_AffineBasedEstimator = detail.AffineBasedEstimator
detail_AffineBestOf2NearestMatcher = detail.AffineBestOf2NearestMatcher
detail_BestOf2NearestMatcher = detail.BestOf2NearestMatcher
detail_BestOf2NearestRangeMatcher = detail.BestOf2NearestRangeMatcher
detail_Blender = detail.Blender
detail_BlocksChannelsCompensator = detail.BlocksChannelsCompensator
detail_BlocksCompensator = detail.BlocksCompensator
detail_BlocksGainCompensator = detail.BlocksGainCompensator
detail_BundleAdjusterAffine = detail.BundleAdjusterAffine
detail_BundleAdjusterAffinePartial = detail.BundleAdjusterAffinePartial
detail_BundleAdjusterBase = detail.BundleAdjusterBase
detail_BundleAdjusterRay = detail.BundleAdjusterRay
detail_BundleAdjusterReproj = detail.BundleAdjusterReproj
detail_CameraParams = detail.CameraParams
detail_ChannelsCompensator = detail.ChannelsCompensator
detail_DpSeamFinder = detail.DpSeamFinder
detail_Estimator = detail.Estimator
detail_ExposureCompensator = detail.ExposureCompensator
detail_FeatherBlender = detail.FeatherBlender
detail_FeaturesMatcher = detail.FeaturesMatcher
detail_GainCompensator = detail.GainCompensator
detail_GraphCutSeamFinder = detail.GraphCutSeamFinder
detail_HomographyBasedEstimator = detail.HomographyBasedEstimator
detail_ImageFeatures = detail.ImageFeatures
detail_MatchesInfo = detail.MatchesInfo
detail_MultiBandBlender = detail.MultiBandBlender
detail_NoBundleAdjuster = detail.NoBundleAdjuster
detail_NoExposureCompensator = detail.NoExposureCompensator
detail_NoSeamFinder = detail.NoSeamFinder
detail_PairwiseSeamFinder = detail.PairwiseSeamFinder
detail_ProjectorBase = detail.ProjectorBase
detail_SeamFinder = detail.SeamFinder
detail_SphericalProjector = detail.SphericalProjector
detail_Timelapser = detail.Timelapser
detail_TimelapserCrop = detail.TimelapserCrop
detail_VoronoiSeamFinder = detail.VoronoiSeamFinder
dnn_ClassificationModel = dnn.ClassificationModel
dnn_DetectionModel = dnn.DetectionModel
dnn_DictValue = dnn.DictValue
dnn_KeypointsModel = dnn.KeypointsModel
dnn_Layer = dnn.Layer
dnn_Model = dnn.Model
dnn_Net = dnn.Net
dnn_SegmentationModel = dnn.SegmentationModel
dnn_TextDetectionModel = dnn.TextDetectionModel
dnn_TextDetectionModel_DB = dnn.TextDetectionModel_DB
dnn_TextDetectionModel_EAST = dnn.TextDetectionModel_EAST
dnn_TextRecognitionModel = dnn.TextRecognitionModel
dnn_superres_DnnSuperResImpl = dnn_superres.DnnSuperResImpl
dpm_DPMDetector = dpm.DPMDetector
dpm_DPMDetector_ObjectDetection = dpm.DPMDetector.ObjectDetection
dynafu_DynaFu = dynafu.DynaFu
face_BIF = face.BIF
face_BasicFaceRecognizer = face.BasicFaceRecognizer
face_EigenFaceRecognizer = face.EigenFaceRecognizer
face_FaceRecognizer = face.FaceRecognizer
face_Facemark = face.Facemark
face_FacemarkAAM = face.FacemarkAAM
face_FacemarkKazemi = face.FacemarkKazemi
face_FacemarkLBF = face.FacemarkLBF
face_FacemarkTrain = face.FacemarkTrain
face_FisherFaceRecognizer = face.FisherFaceRecognizer
face_LBPHFaceRecognizer = face.LBPHFaceRecognizer
face_MACE = face.MACE
face_PredictCollector = face.PredictCollector
face_StandardCollector = face.StandardCollector
flann_Index = flann.Index
gapi_GNetPackage = gapi.GNetPackage
gapi_GNetParam = gapi.GNetParam
gapi_ie_PyParams = gapi.ie.PyParams
gapi_onnx_PyParams = gapi.onnx.PyParams
gapi_streaming_queue_capacity = gapi.streaming.queue_capacity
gapi_wip_GOutputs = gapi.wip.GOutputs
gapi_wip_IStreamSource = gapi.wip.IStreamSource
gapi_wip_draw_Circle = gapi.wip.draw.Circle
gapi_wip_draw_Image = gapi.wip.draw.Image
gapi_wip_draw_Line = gapi.wip.draw.Line
gapi_wip_draw_Mosaic = gapi.wip.draw.Mosaic
gapi_wip_draw_Poly = gapi.wip.draw.Poly
gapi_wip_draw_Rect = gapi.wip.draw.Rect
gapi_wip_draw_Text = gapi.wip.draw.Text
gapi_wip_gst_GStreamerPipeline = gapi.wip.gst.GStreamerPipeline
hfs_HfsSegment = hfs.HfsSegment
img_hash_AverageHash = img.AverageHash
img_hash_BlockMeanHash = img.BlockMeanHash
img_hash_ColorMomentHash = img.ColorMomentHash
img_hash_ImgHashBase = img.ImgHashBase
img_hash_MarrHildrethHash = img.MarrHildrethHash
img_hash_PHash = img.PHash
img_hash_RadialVarianceHash = img.RadialVarianceHash
kinfu_KinFu = kinfu.KinFu
kinfu_Params = kinfu.Params
kinfu_Volume = kinfu.Volume
kinfu_VolumeParams = kinfu.VolumeParams
kinfu_detail_PoseGraph = kinfu.detail.PoseGraph
large_kinfu_LargeKinfu = large.LargeKinfu
large_kinfu_Params = large.Params
legacy_MultiTracker = legacy.MultiTracker
legacy_Tracker = legacy.Tracker
legacy_TrackerBoosting = legacy.TrackerBoosting
legacy_TrackerCSRT = legacy.TrackerCSRT
legacy_TrackerKCF = legacy.TrackerKCF
legacy_TrackerMIL = legacy.TrackerMIL
legacy_TrackerMOSSE = legacy.TrackerMOSSE
legacy_TrackerMedianFlow = legacy.TrackerMedianFlow
legacy_TrackerTLD = legacy.TrackerTLD
line_descriptor_BinaryDescriptor = line.BinaryDescriptor
line_descriptor_BinaryDescriptorMatcher = line.BinaryDescriptorMatcher
line_descriptor_DrawLinesMatchesFlags = line.DrawLinesMatchesFlags
line_descriptor_KeyLine = line.KeyLine
line_descriptor_LSDDetector = line.LSDDetector
line_descriptor_LSDParam = line.LSDParam
linemod_ColorGradient = linemod.ColorGradient
linemod_DepthNormal = linemod.DepthNormal
linemod_Detector = linemod.Detector
linemod_Feature = linemod.Feature
linemod_Match = linemod.Match
linemod_Modality = linemod.Modality
linemod_QuantizedPyramid = linemod.QuantizedPyramid
linemod_Template = linemod.Template
mcc_CChecker = mcc.CChecker
mcc_CCheckerDetector = mcc.CCheckerDetector
mcc_CCheckerDraw = mcc.CCheckerDraw
mcc_DetectorParameters = mcc.DetectorParameters
ml_ANN_MLP = ml.ANN_MLP
ml_Boost = ml.Boost
ml_DTrees = ml.DTrees
ml_EM = ml.EM
ml_KNearest = ml.KNearest
ml_LogisticRegression = ml.LogisticRegression
ml_NormalBayesClassifier = ml.NormalBayesClassifier
ml_ParamGrid = ml.ParamGrid
ml_RTrees = ml.RTrees
ml_SVM = ml.SVM
ml_SVMSGD = ml.SVMSGD
ml_StatModel = ml.StatModel
ml_TrainData = ml.TrainData
ocl_Device = ocl.Device
ocl_OpenCLExecutionContext = ocl.OpenCLExecutionContext
optflow_DenseRLOFOpticalFlow = optflow.DenseRLOFOpticalFlow
optflow_DualTVL1OpticalFlow = optflow.DualTVL1OpticalFlow
optflow_GPCDetails = optflow.GPCDetails
optflow_GPCPatchDescriptor = optflow.GPCPatchDescriptor
optflow_GPCPatchSample = optflow.GPCPatchSample
optflow_GPCTrainingSamples = optflow.GPCTrainingSamples
optflow_GPCTree = optflow.GPCTree
optflow_OpticalFlowPCAFlow = optflow.OpticalFlowPCAFlow
optflow_PCAPrior = optflow.PCAPrior
optflow_RLOFOpticalFlowParameter = optflow.RLOFOpticalFlowParameter
optflow_SparseRLOFOpticalFlow = optflow.SparseRLOFOpticalFlow
phase_unwrapping_HistogramPhaseUnwrapping = phase.HistogramPhaseUnwrapping
phase_unwrapping_HistogramPhaseUnwrapping_Params = phase.Params
phase_unwrapping_PhaseUnwrapping = phase.PhaseUnwrapping
plot_Plot2d = plot.Plot2d
ppf_match_3d_ICP = ppf.ICP
ppf_match_3d_PPF3DDetector = ppf.PPF3DDetector
ppf_match_3d_Pose3D = ppf.Pose3D
ppf_match_3d_PoseCluster3D = ppf.PoseCluster3D
quality_QualityBRISQUE = quality.QualityBRISQUE
quality_QualityBase = quality.QualityBase
quality_QualityGMSD = quality.QualityGMSD
quality_QualityMSE = quality.QualityMSE
quality_QualityPSNR = quality.QualityPSNR
quality_QualitySSIM = quality.QualitySSIM
rapid_GOSTracker = rapid.GOSTracker
rapid_OLSTracker = rapid.OLSTracker
rapid_Rapid = rapid.Rapid
rapid_Tracker = rapid.Tracker
reg_Map = reg.Map
reg_MapAffine = reg.MapAffine
reg_MapProjec = reg.MapProjec
reg_MapShift = reg.MapShift
reg_MapTypeCaster = reg.MapTypeCaster
reg_Mapper = reg.Mapper
reg_MapperGradAffine = reg.MapperGradAffine
reg_MapperGradEuclid = reg.MapperGradEuclid
reg_MapperGradProj = reg.MapperGradProj
reg_MapperGradShift = reg.MapperGradShift
reg_MapperGradSimilar = reg.MapperGradSimilar
reg_MapperPyramid = reg.MapperPyramid
rgbd_DepthCleaner = rgbd.DepthCleaner
rgbd_FastICPOdometry = rgbd.FastICPOdometry
rgbd_ICPOdometry = rgbd.ICPOdometry
rgbd_Odometry = rgbd.Odometry
rgbd_OdometryFrame = rgbd.OdometryFrame
rgbd_RgbdFrame = rgbd.RgbdFrame
rgbd_RgbdICPOdometry = rgbd.RgbdICPOdometry
rgbd_RgbdNormals = rgbd.RgbdNormals
rgbd_RgbdOdometry = rgbd.RgbdOdometry
rgbd_RgbdPlane = rgbd.RgbdPlane
saliency_MotionSaliency = saliency.MotionSaliency
saliency_MotionSaliencyBinWangApr2014 = saliency.MotionSaliencyBinWangApr2014
saliency_Objectness = saliency.Objectness
saliency_ObjectnessBING = saliency.ObjectnessBING
saliency_Saliency = saliency.Saliency
saliency_StaticSaliency = saliency.StaticSaliency
saliency_StaticSaliencyFineGrained = saliency.StaticSaliencyFineGrained
saliency_StaticSaliencySpectralResidual = saliency.StaticSaliencySpectralResidual
segmentation_IntelligentScissorsMB = segmentation.IntelligentScissorsMB
stereo_MatchQuasiDense = stereo.MatchQuasiDense
stereo_PropagationParameters = stereo.PropagationParameters
stereo_QuasiDenseStereo = stereo.QuasiDenseStereo
structured_light_GrayCodePattern = structured.GrayCodePattern
structured_light_SinusoidalPattern = structured.SinusoidalPattern
structured_light_SinusoidalPattern_Params = structured.Params
structured_light_StructuredLightPattern = structured.StructuredLightPattern
text_BaseOCR = text.BaseOCR
text_ERFilter = text.ERFilter
text_ERFilter_Callback = text.ERFilter.Callback
text_OCRBeamSearchDecoder = text.OCRBeamSearchDecoder
text_OCRBeamSearchDecoder_ClassifierCallback = text.OCRBeamSearchDecoder.ClassifierCallback
text_OCRHMMDecoder = text.OCRHMMDecoder
text_OCRHMMDecoder_ClassifierCallback = text.OCRHMMDecoder.ClassifierCallback
text_OCRTesseract = text.OCRTesseract
text_TextDetector = text.TextDetector
text_TextDetectorCNN = text.TextDetectorCNN
utils_ClassWithKeywordProperties = utils.ClassWithKeywordProperties
utils_nested_ExportClassName = utils.nested.ExportClassName
utils_nested_ExportClassName_Params = utils.nested.ExportClassName.Params
wechat_qrcode_WeChatQRCode = wechat.WeChatQRCode
xfeatures2d_AffineFeature2D = xfeatures2d.AffineFeature2D
xfeatures2d_BEBLID = xfeatures2d.BEBLID
xfeatures2d_BoostDesc = xfeatures2d.BoostDesc
xfeatures2d_BriefDescriptorExtractor = xfeatures2d.BriefDescriptorExtractor
xfeatures2d_DAISY = xfeatures2d.DAISY
xfeatures2d_FREAK = xfeatures2d.FREAK
xfeatures2d_HarrisLaplaceFeatureDetector = xfeatures2d.HarrisLaplaceFeatureDetector
xfeatures2d_LATCH = xfeatures2d.LATCH
xfeatures2d_LUCID = xfeatures2d.LUCID
xfeatures2d_MSDDetector = xfeatures2d.MSDDetector
xfeatures2d_PCTSignatures = xfeatures2d.PCTSignatures
xfeatures2d_PCTSignaturesSQFD = xfeatures2d.PCTSignaturesSQFD
xfeatures2d_SURF = xfeatures2d.SURF
xfeatures2d_StarDetector = xfeatures2d.StarDetector
xfeatures2d_TBMR = xfeatures2d.TBMR
xfeatures2d_TEBLID = xfeatures2d.TEBLID
xfeatures2d_VGG = xfeatures2d.VGG
ximgproc_AdaptiveManifoldFilter = ximgproc.AdaptiveManifoldFilter
ximgproc_ContourFitting = ximgproc.ContourFitting
ximgproc_DTFilter = ximgproc.DTFilter
ximgproc_DisparityFilter = ximgproc.DisparityFilter
ximgproc_DisparityWLSFilter = ximgproc.DisparityWLSFilter
ximgproc_EdgeAwareInterpolator = ximgproc.EdgeAwareInterpolator
ximgproc_EdgeBoxes = ximgproc.EdgeBoxes
ximgproc_EdgeDrawing = ximgproc.EdgeDrawing
ximgproc_EdgeDrawing_Params = ximgproc.EdgeDrawing.Params
ximgproc_FastBilateralSolverFilter = ximgproc.FastBilateralSolverFilter
ximgproc_FastGlobalSmootherFilter = ximgproc.FastGlobalSmootherFilter
ximgproc_FastLineDetector = ximgproc.FastLineDetector
ximgproc_GuidedFilter = ximgproc.GuidedFilter
ximgproc_RFFeatureGetter = ximgproc.RFFeatureGetter
ximgproc_RICInterpolator = ximgproc.RICInterpolator
ximgproc_RidgeDetectionFilter = ximgproc.RidgeDetectionFilter
ximgproc_ScanSegment = ximgproc.ScanSegment
ximgproc_SparseMatchInterpolator = ximgproc.SparseMatchInterpolator
ximgproc_StructuredEdgeDetection = ximgproc.StructuredEdgeDetection
ximgproc_SuperpixelLSC = ximgproc.SuperpixelLSC
ximgproc_SuperpixelSEEDS = ximgproc.SuperpixelSEEDS
ximgproc_SuperpixelSLIC = ximgproc.SuperpixelSLIC
ximgproc_segmentation_GraphSegmentation = ximgproc.segmentation.GraphSegmentation
ximgproc_segmentation_SelectiveSearchSegmentation = ximgproc.segmentation.SelectiveSearchSegmentation
ximgproc_segmentation_SelectiveSearchSegmentationStrategy = ximgproc.segmentation.SelectiveSearchSegmentationStrategy
ximgproc_segmentation_SelectiveSearchSegmentationStrategyColor = ximgproc.segmentation.SelectiveSearchSegmentationStrategyColor
ximgproc_segmentation_SelectiveSearchSegmentationStrategyFill = ximgproc.segmentation.SelectiveSearchSegmentationStrategyFill
ximgproc_segmentation_SelectiveSearchSegmentationStrategyMultiple = ximgproc.segmentation.SelectiveSearchSegmentationStrategyMultiple
ximgproc_segmentation_SelectiveSearchSegmentationStrategySize = ximgproc.segmentation.SelectiveSearchSegmentationStrategySize
ximgproc_segmentation_SelectiveSearchSegmentationStrategyTexture = ximgproc.segmentation.SelectiveSearchSegmentationStrategyTexture
xphoto_GrayworldWB = xphoto.GrayworldWB
xphoto_LearningBasedWB = xphoto.LearningBasedWB
xphoto_SimpleWB = xphoto.SimpleWB
xphoto_TonemapDurand = xphoto.TonemapDurand
xphoto_WhiteBalancer = xphoto.WhiteBalancer
