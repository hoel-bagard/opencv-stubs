import builtins
from typing import Any, TypeAlias, overload

dst: TypeAlias = Any
retval: TypeAlias = Any

class AffineBasedEstimator(Estimator):
    ...


class AffineBestOf2NearestMatcher(BestOf2NearestMatcher):
    ...


class BestOf2NearestMatcher(FeaturesMatcher):
    def collectGarbage(self) -> None:
        """"""

    def create(self, try_use_gpu = ..., match_conf = ..., num_matches_thresh1 = ..., num_matches_thresh2 = ..., matches_confindece_thresh = ...) -> retval:
        """"""


class BestOf2NearestRangeMatcher(BestOf2NearestMatcher):
    ...


class Blender(builtins.object):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """
        @brief Blends and returns the final pano.

        @param dst Final pano
        @param dst_mask Final pano mask
        """

    def feed(self, img, mask, tl) -> None:
        """
        @brief Processes the image.

        @param img Source image
        @param mask Source image mask
        @param tl Source image top-left corners
        """

    def prepare(self, corners, sizes) -> None:
        """
        @brief Prepares the blender for blending.

        @param corners Source images top-left corners
        @param sizes Source image sizes
        """

    @overload
    def prepare(self, dst_roi) -> None:
        """
        @overload
        """

    def createDefault(self, type, try_gpu = ...) -> retval:
        """"""


class BlocksChannelsCompensator(BlocksCompensator):
    ...


class BlocksCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getBlockSize(self) -> retval:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getNrGainsFilteringIterations(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setBlockSize(self, width, height) -> None:
        """"""

    def setBlockSize(self, size) -> None:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setNrGainsFilteringIterations(self, nr_iterations) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class BlocksGainCompensator(BlocksCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""


class BundleAdjusterAffine(BundleAdjusterBase):
    ...


class BundleAdjusterAffinePartial(BundleAdjusterBase):
    ...


class BundleAdjusterBase(Estimator):
    def confThresh(self) -> retval:
        """"""

    def refinementMask(self) -> retval:
        """"""

    def setConfThresh(self, conf_thresh) -> None:
        """"""

    def setRefinementMask(self, mask) -> None:
        """"""

    def setTermCriteria(self, term_criteria) -> None:
        """"""

    def termCriteria(self) -> retval:
        """"""


class BundleAdjusterRay(BundleAdjusterBase):
    ...


class BundleAdjusterReproj(BundleAdjusterBase):
    ...


class CameraParams(builtins.object):
    def K(self) -> retval:
        """


        """


class ChannelsCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class DpSeamFinder(SeamFinder):
    def setCostFunction(self, val) -> None:
        """"""


class Estimator(builtins.object):
    def apply(self, features, pairwise_matches, cameras) -> tuple[retval, cameras]:
        """
        @brief Estimates camera parameters.

        @param features Features of images
        @param pairwise_matches Pairwise matches of images
        @param cameras Estimated camera parameters @return True in case of success, false otherwise
        """


class ExposureCompensator(builtins.object):
    def apply(self, index, corner, image, mask) -> image:
        """
        @brief Compensate exposure in the specified image.

        @param index Image index
        @param corner Image top-left corner
        @param image Image to process
        @param mask Image mask
        """

    def feed(self, corners, images, masks) -> None:
        """
        @param corners Source image top-left corners
        @param images Source images
        @param masks Image masks to update (second value in pair specifies the value which should be used to detect where image is)
        """

    def getMatGains(self, arg1 = ...) -> arg1:
        """"""

    def getUpdateGain(self) -> retval:
        """"""

    def setMatGains(self, arg1) -> None:
        """"""

    def setUpdateGain(self, b) -> None:
        """"""

    def createDefault(self, type) -> retval:
        """"""


class FeatherBlender(Blender):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """"""

    def createWeightMaps(self, masks, corners, weight_maps) -> tuple[retval, weight_maps]:
        """"""

    def feed(self, img, mask, tl) -> None:
        """"""

    def prepare(self, dst_roi) -> None:
        """"""

    def setSharpness(self, val) -> None:
        """"""

    def sharpness(self) -> retval:
        """"""


class FeaturesMatcher(builtins.object):
    @overload
    def apply(self, features1, features2) -> matches_info:
        """
        @overload
        @param features1 First image features
        @param features2 Second image features
        @param matches_info Found matches
        """

    def apply2(self, features, mask = ...) -> pairwise_matches:
        """
        @brief Performs images matching.

        @param features Features of the source images
        @param pairwise_matches Found pairwise matches
        @param mask Mask indicating which image pairs must be matched  The function is parallelized with the TBB library.  @sa detail::MatchesInfo
        """

    def collectGarbage(self) -> None:
        """
        @brief Frees unused memory allocated before if there is any.
        """

    def isThreadSafe(self) -> retval:
        """
        @return True, if it's possible to use the same matcher instance in parallel, false otherwise
        """


class GainCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class GraphCutSeamFinder(builtins.object):
    def find(self, src, corners, masks) -> masks:
        """"""


class HomographyBasedEstimator(Estimator):
    ...


class ImageFeatures(builtins.object):
    def getKeypoints(self) -> retval:
        """"""


class MatchesInfo(builtins.object):
    def getInliers(self) -> retval:
        """"""

    def getMatches(self) -> retval:
        """"""


class MultiBandBlender(Blender):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """"""

    def feed(self, img, mask, tl) -> None:
        """"""

    def numBands(self) -> retval:
        """"""

    def prepare(self, dst_roi) -> None:
        """"""

    def setNumBands(self, val) -> None:
        """"""


class NoBundleAdjuster(BundleAdjusterBase):
    ...


class NoExposureCompensator(ExposureCompensator):
    def apply(self, arg1, arg2, arg3, arg4) -> arg3:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""


class NoSeamFinder(SeamFinder):
    def find(self, arg1, arg2, arg3) -> arg3:
        """"""


class PairwiseSeamFinder(SeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""


class ProjectorBase(builtins.object):
    ...


class SeamFinder(builtins.object):
    def find(self, src, corners, masks) -> masks:
        """
        @brief Estimates seams.

        @param src Source images
        @param corners Source image top-left corners
        @param masks Source image masks to update
        """

    def createDefault(self, type) -> retval:
        """"""


class SphericalProjector(ProjectorBase):
    def mapBackward(self, u, v, x, y) -> None:
        """"""

    def mapForward(self, x, y, u, v) -> None:
        """"""


class Timelapser(builtins.object):
    def getDst(self) -> retval:
        """"""

    def initialize(self, corners, sizes) -> None:
        """"""

    def process(self, img, mask, tl) -> None:
        """"""

    def createDefault(self, type) -> retval:
        """"""


class TimelapserCrop(Timelapser):
    ...


class VoronoiSeamFinder(PairwiseSeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""


def BestOf2NearestMatcher_create(try_use_gpu = ..., match_conf = ..., num_matches_thresh1 = ..., num_matches_thresh2 = ..., matches_confindece_thresh = ...) -> retval:
    """
        .
    """

def Blender_createDefault(type, try_gpu = ...) -> retval:
    """
        .
    """

def ExposureCompensator_createDefault(type) -> retval:
    """
        .
    """

def SeamFinder_createDefault(type) -> retval:
    """
        .
    """

def Timelapser_createDefault(type) -> retval:
    """
        .
    """

def calibrateRotatingCamera(Hs, K = ...) -> tuple[retval, K]:
    """
    @brief Estimates focal lengths for each given camera.

    @param features Features of images.
    @param pairwise_matches Matches between all image pairs.
    @param focals Estimated focal lengths for each camera.
    """

def computeImageFeatures(featuresFinder, images, masks = ...) -> features:
    """
    @brief

    @param featuresFinder
    @param images
    @param features
    @param masks
    """

def computeImageFeatures2(featuresFinder, image, mask = ...) -> features:
    """
    @brief

    @param featuresFinder
    @param image
    @param features
    @param mask
    """

def createLaplacePyr(img, num_levels, pyr) -> pyr:
    """
        .
    """

def createLaplacePyrGpu(img, num_levels, pyr) -> pyr:
    """
        .
    """

def createWeightMap(mask, sharpness, weight) -> weight:
    """
        .
    """

def focalsFromHomography(H, f0, f1, f0_ok, f1_ok) -> None:
    """
    @brief Tries to estimate focal lengths from the given homography under the assumption that the camera
    undergoes rotations around its centre only.

    @param H Homography.
    @param f0 Estimated focal length along X axis.
    @param f1 Estimated focal length along Y axis.
    @param f0_ok True, if f0 was estimated successfully, false otherwise.
    @param f1_ok True, if f1 was estimated successfully, false otherwise.

    See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
    by Heung-Yeung Shum and Richard Szeliski.
    """

def leaveBiggestComponent(features, pairwise_matches, conf_threshold) -> retval:
    """
        .
    """

def matchesGraphAsString(pathes, pairwise_matches, conf_threshold) -> retval:
    """
        .
    """

def normalizeUsingWeightMap(weight, src) -> src:
    """
        .
    """

def overlapRoi(tl1, tl2, sz1, sz2, roi) -> retval:
    """
        .
    """

def restoreImageFromLaplacePyr(pyr) -> pyr:
    """
        .
    """

def restoreImageFromLaplacePyrGpu(pyr) -> pyr:
    """
        .
    """

@overload
def resultRoi(corners, images) -> retval:
    """
    """

@overload
def resultRoi(corners, images) -> retval:
    """
        .
    """

def resultRoiIntersection(corners, sizes) -> retval:
    """
        .
    """

def resultTl(corners) -> retval:
    """
        .
    """

def selectRandomSubset(count, size, subset) -> None:
    """
        .
    """

def stitchingLogLevel() -> retval:
    """
        .
    """

def strip(params) -> retval:
    """
        .
    """

def waveCorrect(rmats, kind) -> rmats:
    """
    @brief Tries to make panorama more horizontal (or vertical).

    @param rmats Camera rotation matrices.
    @param kind Correction kind, see detail::WaveCorrectKind.
    """

ARG_KIND_GARRAY: int
ARG_KIND_GFRAME: int
ARG_KIND_GMAT: int
ARG_KIND_GMATP: int
ARG_KIND_GOBJREF: int
ARG_KIND_GOPAQUE: int
ARG_KIND_GSCALAR: int
ARG_KIND_OPAQUE: int
ARG_KIND_OPAQUE_VAL: int
ArgKind_GARRAY: int
ArgKind_GFRAME: int
ArgKind_GMAT: int
ArgKind_GMATP: int
ArgKind_GOBJREF: int
ArgKind_GOPAQUE: int
ArgKind_GSCALAR: int
ArgKind_OPAQUE: int
ArgKind_OPAQUE_VAL: int
BLENDER_FEATHER: int
BLENDER_MULTI_BAND: int
BLENDER_NO: int
Blender_FEATHER: int
Blender_MULTI_BAND: int
Blender_NO: int
CV_FEATURE_PARAMS_HAAR: int
CV_FEATURE_PARAMS_HOG: int
CV_FEATURE_PARAMS_LBP: int
CvFeatureParams_HAAR: int
CvFeatureParams_HOG: int
CvFeatureParams_LBP: int
DP_SEAM_FINDER_COLOR: int
DP_SEAM_FINDER_COLOR_GRAD: int
DpSeamFinder_COLOR: int
DpSeamFinder_COLOR_GRAD: int
EXPOSURE_COMPENSATOR_CHANNELS: int
EXPOSURE_COMPENSATOR_CHANNELS_BLOCKS: int
EXPOSURE_COMPENSATOR_GAIN: int
EXPOSURE_COMPENSATOR_GAIN_BLOCKS: int
EXPOSURE_COMPENSATOR_NO: int
ExposureCompensator_CHANNELS: int
ExposureCompensator_CHANNELS_BLOCKS: int
ExposureCompensator_GAIN: int
ExposureCompensator_GAIN_BLOCKS: int
ExposureCompensator_NO: int
GRAPH_CUT_SEAM_FINDER_BASE_COST_COLOR: int
GRAPH_CUT_SEAM_FINDER_BASE_COST_COLOR_GRAD: int
GraphCutSeamFinderBase_COST_COLOR: int
GraphCutSeamFinderBase_COST_COLOR_GRAD: int
OPAQUE_KIND_CV_BOOL: int
OPAQUE_KIND_CV_DOUBLE: int
OPAQUE_KIND_CV_DRAW_PRIM: int
OPAQUE_KIND_CV_FLOAT: int
OPAQUE_KIND_CV_INT: int
OPAQUE_KIND_CV_INT64: int
OPAQUE_KIND_CV_MAT: int
OPAQUE_KIND_CV_POINT: int
OPAQUE_KIND_CV_POINT2F: int
OPAQUE_KIND_CV_POINT3F: int
OPAQUE_KIND_CV_RECT: int
OPAQUE_KIND_CV_SCALAR: int
OPAQUE_KIND_CV_SIZE: int
OPAQUE_KIND_CV_STRING: int
OPAQUE_KIND_CV_UINT64: int
OPAQUE_KIND_CV_UNKNOWN: int
OpaqueKind_CV_BOOL: int
OpaqueKind_CV_DOUBLE: int
OpaqueKind_CV_DRAW_PRIM: int
OpaqueKind_CV_FLOAT: int
OpaqueKind_CV_INT: int
OpaqueKind_CV_INT64: int
OpaqueKind_CV_MAT: int
OpaqueKind_CV_POINT: int
OpaqueKind_CV_POINT2F: int
OpaqueKind_CV_POINT3F: int
OpaqueKind_CV_RECT: int
OpaqueKind_CV_SCALAR: int
OpaqueKind_CV_SIZE: int
OpaqueKind_CV_STRING: int
OpaqueKind_CV_UINT64: int
OpaqueKind_CV_UNKNOWN: int
SEAM_FINDER_DP_SEAM: int
SEAM_FINDER_NO: int
SEAM_FINDER_VORONOI_SEAM: int
SeamFinder_DP_SEAM: int
SeamFinder_NO: int
SeamFinder_VORONOI_SEAM: int
TEST_CUSTOM: int
TEST_EQ: int
TEST_GE: int
TEST_GT: int
TEST_LE: int
TEST_LT: int
TEST_NE: int
TIMELAPSER_AS_IS: int
TIMELAPSER_CROP: int
TRACKER_CONTRIB_SAMPLER_CSC_MODE_DETECT: int
TRACKER_CONTRIB_SAMPLER_CSC_MODE_INIT_NEG: int
TRACKER_CONTRIB_SAMPLER_CSC_MODE_INIT_POS: int
TRACKER_CONTRIB_SAMPLER_CSC_MODE_TRACK_NEG: int
TRACKER_CONTRIB_SAMPLER_CSC_MODE_TRACK_POS: int
TRACKER_SAMPLER_CSC_MODE_DETECT: int
TRACKER_SAMPLER_CSC_MODE_INIT_NEG: int
TRACKER_SAMPLER_CSC_MODE_INIT_POS: int
TRACKER_SAMPLER_CSC_MODE_TRACK_NEG: int
TRACKER_SAMPLER_CSC_MODE_TRACK_POS: int
TRACKER_SAMPLER_CS_MODE_CLASSIFY: int
TRACKER_SAMPLER_CS_MODE_NEGATIVE: int
TRACKER_SAMPLER_CS_MODE_POSITIVE: int
Timelapser_AS_IS: int
Timelapser_CROP: int
TrackerContribSamplerCSC_MODE_DETECT: int
TrackerContribSamplerCSC_MODE_INIT_NEG: int
TrackerContribSamplerCSC_MODE_INIT_POS: int
TrackerContribSamplerCSC_MODE_TRACK_NEG: int
TrackerContribSamplerCSC_MODE_TRACK_POS: int
TrackerSamplerCSC_MODE_DETECT: int
TrackerSamplerCSC_MODE_INIT_NEG: int
TrackerSamplerCSC_MODE_INIT_POS: int
TrackerSamplerCSC_MODE_TRACK_NEG: int
TrackerSamplerCSC_MODE_TRACK_POS: int
TrackerSamplerCS_MODE_CLASSIFY: int
TrackerSamplerCS_MODE_NEGATIVE: int
TrackerSamplerCS_MODE_POSITIVE: int
WAVE_CORRECT_AUTO: int
WAVE_CORRECT_HORIZ: int
WAVE_CORRECT_VERT: int
