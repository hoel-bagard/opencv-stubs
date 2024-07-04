import builtins
from typing import Any, Final, overload, TypeAlias

matches_info: TypeAlias = Any
rmats: TypeAlias = Any
src: TypeAlias = Any
image: TypeAlias = Any
weight: TypeAlias = Any
K: TypeAlias = Any
umv: TypeAlias = Any
pyr: TypeAlias = Any
arg3: TypeAlias = Any
arg1: TypeAlias = Any
dst_mask: TypeAlias = Any
masks: TypeAlias = Any
weight_maps: TypeAlias = Any
features: TypeAlias = Any
pairwise_matches: TypeAlias = Any
cameras: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any

class AffineBasedEstimator(Estimator): ...
class AffineBestOf2NearestMatcher(BestOf2NearestMatcher): ...

class BestOf2NearestMatcher(FeaturesMatcher):
    def collectGarbage(self) -> None:
        """"""

    def create(self, try_use_gpu=..., match_conf=..., num_matches_thresh1=..., num_matches_thresh2=..., matches_confindece_thresh=...) -> retval:
        """"""

class BestOf2NearestRangeMatcher(BestOf2NearestMatcher): ...

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

    @overload
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

    def createDefault(self, type, try_gpu=...) -> retval:
        """"""

class BlocksChannelsCompensator(BlocksCompensator): ...

class BlocksCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getBlockSize(self) -> retval:
        """"""

    def getMatGains(self, umv=...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getNrGainsFilteringIterations(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    @overload
    def setBlockSize(self, width, height) -> None:
        """"""

    @overload
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

    def getMatGains(self, umv=...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

class BundleAdjusterAffine(BundleAdjusterBase): ...
class BundleAdjusterAffinePartial(BundleAdjusterBase): ...

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

class BundleAdjusterRay(BundleAdjusterBase): ...
class BundleAdjusterReproj(BundleAdjusterBase): ...

class CameraParams(builtins.object):
    def K(self) -> retval:
        """ """

class ChannelsCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv=...) -> umv:
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

    def getMatGains(self, arg1=...) -> arg1:
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
    def apply(self, features1, features2) -> matches_info:
        """
        @overload
        @param features1 First image features
        @param features2 Second image features
        @param matches_info Found matches
        """

    def apply2(self, features, mask=...) -> pairwise_matches:
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

    def getMatGains(self, umv=...) -> umv:
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

class HomographyBasedEstimator(Estimator): ...

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

class NoBundleAdjuster(BundleAdjusterBase): ...

class NoExposureCompensator(ExposureCompensator):
    def apply(self, arg1, arg2, arg3, arg4) -> arg3:
        """"""

    def getMatGains(self, umv=...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

class NoSeamFinder(SeamFinder):
    def find(self, arg1, arg2, arg3) -> arg3:
        """"""

class PairwiseSeamFinder(SeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""

class ProjectorBase(builtins.object): ...

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

class TimelapserCrop(Timelapser): ...

class VoronoiSeamFinder(PairwiseSeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""

def BestOf2NearestMatcher_create(try_use_gpu=..., match_conf=..., num_matches_thresh1=..., num_matches_thresh2=..., matches_confindece_thresh=...) -> retval:
    """
    .
    """

def Blender_createDefault(type, try_gpu=...) -> retval:
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

def calibrateRotatingCamera(Hs, K=...) -> tuple[retval, K]:
    """
    @brief Estimates focal lengths for each given camera.

    @param features Features of images.
    @param pairwise_matches Matches between all image pairs.
    @param focals Estimated focal lengths for each camera.
    """

def computeImageFeatures(featuresFinder, images, masks=...) -> features:
    """
    @brief

    @param featuresFinder
    @param images
    @param features
    @param masks
    """

def computeImageFeatures2(featuresFinder, image, mask=...) -> features:
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
    """ """

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

ARG_KIND_GARRAY: Final[int]
ARG_KIND_GFRAME: Final[int]
ARG_KIND_GMAT: Final[int]
ARG_KIND_GMATP: Final[int]
ARG_KIND_GOBJREF: Final[int]
ARG_KIND_GOPAQUE: Final[int]
ARG_KIND_GSCALAR: Final[int]
ARG_KIND_OPAQUE: Final[int]
ARG_KIND_OPAQUE_VAL: Final[int]
ArgKind_GARRAY: Final[int]
ArgKind_GFRAME: Final[int]
ArgKind_GMAT: Final[int]
ArgKind_GMATP: Final[int]
ArgKind_GOBJREF: Final[int]
ArgKind_GOPAQUE: Final[int]
ArgKind_GSCALAR: Final[int]
ArgKind_OPAQUE: Final[int]
ArgKind_OPAQUE_VAL: Final[int]
BLENDER_FEATHER: Final[int]
BLENDER_MULTI_BAND: Final[int]
BLENDER_NO: Final[int]
Blender_FEATHER: Final[int]
Blender_MULTI_BAND: Final[int]
Blender_NO: Final[int]
CV_FEATURE_PARAMS_HAAR: Final[int]
CV_FEATURE_PARAMS_HOG: Final[int]
CV_FEATURE_PARAMS_LBP: Final[int]
CvFeatureParams_HAAR: Final[int]
CvFeatureParams_HOG: Final[int]
CvFeatureParams_LBP: Final[int]
DP_SEAM_FINDER_COLOR: Final[int]
DP_SEAM_FINDER_COLOR_GRAD: Final[int]
DpSeamFinder_COLOR: Final[int]
DpSeamFinder_COLOR_GRAD: Final[int]
EXPOSURE_COMPENSATOR_CHANNELS: Final[int]
EXPOSURE_COMPENSATOR_CHANNELS_BLOCKS: Final[int]
EXPOSURE_COMPENSATOR_GAIN: Final[int]
EXPOSURE_COMPENSATOR_GAIN_BLOCKS: Final[int]
EXPOSURE_COMPENSATOR_NO: Final[int]
ExposureCompensator_CHANNELS: Final[int]
ExposureCompensator_CHANNELS_BLOCKS: Final[int]
ExposureCompensator_GAIN: Final[int]
ExposureCompensator_GAIN_BLOCKS: Final[int]
ExposureCompensator_NO: Final[int]
GRAPH_CUT_SEAM_FINDER_BASE_COST_COLOR: Final[int]
GRAPH_CUT_SEAM_FINDER_BASE_COST_COLOR_GRAD: Final[int]
GraphCutSeamFinderBase_COST_COLOR: Final[int]
GraphCutSeamFinderBase_COST_COLOR_GRAD: Final[int]
OPAQUE_KIND_CV_BOOL: Final[int]
OPAQUE_KIND_CV_DOUBLE: Final[int]
OPAQUE_KIND_CV_DRAW_PRIM: Final[int]
OPAQUE_KIND_CV_FLOAT: Final[int]
OPAQUE_KIND_CV_INT: Final[int]
OPAQUE_KIND_CV_INT64: Final[int]
OPAQUE_KIND_CV_MAT: Final[int]
OPAQUE_KIND_CV_POINT: Final[int]
OPAQUE_KIND_CV_POINT2F: Final[int]
OPAQUE_KIND_CV_POINT3F: Final[int]
OPAQUE_KIND_CV_RECT: Final[int]
OPAQUE_KIND_CV_SCALAR: Final[int]
OPAQUE_KIND_CV_SIZE: Final[int]
OPAQUE_KIND_CV_STRING: Final[int]
OPAQUE_KIND_CV_UINT64: Final[int]
OPAQUE_KIND_CV_UNKNOWN: Final[int]
OpaqueKind_CV_BOOL: Final[int]
OpaqueKind_CV_DOUBLE: Final[int]
OpaqueKind_CV_DRAW_PRIM: Final[int]
OpaqueKind_CV_FLOAT: Final[int]
OpaqueKind_CV_INT: Final[int]
OpaqueKind_CV_INT64: Final[int]
OpaqueKind_CV_MAT: Final[int]
OpaqueKind_CV_POINT: Final[int]
OpaqueKind_CV_POINT2F: Final[int]
OpaqueKind_CV_POINT3F: Final[int]
OpaqueKind_CV_RECT: Final[int]
OpaqueKind_CV_SCALAR: Final[int]
OpaqueKind_CV_SIZE: Final[int]
OpaqueKind_CV_STRING: Final[int]
OpaqueKind_CV_UINT64: Final[int]
OpaqueKind_CV_UNKNOWN: Final[int]
SEAM_FINDER_DP_SEAM: Final[int]
SEAM_FINDER_NO: Final[int]
SEAM_FINDER_VORONOI_SEAM: Final[int]
SeamFinder_DP_SEAM: Final[int]
SeamFinder_NO: Final[int]
SeamFinder_VORONOI_SEAM: Final[int]
TEST_CUSTOM: Final[int]
TEST_EQ: Final[int]
TEST_GE: Final[int]
TEST_GT: Final[int]
TEST_LE: Final[int]
TEST_LT: Final[int]
TEST_NE: Final[int]
TIMELAPSER_AS_IS: Final[int]
TIMELAPSER_CROP: Final[int]
TRACKER_CONTRIB_SAMPLER_CSC_MODE_DETECT: Final[int]
TRACKER_CONTRIB_SAMPLER_CSC_MODE_INIT_NEG: Final[int]
TRACKER_CONTRIB_SAMPLER_CSC_MODE_INIT_POS: Final[int]
TRACKER_CONTRIB_SAMPLER_CSC_MODE_TRACK_NEG: Final[int]
TRACKER_CONTRIB_SAMPLER_CSC_MODE_TRACK_POS: Final[int]
TRACKER_SAMPLER_CSC_MODE_DETECT: Final[int]
TRACKER_SAMPLER_CSC_MODE_INIT_NEG: Final[int]
TRACKER_SAMPLER_CSC_MODE_INIT_POS: Final[int]
TRACKER_SAMPLER_CSC_MODE_TRACK_NEG: Final[int]
TRACKER_SAMPLER_CSC_MODE_TRACK_POS: Final[int]
TRACKER_SAMPLER_CS_MODE_CLASSIFY: Final[int]
TRACKER_SAMPLER_CS_MODE_NEGATIVE: Final[int]
TRACKER_SAMPLER_CS_MODE_POSITIVE: Final[int]
Timelapser_AS_IS: Final[int]
Timelapser_CROP: Final[int]
TrackerContribSamplerCSC_MODE_DETECT: Final[int]
TrackerContribSamplerCSC_MODE_INIT_NEG: Final[int]
TrackerContribSamplerCSC_MODE_INIT_POS: Final[int]
TrackerContribSamplerCSC_MODE_TRACK_NEG: Final[int]
TrackerContribSamplerCSC_MODE_TRACK_POS: Final[int]
TrackerSamplerCSC_MODE_DETECT: Final[int]
TrackerSamplerCSC_MODE_INIT_NEG: Final[int]
TrackerSamplerCSC_MODE_INIT_POS: Final[int]
TrackerSamplerCSC_MODE_TRACK_NEG: Final[int]
TrackerSamplerCSC_MODE_TRACK_POS: Final[int]
TrackerSamplerCS_MODE_CLASSIFY: Final[int]
TrackerSamplerCS_MODE_NEGATIVE: Final[int]
TrackerSamplerCS_MODE_POSITIVE: Final[int]
WAVE_CORRECT_AUTO: Final[int]
WAVE_CORRECT_HORIZ: Final[int]
WAVE_CORRECT_VERT: Final[int]
