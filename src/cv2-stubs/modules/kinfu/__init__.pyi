import builtins
from typing import Any, Final, overload, TypeAlias

from . import detail

normals: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any
retval: TypeAlias = Any

class KinFu(builtins.object):
    def getCloud(self, points=..., normals=...) -> tuple[points, normals]:
        """
        @brief Gets points and normals of current 3d mesh

        The order of normals corresponds to order of points.
        The order of points is undefined.

        @param points vector of points which are 4-float vectors
        @param normals vector of normals which are 4-float vectors
        """

    def getNormals(self, points, normals=...) -> normals:
        """
        @brief Calculates normals for given points
        @param points input vector of points which are 4-float vectors
        @param normals output vector of corresponding normals which are 4-float vectors
        """

    def getPoints(self, points=...) -> points:
        """
        @brief Gets points of current 3d mesh

        The order of points is undefined.

        @param points vector of points which are 4-float vectors
        """

    @overload
    def render(self, image=...) -> image:
        """
        @brief Renders a volume into an image

        Renders a 0-surface of TSDF using Phong shading into a CV_8UC4 Mat.
        Light pose is fixed in KinFu params.

        @param image resulting image
        """

    @overload
    def render(self, cameraPose, image=...) -> image:
        """
        @brief Renders a volume into an image

        Renders a 0-surface of TSDF using Phong shading into a CV_8UC4 Mat.
        Light pose is fixed in KinFu params.

        @param image resulting image
        @param cameraPose pose of camera to render from. If empty then render from current pose which is a last frame camera pose.
        """

    def reset(self) -> None:
        """
        @brief Resets the algorithm

        Clears current model and resets a pose.
        """

    def update(self, depth) -> retval:
        """
        @brief Process next depth frame

        Integrates depth into voxel space with respect to its ICP-calculated pose.
        Input image is converted to CV_32F internally if has another type.

        @param depth one-channel image which size and depth scale is described in algorithm's parameters @return true if succeeded to align new frame with current scene, false if opposite
        """

    def create(self, _params) -> retval:
        """"""

class Params(builtins.object):
    @overload
    def setInitialVolumePose(self, R, t) -> None:
        """
        * @brief Set Initial Volume Pose
        * Sets the initial pose of the TSDF volume.
        * @param R rotation matrix
        * @param t translation vector
        """

    @overload
    def setInitialVolumePose(self, homogen_tf) -> None:
        """
        * @brief Set Initial Volume Pose
        * Sets the initial pose of the TSDF volume.
        * @param homogen_tf 4 by 4 Homogeneous Transform matrix to set the intial pose of TSDF volume
        """

    def coarseParams(self) -> retval:
        """
        @brief Coarse parameters
        A set of parameters which provides better speed, can fail to match frames
        in case of rapid sensor motion.
        """

    def coloredTSDFParams(self, isCoarse) -> retval:
        """
        @brief ColoredTSDF parameters
        A set of parameters suitable for use with ColoredTSDFVolume
        """

    def defaultParams(self) -> retval:
        """
        * @brief Default parameters
        * A set of parameters which provides better model quality, can be very slow.
        """

    def hashTSDFParams(self, isCoarse) -> retval:
        """
        @brief HashTSDF parameters
        A set of parameters suitable for use with HashTSDFVolume
        """

class Volume(builtins.object): ...

class VolumeParams(builtins.object):
    def coarseParams(self, _volumeType) -> retval:
        """
        @brief Coarse set of parameters that provides relatively higher performance
        at the cost of reconstrution quality.
        """

    def defaultParams(self, _volumeType) -> retval:
        """
        @brief Default set of parameters that provide higher quality reconstruction
        at the cost of slow performance.
        """

def KinFu_create(_params) -> retval:
    """
    .
    """

@overload
def Params_coarseParams() -> retval:
    """
    @brief Coarse parameters
    """

@overload
def Params_coarseParams() -> retval:
    """ """

@overload
def Params_coarseParams() -> retval:
    """ """

def Params_coloredTSDFParams(isCoarse) -> retval:
    """
    @brief ColoredTSDF parameters
          A set of parameters suitable for use with ColoredTSDFVolume
    """

def Params_defaultParams() -> retval:
    """
    * @brief Default parameters
         * A set of parameters which provides better model quality, can be very slow.
    """

def Params_hashTSDFParams(isCoarse) -> retval:
    """
    @brief HashTSDF parameters
          A set of parameters suitable for use with HashTSDFVolume
    """

def VolumeParams_coarseParams(_volumeType) -> retval:
    """
    @brief Coarse set of parameters that provides relatively higher performance
            at the cost of reconstrution quality.
    """

def VolumeParams_defaultParams(_volumeType) -> retval:
    """
    @brief Default set of parameters that provide higher quality reconstruction
            at the cost of slow performance.
    """

def makeVolume(_volumeType, _voxelSize, _pose, _raycastStepFactor, _truncDist, _maxWeight, _truncateThreshold, _resolution) -> retval:
    """
    .
    """

VOLUME_TYPE_COLOREDTSDF: Final[int]
VOLUME_TYPE_HASHTSDF: Final[int]
VOLUME_TYPE_TSDF: Final[int]
VolumeType_COLOREDTSDF: Final[int]
VolumeType_HASHTSDF: Final[int]
VolumeType_TSDF: Final[int]
