import builtins
from typing import Any, overload, TypeAlias

normals: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any
colors: TypeAlias = Any

retval: TypeAlias = Any

class ColoredKinFu(builtins.object):
    def getCloud(self, points=..., normals=..., colors=...) -> tuple[points, normals, colors]:
        """
        @brief Gets points, normals and colors of current 3d mesh

        The order of normals corresponds to order of points.
        The order of points is undefined.

        @param points vector of points which are 4-float vectors
        @param normals vector of normals which are 4-float vectors
        @param colors vector of colors which are 4-float vectors
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

    def update(self, depth, rgb) -> retval:
        """
        @brief Process next depth frame
        @param depth input Mat of depth frame
        @param rgb   input Mat of rgb (colored) frame  @return true if succeeded to align new frame with current scene, false if opposite
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
        A set of parameters suitable for use with HashTSDFVolume
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

def ColoredKinFu_create(_params) -> retval:
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
          A set of parameters suitable for use with HashTSDFVolume
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
