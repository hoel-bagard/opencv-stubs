import builtins
from typing import Any, TypeAlias

normals: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any

retval: TypeAlias = Any

class DynaFu(builtins.object):
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

    def render(self, image=..., cameraPose=...) -> image:
        """
        @brief Renders a volume into an image

        Renders a 0-surface of TSDF using Phong shading into a CV_8UC4 Mat.
        Light pose is fixed in DynaFu params.

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

def DynaFu_create(_params) -> retval:
    """
    .
    """
