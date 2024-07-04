from typing import Any, TypeAlias

from .. import functions as cv2

tvec: TypeAlias = Any
img: TypeAlias = Any
rmsd: TypeAlias = Any
pts3d: TypeAlias = Any
bundle: TypeAlias = Any
cols: TypeAlias = Any
pts2d: TypeAlias = Any
ctl2d: TypeAlias = Any
srcLocations: TypeAlias = Any
response: TypeAlias = Any
ctl3d: TypeAlias = Any
rvec: TypeAlias = Any

retval: TypeAlias = Any

class GOSTracker(Tracker):
    def create(self, pts3d, tris, histBins=..., sobelThesh=...) -> retval:
        """"""

class OLSTracker(Tracker):
    def create(self, pts3d, tris, histBins=..., sobelThesh=...) -> retval:
        """"""

class Rapid(Tracker):
    def create(self, pts3d, tris) -> retval:
        """"""

class Tracker(cv2.Algorithm):
    def clearState(self) -> None:
        """"""

    def compute(self, img, num, len, K, rvec, tvec, termcrit=...) -> tuple[retval, rvec, tvec]:
        """"""

def GOSTracker_create(pts3d, tris, histBins=..., sobelThesh=...) -> retval:
    """
    .
    """

def OLSTracker_create(pts3d, tris, histBins=..., sobelThesh=...) -> retval:
    """
    .
    """

def Rapid_create(pts3d, tris) -> retval:
    """
    .
    """

def convertCorrespondencies(cols, srcLocations, pts2d=..., pts3d=..., mask=...) -> tuple[pts2d, pts3d]:
    """
    * Collect corresponding 2d and 3d points based on correspondencies and mask
    @param cols correspondence-position per line in line-bundle-space
    @param srcLocations the source image location
    @param pts2d 2d points
    @param pts3d 3d points
    @param mask mask containing non-zero values for the elements to be retained
    """

def drawCorrespondencies(bundle, cols, colors=...) -> bundle:
    """
    * Debug draw markers of matched correspondences onto a lineBundle
    @param bundle the lineBundle
    @param cols column coordinates in the line bundle
    @param colors colors for the markers. Defaults to white.
    """

def drawSearchLines(img, locations, color) -> img:
    """
    * Debug draw search lines onto an image
    @param img the output image
    @param locations the source locations of a line bundle
    @param color the line color
    """

def drawWireframe(img, pts2d, tris, color, type=..., cullBackface=...) -> img:
    """
    * Draw a wireframe of a triangle mesh
    @param img the output image
    @param pts2d the 2d points obtained by @ref projectPoints
    @param tris triangle face connectivity
    @param color line color
    @param type line type. See @ref LineTypes.
    @param cullBackface enable back-face culling based on CCW order
    """

def extractControlPoints(num, len, pts3d, rvec, tvec, K, imsize, tris, ctl2d=..., ctl3d=...) -> tuple[ctl2d, ctl3d]:
    """
    * Extract control points from the projected silhouette of a mesh

    see @cite drummond2002real Sec 2.1, Step b
    @param num number of control points
    @param len search radius (used to restrict the ROI)
    @param pts3d the 3D points of the mesh
    @param rvec rotation between mesh and camera
    @param tvec translation between mesh and camera
    @param K camera intrinsic
    @param imsize size of the video frame
    @param tris triangle face connectivity
    @param ctl2d the 2D locations of the control points
    @param ctl3d matching 3D points of the mesh
    """

def extractLineBundle(len, ctl2d, img, bundle=..., srcLocations=...) -> tuple[bundle, srcLocations]:
    """
    * Extract the line bundle from an image
    @param len the search radius. The bundle will have `2*len + 1` columns.
    @param ctl2d the search lines will be centered at this points and orthogonal to the contour defined by
    them. The bundle will have as many rows.
    @param img the image to read the pixel intensities values from
    @param bundle line bundle image with size `ctl2d.rows() x (2 * len + 1)` and the same type as @p img
    @param srcLocations the source pixel locations of @p bundle in @p img as CV_16SC2
    """

def findCorrespondencies(bundle, cols=..., response=...) -> tuple[cols, response]:
    """
    * Find corresponding image locations by searching for a maximal sobel edge along the search line (a single
    row in the bundle)
    @param bundle the line bundle
    @param cols correspondence-position per line in line-bundle-space
    @param response the sobel response for the selected point
    """

def rapid(img, num, len, pts3d, tris, K, rvec, tvec) -> tuple[retval, rvec, tvec, rmsd]:
    """
    * High level function to execute a single rapid @cite harris1990rapid iteration

    1. @ref extractControlPoints
    2. @ref extractLineBundle
    3. @ref findCorrespondencies
    4. @ref convertCorrespondencies
    5. @ref solvePnPRefineLM

    @param img the video frame
    @param num number of search lines
    @param len search line radius
    @param pts3d the 3D points of the mesh
    @param tris triangle face connectivity
    @param K camera matrix
    @param rvec rotation between mesh and camera. Input values are used as an initial solution.
    @param tvec translation between mesh and camera. Input values are used as an initial solution.
    @param rmsd the 2d reprojection difference
    @return ratio of search lines that could be extracted and matched
    """
