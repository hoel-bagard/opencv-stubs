import builtins
from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

depth: TypeAlias = Any
mask: TypeAlias = Any
normals: TypeAlias = Any
out: TypeAlias = Any
plane_coefficients: TypeAlias = Any
Rt: TypeAlias = Any
warpedMask: TypeAlias = Any
points3d: TypeAlias = Any
warpedImage: TypeAlias = Any
registeredDepth: TypeAlias = Any
warpedDepth: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any

class DepthCleaner(cv2.Algorithm):
    def apply(self, points, depth=...) -> depth:
        """
        Given a set of 3d points in a depth image, compute the normals at each point.
        * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
        * @param depth a rows x cols matrix of the cleaned up depth
        """

    def getDepth(self) -> retval:
        """"""

    def getMethod(self) -> retval:
        """"""

    def getWindowSize(self) -> retval:
        """"""

    def initialize(self) -> None:
        """
        Initializes some data that is cached for later computation
        * If that function is not called, it will be called the first time normals are computed
        """

    def setDepth(self, val) -> None:
        """"""

    def setMethod(self, val) -> None:
        """"""

    def setWindowSize(self, val) -> None:
        """"""

    def create(self, depth, window_size=..., method=...) -> retval:
        """
        Constructor
        * @param depth the depth of the normals (only CV_32F or CV_64F)
        * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
        * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
        """

class FastICPOdometry(Odometry):
    def getAngleThreshold(self) -> retval:
        """"""

    def getCameraMatrix(self) -> retval:
        """"""

    def getIterationCounts(self) -> retval:
        """"""

    def getKernelSize(self) -> retval:
        """"""

    def getMaxDistDiff(self) -> retval:
        """"""

    def getSigmaDepth(self) -> retval:
        """"""

    def getSigmaSpatial(self) -> retval:
        """"""

    def getTransformType(self) -> retval:
        """"""

    def prepareFrameCache(self, frame, cacheType) -> retval:
        """"""

    def setAngleThreshold(self, f) -> None:
        """"""

    def setCameraMatrix(self, val) -> None:
        """"""

    def setIterationCounts(self, val) -> None:
        """"""

    def setKernelSize(self, f) -> None:
        """"""

    def setMaxDistDiff(self, val) -> None:
        """"""

    def setSigmaDepth(self, f) -> None:
        """"""

    def setSigmaSpatial(self, f) -> None:
        """"""

    def setTransformType(self, val) -> None:
        """"""

    def create(self, cameraMatrix, maxDistDiff=..., angleThreshold=..., sigmaDepth=..., sigmaSpatial=..., kernelSize=..., iterCounts=...) -> retval:
        """
        Constructor.
        * @param cameraMatrix Camera matrix
        * @param maxDistDiff Correspondences between pixels of two given frames will be filtered out *                     if their depth difference is larger than maxDepthDiff
        * @param angleThreshold Correspondence will be filtered out *                     if an angle between their normals is bigger than threshold
        * @param sigmaDepth Depth sigma in meters for bilateral smooth
        * @param sigmaSpatial Spatial sigma in pixels for bilateral smooth
        * @param kernelSize Kernel size in pixels for bilateral smooth
        * @param iterCounts Count of iterations on each pyramid level
        """

class ICPOdometry(Odometry):
    def getCameraMatrix(self) -> retval:
        """"""

    def getIterationCounts(self) -> retval:
        """"""

    def getMaxDepth(self) -> retval:
        """"""

    def getMaxDepthDiff(self) -> retval:
        """"""

    def getMaxPointsPart(self) -> retval:
        """"""

    def getMaxRotation(self) -> retval:
        """"""

    def getMaxTranslation(self) -> retval:
        """"""

    def getMinDepth(self) -> retval:
        """"""

    def getNormalsComputer(self) -> retval:
        """"""

    def getTransformType(self) -> retval:
        """"""

    def prepareFrameCache(self, frame, cacheType) -> retval:
        """"""

    def setCameraMatrix(self, val) -> None:
        """"""

    def setIterationCounts(self, val) -> None:
        """"""

    def setMaxDepth(self, val) -> None:
        """"""

    def setMaxDepthDiff(self, val) -> None:
        """"""

    def setMaxPointsPart(self, val) -> None:
        """"""

    def setMaxRotation(self, val) -> None:
        """"""

    def setMaxTranslation(self, val) -> None:
        """"""

    def setMinDepth(self, val) -> None:
        """"""

    def setTransformType(self, val) -> None:
        """"""

    def create(self, cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., maxPointsPart=..., iterCounts=..., transformType=...) -> retval:
        """
        Constructor.
        * @param cameraMatrix Camera matrix
        * @param minDepth Pixels with depth less than minDepth will not be used
        * @param maxDepth Pixels with depth larger than maxDepth will not be used
        * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out *                     if their depth difference is larger than maxDepthDiff
        * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
        * @param iterCounts Count of iterations on each pyramid level.
        * @param transformType Class of trasformation
        """

class Odometry(cv2.Algorithm):
    def DEFAULT_MAX_DEPTH(self) -> retval:
        """"""

    def DEFAULT_MAX_DEPTH_DIFF(self) -> retval:
        """"""

    def DEFAULT_MAX_POINTS_PART(self) -> retval:
        """"""

    def DEFAULT_MAX_ROTATION(self) -> retval:
        """"""

    def DEFAULT_MAX_TRANSLATION(self) -> retval:
        """"""

    def DEFAULT_MIN_DEPTH(self) -> retval:
        """ """

    def compute(self, srcImage, srcDepth, srcMask, dstImage, dstDepth, dstMask, Rt=..., initRt=...) -> tuple[retval, Rt]:
        """
        Method to compute a transformation from the source frame to the destination one.
        * Some odometry algorithms do not used some data of frames (eg. ICP does not use images).
        * In such case corresponding arguments can be set as empty Mat.
        * The method returns true if all internal computations were possible (e.g. there were enough correspondences,
        * system of equations has a solution, etc) and resulting transformation satisfies some test if it's provided
        * by the Odometry inheritor implementation (e.g. thresholds for maximum translation and rotation).
        * @param srcImage Image data of the source frame (CV_8UC1)
        * @param srcDepth Depth data of the source frame (CV_32FC1, in meters)
        * @param srcMask Mask that sets which pixels have to be used from the source frame (CV_8UC1)
        * @param dstImage Image data of the destination frame (CV_8UC1)
        * @param dstDepth Depth data of the destination frame (CV_32FC1, in meters)
        * @param dstMask Mask that sets which pixels have to be used from the destination frame (CV_8UC1)
        * @param Rt Resulting transformation from the source frame to the destination one (rigid body motion): dst_p = Rt * src_p, where dst_p is a homogeneous point in the destination frame and src_p is homogeneous point in the source frame, Rt is 4x4 matrix of CV_64FC1 type.
        * @param initRt Initial transformation from the source frame to the destination one (optional)
        """

    def compute2(self, srcFrame, dstFrame, Rt=..., initRt=...) -> tuple[retval, Rt]:
        """
        One more method to compute a transformation from the source frame to the destination one.
        * It is designed to save on computing the frame data (image pyramids, normals, etc.).
        """

    def getCameraMatrix(self) -> retval:
        """
        @see setCameraMatrix
        """

    def getTransformType(self) -> retval:
        """
        @see setTransformType
        """

    def prepareFrameCache(self, frame, cacheType) -> retval:
        """
        Prepare a cache for the frame. The function checks the precomputed/passed data (throws the error if this data
        * does not satisfy) and computes all remaining cache data needed for the frame. Returned size is a resolution
        * of the prepared frame.
        * @param frame The odometry which will process the frame.
        * @param cacheType The cache type: CACHE_SRC, CACHE_DST or CACHE_ALL.
        """

    def setCameraMatrix(self, val) -> None:
        """
        @copybrief getCameraMatrix @see getCameraMatrix
        """

    def setTransformType(self, val) -> None:
        """
        @copybrief getTransformType @see getTransformType
        """

    def create(self, odometryType) -> retval:
        """"""

class OdometryFrame(RgbdFrame):
    def release(self) -> None:
        """"""

    def releasePyramids(self) -> None:
        """"""

    def create(self, image=..., depth=..., mask=..., normals=..., ID=...) -> retval:
        """"""

class RgbdFrame(builtins.object):
    def release(self) -> None:
        """"""

    def create(self, image=..., depth=..., mask=..., normals=..., ID=...) -> retval:
        """"""

class RgbdICPOdometry(Odometry):
    def getCameraMatrix(self) -> retval:
        """"""

    def getIterationCounts(self) -> retval:
        """"""

    def getMaxDepth(self) -> retval:
        """"""

    def getMaxDepthDiff(self) -> retval:
        """"""

    def getMaxPointsPart(self) -> retval:
        """"""

    def getMaxRotation(self) -> retval:
        """"""

    def getMaxTranslation(self) -> retval:
        """"""

    def getMinDepth(self) -> retval:
        """"""

    def getMinGradientMagnitudes(self) -> retval:
        """"""

    def getNormalsComputer(self) -> retval:
        """"""

    def getTransformType(self) -> retval:
        """"""

    def prepareFrameCache(self, frame, cacheType) -> retval:
        """"""

    def setCameraMatrix(self, val) -> None:
        """"""

    def setIterationCounts(self, val) -> None:
        """"""

    def setMaxDepth(self, val) -> None:
        """"""

    def setMaxDepthDiff(self, val) -> None:
        """"""

    def setMaxPointsPart(self, val) -> None:
        """"""

    def setMaxRotation(self, val) -> None:
        """"""

    def setMaxTranslation(self, val) -> None:
        """"""

    def setMinDepth(self, val) -> None:
        """"""

    def setMinGradientMagnitudes(self, val) -> None:
        """"""

    def setTransformType(self, val) -> None:
        """"""

    def create(self, cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., maxPointsPart=..., iterCounts=..., minGradientMagnitudes=..., transformType=...) -> retval:
        """
        Constructor.
        * @param cameraMatrix Camera matrix
        * @param minDepth Pixels with depth less than minDepth will not be used
        * @param maxDepth Pixels with depth larger than maxDepth will not be used
        * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out *                     if their depth difference is larger than maxDepthDiff
        * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
        * @param iterCounts Count of iterations on each pyramid level.
        * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out *                              if they have gradient magnitude less than minGradientMagnitudes[level].
        * @param transformType Class of trasformation
        """

class RgbdNormals(cv2.Algorithm):
    def apply(self, points, normals=...) -> normals:
        """
        Given a set of 3d points in a depth image, compute the normals at each point.
        * @param points a rows x cols x 3 matrix of CV_32F/CV64F or a rows x cols x 1 CV_U16S
        * @param normals a rows x cols x 3 matrix
        """

    def getCols(self) -> retval:
        """"""

    def getDepth(self) -> retval:
        """"""

    def getK(self) -> retval:
        """"""

    def getMethod(self) -> retval:
        """"""

    def getRows(self) -> retval:
        """"""

    def getWindowSize(self) -> retval:
        """"""

    def initialize(self) -> None:
        """
        Initializes some data that is cached for later computation
        * If that function is not called, it will be called the first time normals are computed
        """

    def setCols(self, val) -> None:
        """"""

    def setDepth(self, val) -> None:
        """"""

    def setK(self, val) -> None:
        """"""

    def setMethod(self, val) -> None:
        """"""

    def setRows(self, val) -> None:
        """"""

    def setWindowSize(self, val) -> None:
        """"""

    def create(self, rows, cols, depth, K, window_size=..., method=...) -> retval:
        """
        Constructor
        * @param rows the number of rows of the depth image normals will be computed on
        * @param cols the number of cols of the depth image normals will be computed on
        * @param depth the depth of the normals (only CV_32F or CV_64F)
        * @param K the calibration matrix to use
        * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
        * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
        """

class RgbdOdometry(Odometry):
    def getCameraMatrix(self) -> retval:
        """"""

    def getIterationCounts(self) -> retval:
        """"""

    def getMaxDepth(self) -> retval:
        """"""

    def getMaxDepthDiff(self) -> retval:
        """"""

    def getMaxPointsPart(self) -> retval:
        """"""

    def getMaxRotation(self) -> retval:
        """"""

    def getMaxTranslation(self) -> retval:
        """"""

    def getMinDepth(self) -> retval:
        """"""

    def getMinGradientMagnitudes(self) -> retval:
        """"""

    def getTransformType(self) -> retval:
        """"""

    def prepareFrameCache(self, frame, cacheType) -> retval:
        """"""

    def setCameraMatrix(self, val) -> None:
        """"""

    def setIterationCounts(self, val) -> None:
        """"""

    def setMaxDepth(self, val) -> None:
        """"""

    def setMaxDepthDiff(self, val) -> None:
        """"""

    def setMaxPointsPart(self, val) -> None:
        """"""

    def setMaxRotation(self, val) -> None:
        """"""

    def setMaxTranslation(self, val) -> None:
        """"""

    def setMinDepth(self, val) -> None:
        """"""

    def setMinGradientMagnitudes(self, val) -> None:
        """"""

    def setTransformType(self, val) -> None:
        """"""

    def create(self, cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., iterCounts=..., minGradientMagnitudes=..., maxPointsPart=..., transformType=...) -> retval:
        """
        Constructor.
        * @param cameraMatrix Camera matrix
        * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
        * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
        * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out *                     if their depth difference is larger than maxDepthDiff (in meters)
        * @param iterCounts Count of iterations on each pyramid level.
        * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out *                              if they have gradient magnitude less than minGradientMagnitudes[level].
        * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
        * @param transformType Class of transformation
        """

class RgbdPlane(cv2.Algorithm):
    @overload
    def apply(self, points3d, normals, mask=..., plane_coefficients=...) -> tuple[mask, plane_coefficients]:
        """
        Find The planes in a depth image
        * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
        * @param normals the normals for every point in the depth image
        * @param mask An image where each pixel is labeled with the plane it belongs to *        and 255 if it does not belong to any plane
        * @param plane_coefficients the coefficients of the corresponding planes (a,b,c,d) such that ax+by+cz+d=0, norm(a,b,c)=1 *        and c < 0 (so that the normal points towards the camera)
        """

    @overload
    def apply(self, points3d, mask=..., plane_coefficients=...) -> tuple[mask, plane_coefficients]:
        """
        Find The planes in a depth image but without doing a normal check, which is faster but less accurate
        * @param points3d the 3d points organized like the depth image: rows x cols with 3 channels
        * @param mask An image where each pixel is labeled with the plane it belongs to *        and 255 if it does not belong to any plane
        * @param plane_coefficients the coefficients of the corresponding planes (a,b,c,d) such that ax+by+cz+d=0
        """

    def getBlockSize(self) -> retval:
        """"""

    def getMethod(self) -> retval:
        """"""

    def getMinSize(self) -> retval:
        """"""

    def getSensorErrorA(self) -> retval:
        """"""

    def getSensorErrorB(self) -> retval:
        """"""

    def getSensorErrorC(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setBlockSize(self, val) -> None:
        """"""

    def setMethod(self, val) -> None:
        """"""

    def setMinSize(self, val) -> None:
        """"""

    def setSensorErrorA(self, val) -> None:
        """"""

    def setSensorErrorB(self, val) -> None:
        """"""

    def setSensorErrorC(self, val) -> None:
        """"""

    def setThreshold(self, val) -> None:
        """"""

    def create(self, method, block_size, min_size, threshold, sensor_error_a=..., sensor_error_b=..., sensor_error_c=...) -> retval:
        """
        Constructor
        * @param block_size The size of the blocks to look at for a stable MSE
        * @param min_size The minimum size of a cluster to be considered a plane
        * @param threshold The maximum distance of a point from a plane to belong to it (in meters)
        * @param sensor_error_a coefficient of the sensor error. 0 by default, 0.0075 for a Kinect
        * @param sensor_error_b coefficient of the sensor error. 0 by default
        * @param sensor_error_c coefficient of the sensor error. 0 by default
        * @param method The method to use to compute the planes.
        """

def DepthCleaner_create(depth, window_size=..., method=...) -> retval:
    """
    Constructor
         * @param depth the depth of the normals (only CV_32F or CV_64F)
         * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
         * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
    """

def FastICPOdometry_create(cameraMatrix, maxDistDiff=..., angleThreshold=..., sigmaDepth=..., sigmaSpatial=..., kernelSize=..., iterCounts=...) -> retval:
    """
    Constructor.
         * @param cameraMatrix Camera matrix
         * @param maxDistDiff Correspondences between pixels of two given frames will be filtered out
         *                     if their depth difference is larger than maxDepthDiff
         * @param angleThreshold Correspondence will be filtered out
         *                     if an angle between their normals is bigger than threshold
         * @param sigmaDepth Depth sigma in meters for bilateral smooth
         * @param sigmaSpatial Spatial sigma in pixels for bilateral smooth
         * @param kernelSize Kernel size in pixels for bilateral smooth
         * @param iterCounts Count of iterations on each pyramid level
    """

def ICPOdometry_create(cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., maxPointsPart=..., iterCounts=..., transformType=...) -> retval:
    """
    Constructor.
         * @param cameraMatrix Camera matrix
         * @param minDepth Pixels with depth less than minDepth will not be used
         * @param maxDepth Pixels with depth larger than maxDepth will not be used
         * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
         *                     if their depth difference is larger than maxDepthDiff
         * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
         * @param iterCounts Count of iterations on each pyramid level.
         * @param transformType Class of trasformation
    """

def OdometryFrame_create(image=..., depth=..., mask=..., normals=..., ID=...) -> retval:
    """
    .
    """

def Odometry_create(odometryType) -> retval:
    """
    .
    """

def RgbdFrame_create(image=..., depth=..., mask=..., normals=..., ID=...) -> retval:
    """
    .
    """

def RgbdICPOdometry_create(cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., maxPointsPart=..., iterCounts=..., minGradientMagnitudes=..., transformType=...) -> retval:
    """
    Constructor.
         * @param cameraMatrix Camera matrix
         * @param minDepth Pixels with depth less than minDepth will not be used
         * @param maxDepth Pixels with depth larger than maxDepth will not be used
         * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
         *                     if their depth difference is larger than maxDepthDiff
         * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
         * @param iterCounts Count of iterations on each pyramid level.
         * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
         *                              if they have gradient magnitude less than minGradientMagnitudes[level].
         * @param transformType Class of trasformation
    """

def RgbdNormals_create(rows, cols, depth, K, window_size=..., method=...) -> retval:
    """
    Constructor
         * @param rows the number of rows of the depth image normals will be computed on
         * @param cols the number of cols of the depth image normals will be computed on
         * @param depth the depth of the normals (only CV_32F or CV_64F)
         * @param K the calibration matrix to use
         * @param window_size the window size to compute the normals: can only be 1,3,5 or 7
         * @param method one of the methods to use: RGBD_NORMALS_METHOD_SRI, RGBD_NORMALS_METHOD_FALS
    """

def RgbdOdometry_create(cameraMatrix=..., minDepth=..., maxDepth=..., maxDepthDiff=..., iterCounts=..., minGradientMagnitudes=..., maxPointsPart=..., transformType=...) -> retval:
    """
    Constructor.
         * @param cameraMatrix Camera matrix
         * @param minDepth Pixels with depth less than minDepth will not be used (in meters)
         * @param maxDepth Pixels with depth larger than maxDepth will not be used (in meters)
         * @param maxDepthDiff Correspondences between pixels of two given frames will be filtered out
         *                     if their depth difference is larger than maxDepthDiff (in meters)
         * @param iterCounts Count of iterations on each pyramid level.
         * @param minGradientMagnitudes For each pyramid level the pixels will be filtered out
         *                              if they have gradient magnitude less than minGradientMagnitudes[level].
         * @param maxPointsPart The method uses a random pixels subset of size frameWidth x frameHeight x pointsPart
         * @param transformType Class of transformation
    """

def RgbdPlane_create(method, block_size, min_size, threshold, sensor_error_a=..., sensor_error_b=..., sensor_error_c=...) -> retval:
    """
    Constructor
         * @param block_size The size of the blocks to look at for a stable MSE
         * @param min_size The minimum size of a cluster to be considered a plane
         * @param threshold The maximum distance of a point from a plane to belong to it (in meters)
         * @param sensor_error_a coefficient of the sensor error. 0 by default, 0.0075 for a Kinect
         * @param sensor_error_b coefficient of the sensor error. 0 by default
         * @param sensor_error_c coefficient of the sensor error. 0 by default
         * @param method The method to use to compute the planes.
    """

def depthTo3d(depth, K, points3d=..., mask=...) -> points3d:
    """
    Converts a depth image to an organized set of 3d points.
       * The coordinate system is x pointing left, y down and z away from the camera
       * @param depth the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
       *              (as done with the Microsoft Kinect), otherwise, if given as CV_32F or CV_64F, it is assumed in meters)
       * @param K The calibration matrix
       * @param points3d the resulting 3d points. They are of depth the same as `depth` if it is CV_32F or CV_64F, and the
       *        depth of `K` if `depth` is of depth CV_U
       * @param mask the mask of the points to consider (can be empty)
    """

def depthTo3dSparse(depth, in_K, in_points, points3d=...) -> points3d:
    """
    * @param depth the depth image
       * @param in_K
       * @param in_points the list of xy coordinates
       * @param points3d the resulting 3d points
    """

def registerDepth(unregisteredCameraMatrix, registeredCameraMatrix, registeredDistCoeffs, Rt, unregisteredDepth, outputImagePlaneSize, registeredDepth=..., depthDilation=...) -> registeredDepth:
    """
    Registers depth data to an external camera
       * Registration is performed by creating a depth cloud, transforming the cloud by
       * the rigid body transformation between the cameras, and then projecting the
       * transformed points into the RGB camera.
       *
       * uv_rgb = K_rgb * [R | t] * z * inv(K_ir) * uv_ir
       *
       * Currently does not check for negative depth values.
       *
       * @param unregisteredCameraMatrix the camera matrix of the depth camera
       * @param registeredCameraMatrix the camera matrix of the external camera
       * @param registeredDistCoeffs the distortion coefficients of the external camera
       * @param Rt the rigid body transform between the cameras. Transforms points from depth camera frame to external camera frame.
       * @param unregisteredDepth the input depth data
       * @param outputImagePlaneSize the image plane dimensions of the external camera (width, height)
       * @param registeredDepth the result of transforming the depth into the external camera
       * @param depthDilation whether or not the depth is dilated to avoid holes and occlusion errors (optional)
    """

def rescaleDepth(in_, depth, out=..., depth_factor=...) -> out:
    """
    If the input image is of type CV_16UC1 (like the Kinect one), the image is converted to floats, divided
       * by depth_factor to get a depth in meters, and the values 0 are converted to std::numeric_limits<float>::quiet_NaN()
       * Otherwise, the image is simply converted to floats
       * @param in the depth image (if given as short int CV_U, it is assumed to be the depth in millimeters
       *              (as done with the Microsoft Kinect), it is assumed in meters)
       * @param depth the desired output depth (floats or double)
       * @param out The rescaled float depth image
       * @param depth_factor (optional) factor by which depth is converted to distance (by default = 1000.0 for Kinect sensor)
    """

def warpFrame(image, depth, mask, Rt, cameraMatrix, distCoeff, warpedImage=..., warpedDepth=..., warpedMask=...) -> tuple[warpedImage, warpedDepth, warpedMask]:
    """
    Warp the image: compute 3d points from the depth, transform them using given transformation,
       * then project color point cloud to an image plane.
       * This function can be used to visualize results of the Odometry algorithm.
       * @param image The image (of CV_8UC1 or CV_8UC3 type)
       * @param depth The depth (of type used in depthTo3d fuction)
       * @param mask The mask of used pixels (of CV_8UC1), it can be empty
       * @param Rt The transformation that will be applied to the 3d points computed from the depth
       * @param cameraMatrix Camera matrix
       * @param distCoeff Distortion coefficients
       * @param warpedImage The warped image.
       * @param warpedDepth The warped depth.
       * @param warpedMask The warped mask.
    """

DEPTH_CLEANER_DEPTH_CLEANER_NIL: Final[int]
DepthCleaner_DEPTH_CLEANER_NIL: Final[int]
ODOMETRY_FRAME_CACHE_ALL: Final[int]
ODOMETRY_FRAME_CACHE_DST: Final[int]
ODOMETRY_FRAME_CACHE_SRC: Final[int]
ODOMETRY_RIGID_BODY_MOTION: Final[int]
ODOMETRY_ROTATION: Final[int]
ODOMETRY_TRANSLATION: Final[int]
OdometryFrame_CACHE_ALL: Final[int]
OdometryFrame_CACHE_DST: Final[int]
OdometryFrame_CACHE_SRC: Final[int]
Odometry_RIGID_BODY_MOTION: Final[int]
Odometry_ROTATION: Final[int]
Odometry_TRANSLATION: Final[int]
RGBD_NORMALS_RGBD_NORMALS_METHOD_FALS: Final[int]
RGBD_NORMALS_RGBD_NORMALS_METHOD_LINEMOD: Final[int]
RGBD_NORMALS_RGBD_NORMALS_METHOD_SRI: Final[int]
RGBD_PLANE_RGBD_PLANE_METHOD_DEFAULT: Final[int]
RgbdNormals_RGBD_NORMALS_METHOD_FALS: Final[int]
RgbdNormals_RGBD_NORMALS_METHOD_LINEMOD: Final[int]
RgbdNormals_RGBD_NORMALS_METHOD_SRI: Final[int]
RgbdPlane_RGBD_PLANE_METHOD_DEFAULT: Final[int]
