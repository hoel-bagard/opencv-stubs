from typing import Any, Final, overload, TypeAlias

rvecs: TypeAlias = Any
K2: TypeAlias = Any
D1: TypeAlias = Any
R1: TypeAlias = Any
map1: TypeAlias = Any
T: TypeAlias = Any
R2: TypeAlias = Any
distorted: TypeAlias = Any
D: TypeAlias = Any
P: TypeAlias = Any
K1: TypeAlias = Any
map2: TypeAlias = Any
R: TypeAlias = Any
Q: TypeAlias = Any
jacobian: TypeAlias = Any
P2: TypeAlias = Any
D2: TypeAlias = Any
K: TypeAlias = Any
imagePoints: TypeAlias = Any
P1: TypeAlias = Any
tvecs: TypeAlias = Any
undistorted: TypeAlias = Any

retval: TypeAlias = Any

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    @brief Performs camera calibration

        @param objectPoints vector of vectors of calibration pattern points in the calibration pattern
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    @param imagePoints vector of vectors of the projections of calibration pattern points.
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """ """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    @param image_size Size of the image used only to initialize the camera intrinsic matrix.
    @param K Output 3x3 floating-point camera intrinsic matrix
    \f$\cameramatrix{A}\f$ . If
    @ref fisheye::CALIB_USE_INTRINSIC_GUESS is specified, some or all of fx, fy, cx, cy must be
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    @param D Output vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each pattern view.
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """ """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """ """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """ """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param flags Different flags that may be zero or a combination of the following values:
    -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  cameraMatrix contains valid initial values of
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """ """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
    -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    -    @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients
    """

@overload
def calibrate(objectPoints, imagePoints, image_size, K, D, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K, D, rvecs, tvecs]:
    """
        -    @ref fisheye::CALIB_FIX_PRINCIPAL_POINT  The principal point is not changed during the global
    optimization. It stays at the center or at a different location specified when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
        -    @ref fisheye::CALIB_FIX_FOCAL_LENGTH The focal length is not changed during the global
    optimization. It is the \f$max(width,height)/\pi\f$ or the provided \f$f_x\f$, \f$f_y\f$ when @ref fisheye::CALIB_USE_INTRINSIC_GUESS is set too.
        @param criteria Termination criteria for the iterative optimization algorithm.
    """

@overload
def distortPoints(undistorted, K, D, distorted=..., alpha=...) -> distorted:
    """
    @brief Distorts 2D points using fisheye model.

        @param undistorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is
    """

@overload
def distortPoints(undistorted, K, D, distorted=..., alpha=...) -> distorted:
    """
    @param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param alpha The skew coefficient.
    @param distorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
    """

@overload
def distortPoints(undistorted, K, D, distorted=..., alpha=...) -> distorted:
    """ """

@overload
def distortPoints(undistorted, K, D, distorted=..., alpha=...) -> distorted:
    """ """

@overload
def estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P=..., balance=..., new_size=..., fov_scale=...) -> P:
    """
    @brief Estimates new camera intrinsic matrix for undistortion or rectification.

        @param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
        @param image_size Size of the image
        @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
        @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
        1-channel or 1x1 3-channel
        @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
        @param balance Sets the new focal length in range between the min focal length and the max focal
    """

@overload
def estimateNewCameraMatrixForUndistortRectify(K, D, image_size, R, P=..., balance=..., new_size=..., fov_scale=...) -> P:
    """
    @param new_size the new size
    @param fov_scale Divisor for new focal length.
    """

@overload
def initUndistortRectifyMap(K, D, R, P, size, m1type, map1=..., map2=...) -> tuple[map1, map2]:
    """
    @brief Computes undistortion and rectification maps for image transform by #remap. If D is empty zero
    """

@overload
def initUndistortRectifyMap(K, D, R, P, size, m1type, map1=..., map2=...) -> tuple[map1, map2]:
    """

    @param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See #convertMaps
    """

@overload
def initUndistortRectifyMap(K, D, R, P, size, m1type, map1=..., map2=...) -> tuple[map1, map2]:
    """
    @param map1 The first output map.
    @param map2 The second output map.
    """

def projectPoints(objectPoints, rvec, tvec, K, D, imagePoints=..., alpha=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """
    @overload
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @brief Performs stereo calibration

        @param objectPoints Vector of vectors of the calibration pattern points.
        @param imagePoints1 Vector of vectors of the projections of the calibration pattern points,
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param imagePoints2 Vector of vectors of the projections of the calibration pattern points,
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param K1 Input/output first camera intrinsic matrix:
    \f$\vecthreethree{f_x^{(j)}}{0}{c_x^{(j)}}{0}{f_y^{(j)}}{c_y^{(j)}}{0}{0}{1}\f$ , \f$j = 0,\, 1\f$ . If
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param D1 Input/output vector of distortion coefficients \f$\distcoeffsfisheye\f$ of 4 elements.
    @param K2 Input/output second camera intrinsic matrix. The parameter is similar to K1 .
    @param D2 Input/output lens distortion coefficients for the second camera. The parameter is
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param imageSize Size of the image used only to initialize camera intrinsic matrix.
    @param R Output rotation matrix between the 1st and the 2nd camera coordinate systems.
    @param T Output translation vector between the coordinate systems of the cameras.
    @param rvecs Output vector of rotation vectors ( @ref Rodrigues ) estimated for each pattern view in the
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param tvecs Output vector of translation vectors estimated for each pattern view, see parameter description
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param flags Different flags that may be zero or a combination of the following values:
    -    @ref fisheye::CALIB_FIX_INTRINSIC  Fix K1, K2? and D1, D2? so that only R, T matrices
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    -    @ref fisheye::CALIB_USE_INTRINSIC_GUESS  K1, K2 contains valid initial values of
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    -    @ref fisheye::CALIB_RECOMPUTE_EXTRINSIC  Extrinsic will be recomputed after each iteration
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    -    @ref fisheye::CALIB_CHECK_COND  The functions will check validity of condition number.
    -    @ref fisheye::CALIB_FIX_SKEW  Skew coefficient (alpha) is set to zero and stay zero.
    -   @ref fisheye::CALIB_FIX_K1,..., @ref fisheye::CALIB_FIX_K4 Selected distortion coefficients are set to zeros and stay
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    @param criteria Termination criteria for the iterative optimization algorithm.
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R=..., T=..., rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, K1, D1, K2, D2, R, T, rvecs, tvecs]:
    """
    .
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @brief Stereo rectification for fisheye camera model

        @param K1 First camera intrinsic matrix.
        @param D1 First camera distortion parameters.
        @param K2 Second camera intrinsic matrix.
        @param D2 Second camera distortion parameters.
        @param imageSize Size of the image used for stereo calibration.
        @param R Rotation matrix between the coordinate systems of the first and the second
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param tvec Translation vector between coordinate systems of the cameras.
    @param R1 Output 3x3 rectification transform (rotation matrix) for the first camera.
    @param R2 Output 3x3 rectification transform (rotation matrix) for the second camera.
    @param P1 Output 3x4 projection matrix in the new (rectified) coordinate systems for the first
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param P2 Output 3x4 projection matrix in the new (rectified) coordinate systems for the second
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param Q Output \f$4 \times 4\f$ disparity-to-depth mapping matrix (see #reprojectImageTo3D ).
    @param flags Operation flags that may be zero or @ref fisheye::CALIB_ZERO_DISPARITY . If the flag is set,
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """ """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """ """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """ """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param newImageSize New image resolution after rectification. The same size should be passed to
    #initUndistortRectifyMap (see the stereo_calib.cpp sample in OpenCV samples directory). When (0,0)
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """ """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param balance Sets the new focal length in range between the min focal length and the max focal
    """

@overload
def stereoRectify(K1, D1, K2, D2, imageSize, R, tvec, flags, R1=..., R2=..., P1=..., P2=..., Q=..., newImageSize=..., balance=..., fov_scale=...) -> tuple[R1, R2, P1, P2, Q]:
    """
    @param fov_scale Divisor for new focal length.
    """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """
    @brief Transforms an image to compensate for fisheye lens distortion.

        @param distorted image with fisheye lens distortion.
        @param undistorted Output image with compensated fisheye lens distortion.
        @param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
        @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
        @param Knew Camera intrinsic matrix of the distorted image. By default, it is the identity matrix but you
    """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """
    @param new_size the new size
    """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """ """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """
    (with bilinear interpolation). See the former function for details of the transformation being
    """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """ """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """
    -   a\) result of undistort of perspective camera model (all possible coefficients (k_1, k_2, k_3,
         k_4, k_5, k_6) of distortion were optimized under calibration)
     -   b\) result of #fisheye::undistortImage of fisheye camera model (all possible coefficients (k_1, k_2,
         k_3, k_4) of fisheye distortion were optimized under calibration)
     -   c\) original image was captured with fisheye lens
    """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """ """

@overload
def undistortImage(distorted, K, D, undistorted=..., Knew=..., new_size=...) -> undistorted:
    """

    ![image](pics/fisheye_undistorted.jpg)
    """

@overload
def undistortPoints(distorted, K, D, undistorted=..., R=..., P=..., criteria=...) -> undistorted:
    """
    @brief Undistorts 2D points using fisheye model

        @param distorted Array of object points, 1xN/Nx1 2-channel (or vector\<Point2f\> ), where N is the
    """

@overload
def undistortPoints(distorted, K, D, undistorted=..., R=..., P=..., criteria=...) -> undistorted:
    """
    @param K Camera intrinsic matrix \f$cameramatrix{K}\f$.
    @param D Input vector of distortion coefficients \f$\distcoeffsfisheye\f$.
    @param R Rectification transformation in the object space: 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param P New camera intrinsic matrix (3x3) or new projection matrix (3x4)
    @param criteria Termination criteria
    @param undistorted Output array of image points, 1xN/Nx1 2-channel, or vector\<Point2f\> .
    """

CALIB_CHECK_COND: Final[int]
CALIB_FIX_FOCAL_LENGTH: Final[int]
CALIB_FIX_INTRINSIC: Final[int]
CALIB_FIX_K1: Final[int]
CALIB_FIX_K2: Final[int]
CALIB_FIX_K3: Final[int]
CALIB_FIX_K4: Final[int]
CALIB_FIX_PRINCIPAL_POINT: Final[int]
CALIB_FIX_SKEW: Final[int]
CALIB_RECOMPUTE_EXTRINSIC: Final[int]
CALIB_USE_INTRINSIC_GUESS: Final[int]
CALIB_ZERO_DISPARITY: Final[int]
