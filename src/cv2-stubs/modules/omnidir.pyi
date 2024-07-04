from typing import Any, Final, overload, TypeAlias

rvecs: TypeAlias = Any
K2: TypeAlias = Any
xi2: TypeAlias = Any
xi1: TypeAlias = Any
disparity: TypeAlias = Any
R1: TypeAlias = Any
map1: TypeAlias = Any
tvecsL: TypeAlias = Any
R2: TypeAlias = Any
tvec: TypeAlias = Any
rvecsL: TypeAlias = Any
D: TypeAlias = Any
image1Rec: TypeAlias = Any
idx: TypeAlias = Any
K1: TypeAlias = Any
rvec: TypeAlias = Any
imagePoints1: TypeAlias = Any
map2: TypeAlias = Any
imagePoints2: TypeAlias = Any
D1: TypeAlias = Any
xi: TypeAlias = Any
jacobian: TypeAlias = Any
D2: TypeAlias = Any
K: TypeAlias = Any
imagePoints: TypeAlias = Any
image2Rec: TypeAlias = Any
pointCloud: TypeAlias = Any
tvecs: TypeAlias = Any
objectPoints: TypeAlias = Any
undistorted: TypeAlias = Any
retval: TypeAlias = Any

@overload
def calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria, rvecs=..., tvecs=..., idx=...) -> tuple[retval, K, xi, D, rvecs, tvecs, idx]:
    """
    @brief Perform omnidirectional camera calibration, the default depth of outputs is CV_64F.

        @param objectPoints Vector of vector of Vec3f object points in world (pattern) coordinate.
    """

@overload
def calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria, rvecs=..., tvecs=..., idx=...) -> tuple[retval, K, xi, D, rvecs, tvecs, idx]:
    """
    @param imagePoints Vector of vector of Vec2f corresponding image points of objectPoints. It must be the same
    """

@overload
def calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria, rvecs=..., tvecs=..., idx=...) -> tuple[retval, K, xi, D, rvecs, tvecs, idx]:
    """
    @param size Image size of calibration images.
    @param K Output calibrated camera matrix.
    @param xi Output parameter xi for CMei's model
    @param D Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$
    @param rvecs Output rotations for each calibration images
    @param tvecs Output translation for each calibration images
    @param flags The flags that control calibrate
    @param criteria Termination criteria for optimization
    @param idx Indices of images that pass initialization, which are really used in calibration. So the size of rvecs is the
    """

@overload
def calibrate(objectPoints, imagePoints, size, K, xi, D, flags, criteria, rvecs=..., tvecs=..., idx=...) -> tuple[retval, K, xi, D, rvecs, tvecs, idx]:
    """ """

@overload
def initUndistortRectifyMap(K, D, xi, R, P, size, m1type, flags, map1=..., map2=...) -> tuple[map1, map2]:
    """
    @brief Computes undistortion and rectification maps for omnidirectional camera image transform by a rotation R.
    """

@overload
def initUndistortRectifyMap(K, D, xi, R, P, size, m1type, flags, map1=..., map2=...) -> tuple[map1, map2]:
    """ """

@overload
def initUndistortRectifyMap(K, D, xi, R, P, size, m1type, flags, map1=..., map2=...) -> tuple[map1, map2]:
    """

    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$, with depth CV_32F or CV_64F
    @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$, with depth CV_32F or CV_64F
    @param xi The parameter xi for CMei's model
    @param R Rotation transform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3, with depth CV_32F or CV_64F
    @param P New camera matrix (3x3) or new projection matrix (3x4)
    @param size Undistorted image size.
    @param m1type Type of the first output map that can be CV_32FC1 or CV_16SC2 . See convertMaps()
    """

@overload
def initUndistortRectifyMap(K, D, xi, R, P, size, m1type, flags, map1=..., map2=...) -> tuple[map1, map2]:
    """
    @param map1 The first output map.
    @param map2 The second output map.
    @param flags Flags indicates the rectification type,  RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and RECTIFY_STEREOGRAPHIC
    """

@overload
def initUndistortRectifyMap(K, D, xi, R, P, size, m1type, flags, map1=..., map2=...) -> tuple[map1, map2]:
    """ """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """
    @brief Projects points for omnidirectional camera using CMei's model

        @param objectPoints Object points in world coordinate, vector of vector of Vec3f or Mat of
        1xN/Nx1 3-channel of type CV_32F and N is the number of points. 64F is also acceptable.
        @param imagePoints Output array of image points, vector of vector of Vec2f or
        1xN/Nx1 2-channel of type CV_32F. 64F is also acceptable.
        @param rvec vector of rotation between world coordinate and camera coordinate, i.e., om
        @param tvec vector of translation between pattern coordinate and camera coordinate
        @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
        @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
        @param xi The parameter xi for CMei's model
        @param jacobian Optional output 2Nx16 of type CV_64F jacobian matrix, contains the derivatives of
    """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """ """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """ """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """ """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """ """

@overload
def projectPoints(objectPoints, rvec, tvec, K, xi, D, imagePoints=..., jacobian=...) -> tuple[imagePoints, jacobian]:
    """ """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """
    @brief Stereo calibration for omnidirectional camera model. It computes the intrinsic parameters for two
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """

    @param objectPoints Object points in world (pattern) coordinate. Its type is vector<vector<Vec3f> >.
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """
    @param imagePoints1 The corresponding image points of the first camera, with type vector<vector<Vec2f> >.
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """
    @param imagePoints2 The corresponding image points of the second camera, with type vector<vector<Vec2f> >.
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """
    @param imageSize1 Image size of calibration images of the first camera.
    @param imageSize2 Image size of calibration images of the second camera.
    @param K1 Output camera matrix for the first camera.
    @param xi1 Output parameter xi of Mei's model for the first camera
    @param D1 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
    @param K2 Output camera matrix for the first camera.
    @param xi2 Output parameter xi of CMei's model for the second camera
    @param D2 Output distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
    @param rvec Output rotation between the first and second camera
    @param tvec Output translation between the first and second camera
    @param rvecsL Output rotation for each image of the first camera
    @param tvecsL Output translation for each image of the first camera
    @param flags The flags that control stereoCalibrate
    @param criteria Termination criteria for optimization
    @param idx Indices of image pairs that pass initialization, which are really used in calibration. So the size of rvecs is the
    """

@overload
def stereoCalibrate(objectPoints, imagePoints1, imagePoints2, imageSize1, imageSize2, K1, xi1, D1, K2, xi2, D2, flags, criteria, rvec=..., tvec=..., rvecsL=..., tvecsL=..., idx=...) -> tuple[retval, objectPoints, imagePoints1, imagePoints2, K1, xi1, D1, K2, xi2, D2, rvec, tvec, rvecsL, tvecsL, idx]:
    """
    @
    """

def stereoReconstruct(image1, image2, K1, D1, xi1, K2, D2, xi2, R, T, flag, numDisparities, SADWindowSize, disparity=..., image1Rec=..., image2Rec=..., newSize=..., Knew=..., pointCloud=..., pointType=...) -> tuple[disparity, image1Rec, image2Rec, pointCloud]:
    """
    @brief Stereo 3D reconstruction from a pair of images

        @param image1 The first input image
        @param image2 The second input image
        @param K1 Input camera matrix of the first camera
        @param D1 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the first camera
        @param xi1 Input parameter xi for the first camera for CMei's model
        @param K2 Input camera matrix of the second camera
        @param D2 Input distortion parameters \f$(k_1, k_2, p_1, p_2)\f$ for the second camera
        @param xi2 Input parameter xi for the second camera for CMei's model
        @param R Rotation between the first and second camera
        @param T Translation between the first and second camera
        @param flag Flag of rectification type, RECTIFY_PERSPECTIVE or RECTIFY_LONGLATI
        @param numDisparities The parameter 'numDisparities' in StereoSGBM, see StereoSGBM for details.
        @param SADWindowSize The parameter 'SADWindowSize' in StereoSGBM, see StereoSGBM for details.
        @param disparity Disparity map generated by stereo matching
        @param image1Rec Rectified image of the first image
        @param image2Rec rectified image of the second image
        @param newSize Image size of rectified image, see omnidir::undistortImage
        @param Knew New camera matrix of rectified image, see omnidir::undistortImage
        @param pointCloud Point cloud of 3D reconstruction, with type CV_64FC3
        @param pointType Point cloud type, it can be XYZRGB or XYZ
    """

def stereoRectify(R, T, R1=..., R2=...) -> tuple[R1, R2]:
    """
    @brief Stereo rectification for omnidirectional camera model. It computes the rectification rotations for two cameras

        @param R Rotation between the first and second camera
        @param T Translation between the first and second camera
        @param R1 Output 3x3 rotation matrix for the first camera
        @param R2 Output 3x3 rotation matrix for the second camera
    """

def undistortImage(distorted, K, D, xi, flags, undistorted=..., Knew=..., new_size=..., R=...) -> undistorted:
    """
    @brief Undistort omnidirectional images to perspective images

        @param distorted The input omnidirectional image.
        @param undistorted The output undistorted image.
        @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
        @param D Input vector of distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
        @param xi The parameter xi for CMei's model.
        @param flags Flags indicates the rectification type,  RECTIFY_PERSPECTIVE, RECTIFY_CYLINDRICAL, RECTIFY_LONGLATI and RECTIFY_STEREOGRAPHIC
        @param Knew Camera matrix of the distorted image. If it is not assigned, it is just K.
        @param new_size The new image size. By default, it is the size of distorted.
        @param R Rotation matrix between the input and output images. By default, it is identity matrix.
    """

@overload
def undistortPoints(distorted, K, D, xi, R, undistorted=...) -> undistorted:
    """
    @brief Undistort 2D image points for omnidirectional camera using CMei's model

        @param distorted Array of distorted image points, vector of Vec2f
    """

@overload
def undistortPoints(distorted, K, D, xi, R, undistorted=...) -> undistorted:
    """
    @param K Camera matrix \f$K = \vecthreethree{f_x}{s}{c_x}{0}{f_y}{c_y}{0}{0}{_1}\f$.
    @param D Distortion coefficients \f$(k_1, k_2, p_1, p_2)\f$.
    @param xi The parameter xi for CMei's model
    @param R Rotation trainsform between the original and object space : 3x3 1-channel, or vector: 3x1/1x3
    1-channel or 1x1 3-channel
    @param undistorted array of normalized object points, vector of Vec2f/Vec2d or 1xN/Nx1 2-channel Mat with the same
    """

@overload
def undistortPoints(distorted, K, D, xi, R, undistorted=...) -> undistorted:
    """ """

CALIB_FIX_CENTER: Final[int]
CALIB_FIX_GAMMA: Final[int]
CALIB_FIX_K1: Final[int]
CALIB_FIX_K2: Final[int]
CALIB_FIX_P1: Final[int]
CALIB_FIX_P2: Final[int]
CALIB_FIX_SKEW: Final[int]
CALIB_FIX_XI: Final[int]
CALIB_USE_GUESS: Final[int]
RECTIFY_CYLINDRICAL: Final[int]
RECTIFY_LONGLATI: Final[int]
RECTIFY_PERSPECTIVE: Final[int]
RECTIFY_STEREOGRAPHIC: Final[int]
XYZ: Final[int]
XYZRGB: Final[int]
