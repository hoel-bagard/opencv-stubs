import builtins
from typing import Any, Final, TypeAlias

from .. import functions as cv2

rvecs: TypeAlias = Any
detectedCorners: TypeAlias = Any
image: TypeAlias = Any
charucoCorners: TypeAlias = Any
tvec: TypeAlias = Any
cameraMatrix: TypeAlias = Any
perViewErrors: TypeAlias = Any
diamondIds: TypeAlias = Any
imgPoints: TypeAlias = Any
corners: TypeAlias = Any
idx: TypeAlias = Any
rejectedCorners: TypeAlias = Any
markerCorners: TypeAlias = Any
rvec: TypeAlias = Any
recoveredIdxs: TypeAlias = Any
charucoIds: TypeAlias = Any
rotation: TypeAlias = Any
markerIds: TypeAlias = Any
ids: TypeAlias = Any
img: TypeAlias = Any
_img: TypeAlias = Any
stdDeviationsExtrinsics: TypeAlias = Any
diamondCorners: TypeAlias = Any
stdDeviationsIntrinsics: TypeAlias = Any
rejectedImgPoints: TypeAlias = Any
objPoints: TypeAlias = Any
detectedIds: TypeAlias = Any
tvecs: TypeAlias = Any
distCoeffs: TypeAlias = Any

retval: TypeAlias = Any

class ArucoDetector(cv2.Algorithm):
    def detectMarkers(self, image, corners=..., ids=..., rejectedImgPoints=...) -> tuple[corners, ids, rejectedImgPoints]:
        """
        @brief Basic marker detection
        *
        * @param image input image
        * @param corners vector of detected marker corners. For each marker, its four corners * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, * the dimensions of this array is Nx4. The order of the corners is clockwise.
        * @param ids vector of identifiers of the detected markers. The identifier is of type int * (e.g. std::vector<int>). For N detected markers, the size of ids is also N. * The identifiers have the same order than the markers in the imgPoints array.
        * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a * correct codification. Useful for debugging purposes. * * Performs marker detection in the input image. Only markers included in the specific dictionary * are searched. For each detected marker, it returns the 2D position of its corner in the image * and its corresponding identifier. * Note that this function does not perform pose estimation. * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort * input image with corresponging camera model, if camera parameters are known * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
        """

    def getDetectorParameters(self) -> retval:
        """"""

    def getDictionary(self) -> retval:
        """"""

    def getRefineParameters(self) -> retval:
        """"""

    def read(self, fn) -> None:
        """
        @brief Reads algorithm parameters from a file storage
        """

    def refineDetectedMarkers(self, image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix=..., distCoeffs=..., recoveredIdxs=...) -> tuple[detectedCorners, detectedIds, rejectedCorners, recoveredIdxs]:
        """
        @brief Refind not detected markers based on the already detected and the board layout
        *
        * @param image input image
        * @param board layout of markers in the board.
        * @param detectedCorners vector of already detected marker corners.
        * @param detectedIds vector of already detected marker identifiers.
        * @param rejectedCorners vector of rejected candidates during the marker detection process.
        * @param cameraMatrix optional input 3x3 floating-point camera matrix * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
        * @param distCoeffs optional vector of distortion coefficients * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
        * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the * original rejectedCorners array. * * This function tries to find markers that were not detected in the basic detecMarkers function. * First, based on the current detected marker and the board layout, the function interpolates * the position of the missing markers. Then it tries to find correspondence between the reprojected * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate parameters. * If camera parameters and distortion coefficients are provided, missing markers are reprojected * using projectPoint function. If not, missing marker projections are interpolated using global * homography, and all the marker corners in the board must have the same Z coordinate.
        """

    def setDetectorParameters(self, detectorParameters) -> None:
        """"""

    def setDictionary(self, dictionary) -> None:
        """"""

    def setRefineParameters(self, refineParameters) -> None:
        """"""

    def write(self, fs, name) -> None:
        """
        @brief simplified API for language bindings
        """

class Board(builtins.object):
    def generateImage(self, outSize, img=..., marginSize=..., borderBits=...) -> img:
        """
        @brief Draw a planar board
        *
        * @param outSize size of the output image in pixels.
        * @param img output image with the board. The size of this image will be outSize * and the board will be on the center, keeping the board proportions.
        * @param marginSize minimum margins (in pixels) of the board in the output image
        * @param borderBits width of the marker borders. * * This function return the image of the board, ready to be printed.
        """

    def getDictionary(self) -> retval:
        """
        @brief return the Dictionary of markers employed for this board
        """

    def getIds(self) -> retval:
        """
        @brief vector of the identifiers of the markers in the board (should be the same size as objPoints)
        * @return vector of the identifiers of the markers
        """

    def getObjPoints(self) -> retval:
        """
        @brief return array of object points of all the marker corners in the board.
        *
        * Each marker include its 4 corners in this order:
        * -   objPoints[i][0] - left-top point of i-th marker
        * -   objPoints[i][1] - right-top point of i-th marker
        * -   objPoints[i][2] - right-bottom point of i-th marker
        * -   objPoints[i][3] - left-bottom point of i-th marker
        *
        * Markers are placed in a certain order - row by row, left to right in every row. For M markers, the size is Mx4.
        """

    def getRightBottomCorner(self) -> retval:
        """
        @brief get coordinate of the bottom right corner of the board, is set when calling the function create()
        """

    def matchImagePoints(self, detectedCorners, detectedIds, objPoints=..., imgPoints=...) -> tuple[objPoints, imgPoints]:
        """
        @brief Given a board configuration and a set of detected markers, returns the corresponding
        * image points and object points to call solvePnP()
        *
        * @param detectedCorners List of detected marker corners of the board. * For CharucoBoard class you can set list of charuco corners.
        * @param detectedIds List of identifiers for each marker or list of charuco identifiers for each corner. * For CharucoBoard class you can set list of charuco identifiers for each corner.
        * @param objPoints Vector of vectors of board marker points in the board coordinate space.
        * @param imgPoints Vector of vectors of the projections of board marker corner points.
        """

class CharucoBoard(Board):
    def checkCharucoCornersCollinear(self, charucoIds) -> retval:
        """
        @brief check whether the ChArUco markers are collinear
        *
        * @param charucoIds list of identifiers for each corner in charucoCorners per frame. * @return bool value, 1 (true) if detected corners form a line, 0 (false) if they do not. * solvePnP, calibration functions will fail if the corners are collinear (true). * * The number of ids in charucoIDs should be <= the number of chessboard corners in the board. * This functions checks whether the charuco corners are on a straight line (returns true, if so), or not (false). * Axis parallel, as well as diagonal and other straight lines detected.  Degenerate cases: * for number of charucoIDs <= 2,the function returns true.
        """

    def getChessboardCorners(self) -> retval:
        """
        @brief get CharucoBoard::chessboardCorners
        """

    def getChessboardSize(self) -> retval:
        """"""

    def getMarkerLength(self) -> retval:
        """"""

    def getSquareLength(self) -> retval:
        """"""

class CharucoDetector(cv2.Algorithm):
    def detectBoard(self, image, charucoCorners=..., charucoIds=..., markerCorners=..., markerIds=...) -> tuple[charucoCorners, charucoIds, markerCorners, markerIds]:
        """
        * @brief detect aruco markers and interpolate position of ChArUco board corners
        * @param image input image necesary for corner refinement. Note that markers are not detected and * should be sent in corners and ids parameters.
        * @param charucoCorners interpolated chessboard corners.
        * @param charucoIds interpolated chessboard corners identifiers.
        * @param markerCorners vector of already detected markers corners. For each marker, its four * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the * dimensions of this array should be Nx4. The order of the corners should be clockwise. * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        * @param markerIds list of identifiers for each marker in corners. *  If markerCorners and markerCorners are empty, the function detect aruco markers and ids. * * This function receives the detected markers and returns the 2D position of the chessboard corners * from a ChArUco board using the detected Aruco markers. * * If markerCorners and markerCorners are empty, the detectMarkers() will run and detect aruco markers and ids. * * If camera parameters are provided, the process is based in an approximated pose estimation, else it is based on local homography. * Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds. * @sa findChessboardCorners
        """

    def detectDiamonds(self, image, diamondCorners=..., diamondIds=..., markerCorners=..., markerIds=...) -> tuple[diamondCorners, diamondIds, markerCorners, markerIds]:
        """
        * @brief Detect ChArUco Diamond markers
        *
        * @param image input image necessary for corner subpixel.
        * @param diamondCorners output list of detected diamond corners (4 corners per diamond). The order * is the same than in marker corners: top left, top right, bottom right and bottom left. Similar * format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ).
        * @param diamondIds ids of the diamonds in diamondCorners. The id of each diamond is in fact of * type Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the * diamond.
        * @param markerCorners list of detected marker corners from detectMarkers function. * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        * @param markerIds list of marker ids in markerCorners. * If markerCorners and markerCorners are empty, the function detect aruco markers and ids. * * This function detects Diamond markers from the previous detected ArUco markers. The diamonds * are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters * are provided, the diamond search is based on reprojection. If not, diamond search is based on * homography. Homography is faster than reprojection, but less accurate.
        """

    def getBoard(self) -> retval:
        """"""

    def getCharucoParameters(self) -> retval:
        """"""

    def getDetectorParameters(self) -> retval:
        """"""

    def getRefineParameters(self) -> retval:
        """"""

    def setBoard(self, board) -> None:
        """"""

    def setCharucoParameters(self, charucoParameters) -> None:
        """"""

    def setDetectorParameters(self, detectorParameters) -> None:
        """"""

    def setRefineParameters(self, refineParameters) -> None:
        """"""

class CharucoParameters(builtins.object): ...

class DetectorParameters(builtins.object):
    def readDetectorParameters(self, fn) -> retval:
        """
        @brief Read a new set of DetectorParameters from FileNode (use FileStorage.root()).
        """

    def writeDetectorParameters(self, fs, name=...) -> retval:
        """
        @brief Write a set of DetectorParameters to FileStorage
        """

class Dictionary(builtins.object):
    def generateImageMarker(self, id, sidePixels, _img=..., borderBits=...) -> _img:
        """
        @brief Generate a canonical marker image
        """

    def getDistanceToId(self, bits, id, allRotations=...) -> retval:
        """
        @brief Returns the distance of the input bits to the specific id.
        *
        * If allRotations is true, the four posible bits rotation are considered
        """

    def identify(self, onlyBits, maxCorrectionRate) -> tuple[retval, idx, rotation]:
        """
        @brief Given a matrix of bits. Returns whether if marker is identified or not.
        *
        * It returns by reference the correct id (if any) and the correct rotation
        """

    def readDictionary(self, fn) -> retval:
        """
        @brief Read a new dictionary from FileNode.
        *
        * Dictionary format:\n
        * nmarkers: 35\n
        * markersize: 6\n
        * maxCorrectionBits: 5\n
        * marker_0: "101011111011111001001001101100000000"\n
        * ...\n
        * marker_34: "011111010000111011111110110101100101"
        """

    def writeDictionary(self, fs, name=...) -> None:
        """
        @brief Write a dictionary to FileStorage, format is the same as in readDictionary().
        """

    def getBitsFromByteList(self, byteList, markerSize) -> retval:
        """
        @brief Transform list of bytes to matrix of bits
        """

    def getByteListFromBits(self, bits) -> retval:
        """
        @brief Transform matrix of bits to list of bytes in the 4 rotations
        """

class EstimateParameters(builtins.object): ...

class GridBoard(Board):
    def getGridSize(self) -> retval:
        """"""

    def getMarkerLength(self) -> retval:
        """"""

    def getMarkerSeparation(self) -> retval:
        """"""

class RefineParameters(builtins.object):
    def readRefineParameters(self, fn) -> retval:
        """
        @brief Read a new set of RefineParameters from FileNode (use FileStorage.root()).
        """

    def writeRefineParameters(self, fs, name=...) -> retval:
        """
        @brief Write a set of RefineParameters to FileStorage
        """

def Dictionary_getBitsFromByteList(byteList, markerSize) -> retval:
    """
    @brief Transform list of bytes to matrix of bits
    """

def Dictionary_getByteListFromBits(bits) -> retval:
    """
    @brief Transform matrix of bits to list of bytes in the 4 rotations
    """

def calibrateCameraAruco(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, cameraMatrix, distCoeffs, rvecs, tvecs]:
    """
    @overload
    @brief It's the same function as #calibrateCameraAruco but without calibration error estimation.
    """

def calibrateCameraArucoExtended(corners, ids, counter, board, imageSize, cameraMatrix, distCoeffs, rvecs=..., tvecs=..., stdDeviationsIntrinsics=..., stdDeviationsExtrinsics=..., perViewErrors=..., flags=..., criteria=...) -> tuple[retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors]:
    """
    * @brief Calibrate a camera using aruco markers

    @param corners vector of detected marker corners in all frames.
    The corners should have the same format returned by detectMarkers (see #detectMarkers).
    @param ids list of identifiers for each marker in corners
    @param counter number of markers in each frame so that corners and ids can be split
    @param board Marker Board layout
    @param imageSize Size of the image used only to initialize the intrinsic camera matrix.
    @param cameraMatrix Output 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
    and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
    initialized before calling the function.
    @param distCoeffs Output vector of distortion coefficients
    \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
    @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each board view
    (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
    k-th translation vector (see the next output parameter description) brings the board pattern
    from the model coordinate space (in which object points are specified) to the world coordinate
    space, that is, a real position of the board pattern in the k-th pattern view (k=0.. *M* -1).
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
    Order of deviations values:
    \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
    s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
    @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
    Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
    \f$R_i, T_i\f$ are concatenated 1x3 vectors.
    @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
    @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
    @param criteria Termination criteria for the iterative optimization algorithm.

    This function calibrates a camera using an Aruco Board. The function receives a list of
    detected markers from several views of the Board. The process is similar to the chessboard
    calibration in calibrateCamera(). The function returns the final re-projection error.
    """

def calibrateCameraCharuco(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=..., tvecs=..., flags=..., criteria=...) -> tuple[retval, cameraMatrix, distCoeffs, rvecs, tvecs]:
    """
    * @brief It's the same function as #calibrateCameraCharuco but without calibration error estimation.
    """

def calibrateCameraCharucoExtended(charucoCorners, charucoIds, board, imageSize, cameraMatrix, distCoeffs, rvecs=..., tvecs=..., stdDeviationsIntrinsics=..., stdDeviationsExtrinsics=..., perViewErrors=..., flags=..., criteria=...) -> tuple[retval, cameraMatrix, distCoeffs, rvecs, tvecs, stdDeviationsIntrinsics, stdDeviationsExtrinsics, perViewErrors]:
    """
    * @brief Calibrate a camera using Charuco corners

    @param charucoCorners vector of detected charuco corners per frame
    @param charucoIds list of identifiers for each corner in charucoCorners per frame
    @param board Marker Board layout
    @param imageSize input image size
    @param cameraMatrix Output 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$ . If CV\_CALIB\_USE\_INTRINSIC\_GUESS
    and/or CV_CALIB_FIX_ASPECT_RATIO are specified, some or all of fx, fy, cx, cy must be
    initialized before calling the function.
    @param distCoeffs Output vector of distortion coefficients
    \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
    @param rvecs Output vector of rotation vectors (see Rodrigues ) estimated for each board view
    (e.g. std::vector<cv::Mat>>). That is, each k-th rotation vector together with the corresponding
    k-th translation vector (see the next output parameter description) brings the board pattern
    from the model coordinate space (in which object points are specified) to the world coordinate
    space, that is, a real position of the board pattern in the k-th pattern view (k=0.. *M* -1).
    @param tvecs Output vector of translation vectors estimated for each pattern view.
    @param stdDeviationsIntrinsics Output vector of standard deviations estimated for intrinsic parameters.
    Order of deviations values:
    \f$(f_x, f_y, c_x, c_y, k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6 , s_1, s_2, s_3,
    s_4, \tau_x, \tau_y)\f$ If one of parameters is not estimated, it's deviation is equals to zero.
    @param stdDeviationsExtrinsics Output vector of standard deviations estimated for extrinsic parameters.
    Order of deviations values: \f$(R_1, T_1, \dotsc , R_M, T_M)\f$ where M is number of pattern views,
    \f$R_i, T_i\f$ are concatenated 1x3 vectors.
    @param perViewErrors Output vector of average re-projection errors estimated for each pattern view.
    @param flags flags Different flags  for the calibration process (see #calibrateCamera for details).
    @param criteria Termination criteria for the iterative optimization algorithm.

    This function calibrates a camera using a set of corners of a  Charuco Board. The function
    receives a list of detected corners and its identifiers from several views of the Board.
    The function returns the final re-projection error.
    """

def detectCharucoDiamond(image, markerCorners, markerIds, squareMarkerLengthRate, diamondCorners=..., diamondIds=..., cameraMatrix=..., distCoeffs=..., dictionary=...) -> tuple[diamondCorners, diamondIds]:
    """
    * @brief Detect ChArUco Diamond markers

    @param image input image necessary for corner subpixel.
    @param markerCorners list of detected marker corners from detectMarkers function.
    @param markerIds list of marker ids in markerCorners.
    @param squareMarkerLengthRate rate between square and marker length:
    squareMarkerLengthRate = squareLength/markerLength. The real units are not necessary.
    @param diamondCorners output list of detected diamond corners (4 corners per diamond). The order
    is the same than in marker corners: top left, top right, bottom right and bottom left. Similar
    format than the corners returned by detectMarkers (e.g std::vector<std::vector<cv::Point2f> > ).
    @param diamondIds ids of the diamonds in diamondCorners. The id of each diamond is in fact of
    type Vec4i, so each diamond has 4 ids, which are the ids of the aruco markers composing the
    diamond.
    @param cameraMatrix Optional camera calibration matrix.
    @param distCoeffs Optional camera distortion coefficients.
    @param dictionary dictionary of markers indicating the type of markers.

    This function detects Diamond markers from the previous detected ArUco markers. The diamonds
    are returned in the diamondCorners and diamondIds parameters. If camera calibration parameters
    are provided, the diamond search is based on reprojection. If not, diamond search is based on
    homography. Homography is faster than reprojection, but less accurate.

    @deprecated Use CharucoDetector::detectDiamonds
    """

def detectMarkers(image, dictionary, corners=..., ids=..., parameters=..., rejectedImgPoints=...) -> tuple[corners, ids, rejectedImgPoints]:
    """
    @brief detect markers
    @deprecated Use class ArucoDetector::detectMarkers
    """

def drawCharucoDiamond(dictionary, ids, squareLength, markerLength, img=..., marginSize=..., borderBits=...) -> img:
    """
    * @brief Draw a ChArUco Diamond marker

    @param dictionary dictionary of markers indicating the type of markers.
    @param ids list of 4 ids for each ArUco marker in the ChArUco marker.
    @param squareLength size of the chessboard squares in pixels.
    @param markerLength size of the markers in pixels.
    @param img output image with the marker. The size of this image will be
    3*squareLength + 2*marginSize,.
    @param marginSize minimum margins (in pixels) of the marker in the output image
    @param borderBits width of the marker borders.

    This function return the image of a ChArUco marker, ready to be printed.
    """

def drawDetectedCornersCharuco(image, charucoCorners, charucoIds=..., cornerColor=...) -> image:
    """
    * @brief Draws a set of Charuco corners
    @param image input/output image. It must have 1 or 3 channels. The number of channels is not
    altered.
    @param charucoCorners vector of detected charuco corners
    @param charucoIds list of identifiers for each corner in charucoCorners
    @param cornerColor color of the square surrounding each corner

    This function draws a set of detected Charuco corners. If identifiers vector is provided, it also
    draws the id of each corner.
    """

def drawDetectedDiamonds(image, diamondCorners, diamondIds=..., borderColor=...) -> image:
    """
    * @brief Draw a set of detected ChArUco Diamond markers

    @param image input/output image. It must have 1 or 3 channels. The number of channels is not
    altered.
    @param diamondCorners positions of diamond corners in the same format returned by
    detectCharucoDiamond(). (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
    the dimensions of this array should be Nx4. The order of the corners should be clockwise.
    @param diamondIds vector of identifiers for diamonds in diamondCorners, in the same format
    returned by detectCharucoDiamond() (e.g. std::vector<Vec4i>).
    Optional, if not provided, ids are not painted.
    @param borderColor color of marker borders. Rest of colors (text color and first corner color)
    are calculated based on this one.

    Given an array of detected diamonds, this functions draws them in the image. The marker borders
    are painted and the markers identifiers if provided.
    Useful for debugging purposes.
    """

def drawDetectedMarkers(image, corners, ids=..., borderColor=...) -> image:
    """
    @brief Draw detected markers in image

    @param image input/output image. It must have 1 or 3 channels. The number of channels is not altered.
    @param corners positions of marker corners on input image.
    (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
    this array should be Nx4. The order of the corners should be clockwise.
    @param ids vector of identifiers for markers in markersCorners .
    Optional, if not provided, ids are not painted.
    @param borderColor color of marker borders. Rest of colors (text color and first corner color)
    are calculated based on this one to improve visualization.

    Given an array of detected marker corners and its corresponding ids, this functions draws
    the markers in the image. The marker borders are painted and the markers identifiers if provided.
    Useful for debugging purposes.
    """

def drawPlanarBoard(board, outSize, marginSize, borderBits, img=...) -> img:
    """
    @brief draw planar board
    @deprecated Use Board::generateImage
    """

def estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=...) -> tuple[retval, rvec, tvec]:
    """
    @deprecated Use cv::solvePnP
    """

def estimatePoseCharucoBoard(charucoCorners, charucoIds, board, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess=...) -> tuple[retval, rvec, tvec]:
    """
    * @brief Pose estimation for a ChArUco board given some of their corners
    @param charucoCorners vector of detected charuco corners
    @param charucoIds list of identifiers for each corner in charucoCorners
    @param board layout of ChArUco board.
    @param cameraMatrix input 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
    @param distCoeffs vector of distortion coefficients
    \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
    @param rvec Output vector (e.g. cv::Mat) corresponding to the rotation vector of the board
    (see cv::Rodrigues).
    @param tvec Output vector (e.g. cv::Mat) corresponding to the translation vector of the board.
    @param useExtrinsicGuess defines whether initial guess for \b rvec and \b tvec will be used or not.

    This function estimates a Charuco board pose from some detected corners.
    The function checks if the input corners are enough and valid to perform pose estimation.
    If pose estimation is valid, returns true, else returns false.
    @sa use cv::drawFrameAxes to get world coordinate system axis for object points
    """

def estimatePoseSingleMarkers(corners, markerLength, cameraMatrix, distCoeffs, rvecs=..., tvecs=..., objPoints=..., estimateParameters=...) -> tuple[rvecs, tvecs, objPoints]:
    """
    @deprecated Use cv::solvePnP
    """

def extendDictionary(nMarkers, markerSize, baseDictionary=..., randomSeed=...) -> retval:
    """
    @brief Extend base dictionary by new nMarkers
      *
      * @param nMarkers number of markers in the dictionary
      * @param markerSize number of bits per dimension of each markers
      * @param baseDictionary Include the markers in this dictionary at the beginning (optional)
      * @param randomSeed a user supplied seed for theRNG()
      *
      * This function creates a new dictionary composed by nMarkers markers and each markers composed
      * by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly
      * included and the rest are generated based on them. If the size of baseDictionary is higher
      * than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added.
    """

def generateImageMarker(dictionary, id, sidePixels, img=..., borderBits=...) -> img:
    """
    @brief Generate a canonical marker image

    @param dictionary dictionary of markers indicating the type of markers
    @param id identifier of the marker that will be returned. It has to be a valid id in the specified dictionary.
    @param sidePixels size of the image in pixels
    @param img output image with the marker
    @param borderBits width of the marker border.

    This function returns a marker image in its canonical form (i.e. ready to be printed)
    """

def getBoardObjectAndImagePoints(board, detectedCorners, detectedIds, objPoints=..., imgPoints=...) -> tuple[objPoints, imgPoints]:
    """
    @brief get board object and image points
    @deprecated Use Board::matchImagePoints
    """

def getPredefinedDictionary(dict) -> retval:
    """
    @brief Returns one of the predefined dictionaries referenced by DICT_*.
    """

def interpolateCornersCharuco(markerCorners, markerIds, image, board, charucoCorners=..., charucoIds=..., cameraMatrix=..., distCoeffs=..., minMarkers=...) -> tuple[retval, charucoCorners, charucoIds]:
    """
    * @brief Interpolate position of ChArUco board corners
    @param markerCorners vector of already detected markers corners. For each marker, its four
    corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the
    dimensions of this array should be Nx4. The order of the corners should be clockwise.
    @param markerIds list of identifiers for each marker in corners
    @param image input image necesary for corner refinement. Note that markers are not detected and
    should be sent in corners and ids parameters.
    @param board layout of ChArUco board.
    @param charucoCorners interpolated chessboard corners
    @param charucoIds interpolated chessboard corners identifiers
    @param cameraMatrix optional 3x3 floating-point camera matrix
    \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
    @param distCoeffs optional vector of distortion coefficients
    \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
    @param minMarkers number of adjacent markers that must be detected to return a charuco corner

    This function receives the detected markers and returns the 2D position of the chessboard corners
    from a ChArUco board using the detected Aruco markers. If camera parameters are provided,
    the process is based in an approximated pose estimation, else it is based on local homography.
    Only visible corners are returned. For each corner, its corresponding identifier is
    also returned in charucoIds.
    The function returns the number of interpolated corners.

    @deprecated Use CharucoDetector::detectBoard
    """

def refineDetectedMarkers(image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix=..., distCoeffs=..., minRepDistance=..., errorCorrectionRate=..., checkAllOrders=..., recoveredIdxs=..., parameters=...) -> tuple[detectedCorners, detectedIds, rejectedCorners, recoveredIdxs]:
    """
    @brief refine detected markers
    @deprecated Use class ArucoDetector::refineDetectedMarkers
    """

def testCharucoCornersCollinear(board, charucoIds) -> retval:
    """
    @deprecated Use CharucoBoard::checkCharucoCornersCollinear
    """

ARUCO_CCW_CENTER: Final[int]
ARUCO_CW_TOP_LEFT_CORNER: Final[int]
CORNER_REFINE_APRILTAG: Final[int]
CORNER_REFINE_CONTOUR: Final[int]
CORNER_REFINE_NONE: Final[int]
CORNER_REFINE_SUBPIX: Final[int]
DICT_4X4_100: int
DICT_4X4_1000: int
DICT_4X4_250: int
DICT_4X4_50: int
DICT_5X5_100: int
DICT_5X5_1000: int
DICT_5X5_250: int
DICT_5X5_50: int
DICT_6X6_100: int
DICT_6X6_1000: int
DICT_6X6_250: int
DICT_6X6_50: int
DICT_7X7_100: int
DICT_7X7_1000: int
DICT_7X7_250: int
DICT_7X7_50: int
DICT_APRILTAG_16H5: Final[int]
DICT_APRILTAG_16h5: Final[int]
DICT_APRILTAG_25H9: Final[int]
DICT_APRILTAG_25h9: Final[int]
DICT_APRILTAG_36H10: int
DICT_APRILTAG_36H11: Final[int]
DICT_APRILTAG_36h10: int
DICT_APRILTAG_36h11: Final[int]
DICT_ARUCO_ORIGINAL: Final[int]
