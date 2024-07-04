import builtins
from typing import Any, Final, TypeAlias

sMatches: TypeAlias = Any
denseMatches: TypeAlias = Any
retval: TypeAlias = Any

class MatchQuasiDense(builtins.object):
    def apply(self, rhs) -> retval:
        """"""

class PropagationParameters(builtins.object): ...

class QuasiDenseStereo(builtins.object):
    def getDenseMatches(self) -> denseMatches:
        """
        * @brief Get The dense corresponding points.
        * @param[out] denseMatches A vector containing all dense matches. * @note The method clears the denseMatches vector. * @note The returned Match elements inside the sMatches vector, do not use corr member.
        """

    def getDisparity(self) -> retval:
        """
        * @brief Compute and return the disparity map based on the correspondences found in the "process" method.
        * @note Default level is 50
        * @return cv::Mat containing a the disparity image in grayscale.
        * @sa computeDisparity
        * @sa quantizeDisparity
        """

    def getMatch(self, x, y) -> retval:
        """
        * @brief Specify pixel coordinates in the left image and get its corresponding location in the right image.
        * @param[in] x The x pixel coordinate in the left image channel.
        * @param[in] y The y pixel coordinate in the left image channel. * @retval cv::Point(x, y) The location of the corresponding pixel in the right image. * @retval cv::Point(0, 0) (NO_MATCH)  if no match is found in the right image for the specified pixel location in the left image. * @note This method should be always called after process, otherwise the matches will not be correct.
        """

    def getSparseMatches(self) -> sMatches:
        """
        * @brief Get The sparse corresponding points.
        * @param[out] sMatches A vector containing all sparse correspondences. * @note The method clears the sMatches vector. * @note The returned Match elements inside the sMatches vector, do not use corr member.
        """

    def loadParameters(self, filepath) -> retval:
        """
        * @brief Load a file containing the configuration parameters of the class.
        * @param[in] filepath The location of the .YAML file containing the configuration parameters. * @note default value is an empty string in which case the default parameters will be loaded. * @retval 1: If the path is not empty and the program loaded the parameters successfully. * @retval 0: If the path is empty and the program loaded default parameters. * @retval -1: If the file location is not valid or the program could not open the file and * loaded default parameters from defaults.hpp. * @note The method is automatically called in the constructor and configures the class. * @note Loading different parameters will have an effect on the output. This is useful for tuning * in case of video processing. * @sa loadParameters
        """

    def process(self, imgLeft, imgRight) -> None:
        """
        * @brief Main process of the algorithm. This method computes the sparse seeds and then densifies them.
        *
        * Initially input images are converted to gray-scale and then the sparseMatching method
        * is called to obtain the sparse stereo. Finally quasiDenseMatching is called to densify the corresponding
        * points.
        * @param[in] imgLeft The left Channel of a stereo image pair.
        * @param[in] imgRight The right Channel of a stereo image pair. * @note If input images are in color, the method assumes that are BGR and converts them to grayscale. * @sa sparseMatching * @sa quasiDenseMatching
        """

    def saveParameters(self, filepath) -> retval:
        """
        * @brief Save a file containing all the configuration parameters the class is currently set to.
        * @param[in] filepath The location to store the parameters file. * @note Calling this method with no arguments will result in storing class parameters to a file * names "qds_parameters.yaml" in the root project folder. * @note This method can be used to generate a template file for tuning the class. * @sa loadParameters
        """

    def create(self, monoImgSize, paramFilepath=...) -> retval:
        """"""

def QuasiDenseStereo_create(monoImgSize, paramFilepath=...) -> retval:
    """
    .
    """

CV_CS_CENSUS: Final[int]
CV_DENSE_CENSUS: Final[int]
CV_MEAN_VARIATION: Final[int]
CV_MODIFIED_CENSUS_TRANSFORM: Final[int]
CV_MODIFIED_CS_CENSUS: Final[int]
CV_QUADRATIC_INTERPOLATION: Final[int]
CV_SIMETRICV_INTERPOLATION: Final[int]
CV_SPARSE_CENSUS: Final[int]
CV_SPECKLE_REMOVAL_ALGORITHM: Final[int]
CV_SPECKLE_REMOVAL_AVG_ALGORITHM: Final[int]
CV_STAR_KERNEL: Final[int]
STEREO_BINARY_BM_PREFILTER_NORMALIZED_RESPONSE: Final[int]
STEREO_BINARY_BM_PREFILTER_XSOBEL: Final[int]
STEREO_BINARY_SGBM_MODE_HH: Final[int]
STEREO_BINARY_SGBM_MODE_SGBM: Final[int]
STEREO_MATCHER_DISP_SCALE: Final[int]
STEREO_MATCHER_DISP_SHIFT: Final[int]
StereoBinaryBM_PREFILTER_NORMALIZED_RESPONSE: Final[int]
StereoBinaryBM_PREFILTER_XSOBEL: Final[int]
StereoBinarySGBM_MODE_HH: Final[int]
StereoBinarySGBM_MODE_SGBM: Final[int]
StereoMatcher_DISP_SCALE: Final[int]
StereoMatcher_DISP_SHIFT: Final[int]
