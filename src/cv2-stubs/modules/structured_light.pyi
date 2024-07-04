from typing import Any, Final, TypeAlias

from .. import functions as cv2

wrappedPhaseMap: TypeAlias = Any
whiteImage: TypeAlias = Any
patternImages: TypeAlias = Any
disparityMap: TypeAlias = Any
dataModulationTerm: TypeAlias = Any
shadowMask: TypeAlias = Any
unwrappedPhaseMap: TypeAlias = Any
projPix: TypeAlias = Any
matches: TypeAlias = Any
blackImage: TypeAlias = Any
retval: TypeAlias = Any

class GrayCodePattern(StructuredLightPattern):
    def getImagesForShadowMasks(self, blackImage, whiteImage) -> tuple[blackImage, whiteImage]:
        """
        @brief Generates the all-black and all-white images needed for shadowMasks computation.
        *
        *  To identify shadow regions, the regions of two images where the pixels are not lit by projector's light and thus where there is not coded information,
        *  the 3DUNDERWORLD algorithm computes a shadow mask for the two cameras views, starting from a white and a black images captured by each camera.
        *  This method generates these two additional images to project.
        *
        *  @param blackImage The generated all-black CV_8U image, at projector's resolution.
        *  @param whiteImage The generated all-white CV_8U image, at projector's resolution.
        """

    def getNumberOfPatternImages(self) -> retval:
        """
        @brief Get the number of pattern images needed for the graycode pattern.
        *
        * @return The number of pattern images needed for the graycode pattern.
        *
        """

    def getProjPixel(self, patternImages, x, y) -> tuple[retval, projPix]:
        """
        @brief For a (x,y) pixel of a camera returns the corresponding projector pixel.
        *
        *  The function decodes each pixel in the pattern images acquired by a camera into their corresponding decimal numbers representing the projector's column and row,
        *  providing a mapping between camera's and projector's pixel.
        *
        *  @param patternImages The pattern images acquired by the camera, stored in a grayscale vector < Mat >.
        *  @param x x coordinate of the image pixel.
        *  @param y y coordinate of the image pixel.
        *  @param projPix Projector's pixel corresponding to the camera's pixel: projPix.x and projPix.y are the image coordinates of the projector's pixel corresponding to the pixel being decoded in a camera.
        """

    def setBlackThreshold(self, value) -> None:
        """
        @brief Sets the value for black threshold, needed for decoding (shadowsmasks computation).
        *
        *  Black threshold is a number between 0-255 that represents the minimum brightness difference required for valid pixels, between the fully illuminated (white) and the not illuminated images (black); used in computeShadowMasks method.
        *
        *  @param value The desired black threshold value. *
        """

    def setWhiteThreshold(self, value) -> None:
        """
        @brief Sets the value for white threshold, needed for decoding.
        *
        *  White threshold is a number between 0-255 that represents the minimum brightness difference required for valid pixels, between the graycode pattern and its inverse images; used in getProjPixel method.
        *
        *  @param value The desired white threshold value. *
        """

    def create(self, width, height) -> retval:
        """
        @brief Constructor
        @param parameters GrayCodePattern parameters GrayCodePattern::Params: the width and the height of the projector.
        """

class SinusoidalPattern(StructuredLightPattern):
    def computeDataModulationTerm(self, patternImages, shadowMask, dataModulationTerm=...) -> dataModulationTerm:
        """
        * @brief compute the data modulation term.
        * @param patternImages captured images with projected patterns.
        * @param dataModulationTerm Mat where the data modulation term is saved.
        * @param shadowMask Mask used to discard shadow regions.
        """

    def computePhaseMap(self, patternImages, wrappedPhaseMap=..., shadowMask=..., fundamental=...) -> tuple[wrappedPhaseMap, shadowMask]:
        """
        * @brief Compute a wrapped phase map from sinusoidal patterns.
        * @param patternImages Input data to compute the wrapped phase map.
        * @param wrappedPhaseMap Wrapped phase map obtained through one of the three methods.
        * @param shadowMask Mask used to discard shadow regions.
        * @param fundamental Fundamental matrix used to compute epipolar lines and ease the matching step.
        """

    def findProCamMatches(self, projUnwrappedPhaseMap, camUnwrappedPhaseMap, matches=...) -> matches:
        """
        * @brief Find correspondences between the two devices thanks to unwrapped phase maps.
        * @param projUnwrappedPhaseMap Projector's unwrapped phase map.
        * @param camUnwrappedPhaseMap Camera's unwrapped phase map.
        * @param matches Images used to display correspondences map.
        """

    def unwrapPhaseMap(self, wrappedPhaseMap, camSize, unwrappedPhaseMap=..., shadowMask=...) -> unwrappedPhaseMap:
        """
        * @brief Unwrap the wrapped phase map to remove phase ambiguities.
        * @param wrappedPhaseMap The wrapped phase map computed from the pattern.
        * @param unwrappedPhaseMap The unwrapped phase map used to find correspondences between the two devices.
        * @param camSize Resolution of the camera.
        * @param shadowMask Mask used to discard shadow regions.
        """

    def create(self, parameters=...) -> retval:
        """
        * @brief Constructor.
        * @param parameters SinusoidalPattern parameters SinusoidalPattern::Params: width, height of the projector and patterns parameters. *
        """

class StructuredLightPattern(cv2.Algorithm):
    def decode(self, patternImages, disparityMap=..., blackImages=..., whiteImages=..., flags=...) -> tuple[retval, disparityMap]:
        """
        @brief Decodes the structured light pattern, generating a disparity map

        @param patternImages The acquired pattern images to decode (vector<vector<Mat>>), loaded as grayscale and previously rectified.
        @param disparityMap The decoding result: a CV_64F Mat at image resolution, storing the computed disparity map.
        @param blackImages The all-black images needed for shadowMasks computation.
        @param whiteImages The all-white images needed for shadowMasks computation.
        @param flags Flags setting decoding algorithms. Default: DECODE_3D_UNDERWORLD. @note All the images must be at the same resolution.
        """

    def generate(self, patternImages=...) -> tuple[retval, patternImages]:
        """
        @brief Generates the structured light pattern to project.

        @param patternImages The generated pattern: a vector<Mat>, in which each image is a CV_8U Mat at projector's resolution.
        """

def GrayCodePattern_create(width, height) -> retval:
    """
    @brief Constructor
       @param parameters GrayCodePattern parameters GrayCodePattern::Params: the width and the height of the projector.
    """

def SinusoidalPattern_create(parameters=...) -> retval:
    """
    * @brief Constructor.
         * @param parameters SinusoidalPattern parameters SinusoidalPattern::Params: width, height of the projector and patterns parameters.
         *
    """

DECODE_3D_UNDERWORLD: Final[int]
FAPS: Final[int]
FTP: Final[int]
PSP: Final[int]
