from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

dstStep2: TypeAlias = Any
dstStep1: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any

class GrayworldWB(WhiteBalancer):
    def getSaturationThreshold(self) -> retval:
        """
        @brief Maximum saturation for a pixel to be included in the
        gray-world assumption
        @see setSaturationThreshold
        """

    def setSaturationThreshold(self, val) -> None:
        """
        @copybrief getSaturationThreshold @see getSaturationThreshold
        """

class LearningBasedWB(WhiteBalancer):
    def extractSimpleFeatures(self, src, dst=...) -> dst:
        """
        @brief Implements the feature extraction part of the algorithm.

        In accordance with @cite Cheng2015 , computes the following features for the input image:
        1. Chromaticity of an average (R,G,B) tuple
        2. Chromaticity of the brightest (R,G,B) tuple (while ignoring saturated pixels)
        3. Chromaticity of the dominant (R,G,B) tuple (the one that has the highest value in the RGB histogram)
        4. Mode of the chromaticity palette, that is constructed by taking 300 most common colors according to
        the RGB histogram and projecting them on the chromaticity plane. Mode is the most high-density point
        of the palette, which is computed by a straightforward fixed-bandwidth kernel density estimator with
        a Epanechnikov kernel function.

        @param src Input three-channel image (BGR color space is assumed).
        @param dst An array of four (r,g) chromaticity tuples corresponding to the features listed above.
        """

    def getHistBinNum(self) -> retval:
        """
        @brief Defines the size of one dimension of a three-dimensional RGB histogram that is used internally
        by the algorithm. It often makes sense to increase the number of bins for images with higher bit depth
        (e.g. 256 bins for a 12 bit image).
        @see setHistBinNum
        """

    def getRangeMaxVal(self) -> retval:
        """
        @brief Maximum possible value of the input image (e.g. 255 for 8 bit images,
        4095 for 12 bit images)
        @see setRangeMaxVal
        """

    def getSaturationThreshold(self) -> retval:
        """
        @brief Threshold that is used to determine saturated pixels, i.e. pixels where at least one of the
        channels exceeds \f$\texttt{saturation_threshold}\times\texttt{range_max_val}\f$ are ignored.
        @see setSaturationThreshold
        """

    def setHistBinNum(self, val) -> None:
        """
        @copybrief getHistBinNum @see getHistBinNum
        """

    def setRangeMaxVal(self, val) -> None:
        """
        @copybrief getRangeMaxVal @see getRangeMaxVal
        """

    def setSaturationThreshold(self, val) -> None:
        """
        @copybrief getSaturationThreshold @see getSaturationThreshold
        """

class SimpleWB(WhiteBalancer):
    def getInputMax(self) -> retval:
        """
        @brief Input image range maximum value
        @see setInputMax
        """

    def getInputMin(self) -> retval:
        """
        @brief Input image range minimum value
        @see setInputMin
        """

    def getOutputMax(self) -> retval:
        """
        @brief Output image range maximum value
        @see setOutputMax
        """

    def getOutputMin(self) -> retval:
        """
        @brief Output image range minimum value
        @see setOutputMin
        """

    def getP(self) -> retval:
        """
        @brief Percent of top/bottom values to ignore
        @see setP
        """

    def setInputMax(self, val) -> None:
        """
        @copybrief getInputMax @see getInputMax
        """

    def setInputMin(self, val) -> None:
        """
        @copybrief getInputMin @see getInputMin
        """

    def setOutputMax(self, val) -> None:
        """
        @copybrief getOutputMax @see getOutputMax
        """

    def setOutputMin(self, val) -> None:
        """
        @copybrief getOutputMin @see getOutputMin
        """

    def setP(self, val) -> None:
        """
        @copybrief getP @see getP
        """

class TonemapDurand(cv2.Tonemap):
    def getContrast(self) -> retval:
        """"""

    def getSaturation(self) -> retval:
        """"""

    def getSigmaColor(self) -> retval:
        """"""

    def getSigmaSpace(self) -> retval:
        """"""

    def setContrast(self, contrast) -> None:
        """"""

    def setSaturation(self, saturation) -> None:
        """"""

    def setSigmaColor(self, sigma_color) -> None:
        """"""

    def setSigmaSpace(self, sigma_space) -> None:
        """"""

class WhiteBalancer(cv2.Algorithm):
    def balanceWhite(self, src, dst=...) -> dst:
        """
        @brief Applies white balancing to the input image

        @param src Input image
        @param dst White balancing result @sa cvtColor, equalizeHist
        """

@overload
def applyChannelGains(src, gainB, gainG, gainR, dst=...) -> dst:
    """
    @brief Implements an efficient fixed-point approximation for applying channel gains, which is
    """

@overload
def applyChannelGains(src, gainB, gainG, gainR, dst=...) -> dst:
    """

    @param src Input three-channel image in the BGR color space (either CV_8UC3 or CV_16UC3)
    @param dst Output image of the same size and type as src.
    @param gainB gain for the B channel
    @param gainG gain for the G channel
    @param gainR gain for the R channel
    """

@overload
def bm3dDenoising(src, dstStep1, dstStep2=..., h=..., templateWindowSize=..., searchWindowSize=..., blockMatchingStep1=..., blockMatchingStep2=..., groupSize=..., slidingStep=..., beta=..., normType=..., step=..., transformType=...) -> tuple[dstStep1, dstStep2]:
    """
    @brief Performs image denoising using the Block-Matching and 3D-filtering algorithm
            <http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D_TIP_2007.pdf> with several computational
            optimizations. Noise expected to be a gaussian white noise.

            @param src Input 8-bit or 16-bit 1-channel image.
            @param dstStep1 Output image of the first step of BM3D with the same size and type as src.
            @param dstStep2 Output image of the second step of BM3D with the same size and type as src.
            @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
            removes image details, smaller h value preserves details but also preserves some noise.
            @param templateWindowSize Size in pixels of the template patch that is used for block-matching.
            Should be power of 2.
            @param searchWindowSize Size in pixels of the window that is used to perform block-matching.
            Affect performance linearly: greater searchWindowsSize - greater denoising time.
            Must be larger than templateWindowSize.
            @param blockMatchingStep1 Block matching threshold for the first step of BM3D (hard thresholding),
            i.e. maximum distance for which two blocks are considered similar.
            Value expressed in euclidean distance.
            @param blockMatchingStep2 Block matching threshold for the second step of BM3D (Wiener filtering),
            i.e. maximum distance for which two blocks are considered similar.
            Value expressed in euclidean distance.
            @param groupSize Maximum size of the 3D group for collaborative filtering.
            @param slidingStep Sliding step to process every next reference block.
            @param beta Kaiser window parameter that affects the sidelobe attenuation of the transform of the
            window. Kaiser window is used in order to reduce border effects. To prevent usage of the window,
            set beta to zero.
            @param normType Norm used to calculate distance between blocks. L2 is slower than L1
            but yields more accurate results.
            @param step Step of BM3D to be executed. Possible variants are: step 1, step 2, both steps.
            @param transformType Type of the orthogonal transform used in collaborative filtering step.
            Currently only Haar transform is supported.

            This function expected to be applied to grayscale images. Advanced usage of this function
            can be manual denoising of colored image in different colorspaces.

            @sa
            fastNlMeansDenoising
    """

@overload
def bm3dDenoising(src, dstStep1, dstStep2=..., h=..., templateWindowSize=..., searchWindowSize=..., blockMatchingStep1=..., blockMatchingStep2=..., groupSize=..., slidingStep=..., beta=..., normType=..., step=..., transformType=...) -> tuple[dstStep1, dstStep2]:
    """
    @brief Performs image denoising using the Block-Matching and 3D-filtering algorithm
            <http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D_TIP_2007.pdf> with several computational
            optimizations. Noise expected to be a gaussian white noise.

            @param src Input 8-bit or 16-bit 1-channel image.
            @param dst Output image with the same size and type as src.
            @param h Parameter regulating filter strength. Big h value perfectly removes noise but also
            removes image details, smaller h value preserves details but also preserves some noise.
            @param templateWindowSize Size in pixels of the template patch that is used for block-matching.
            Should be power of 2.
            @param searchWindowSize Size in pixels of the window that is used to perform block-matching.
            Affect performance linearly: greater searchWindowsSize - greater denoising time.
            Must be larger than templateWindowSize.
            @param blockMatchingStep1 Block matching threshold for the first step of BM3D (hard thresholding),
            i.e. maximum distance for which two blocks are considered similar.
            Value expressed in euclidean distance.
            @param blockMatchingStep2 Block matching threshold for the second step of BM3D (Wiener filtering),
            i.e. maximum distance for which two blocks are considered similar.
            Value expressed in euclidean distance.
            @param groupSize Maximum size of the 3D group for collaborative filtering.
            @param slidingStep Sliding step to process every next reference block.
            @param beta Kaiser window parameter that affects the sidelobe attenuation of the transform of the
            window. Kaiser window is used in order to reduce border effects. To prevent usage of the window,
            set beta to zero.
            @param normType Norm used to calculate distance between blocks. L2 is slower than L1
            but yields more accurate results.
            @param step Step of BM3D to be executed. Allowed are only BM3D_STEP1 and BM3D_STEPALL.
            BM3D_STEP2 is not allowed as it requires basic estimate to be present.
            @param transformType Type of the orthogonal transform used in collaborative filtering step.
            Currently only Haar transform is supported.

            This function expected to be applied to grayscale images. Advanced usage of this function
            can be manual denoising of colored image in different colorspaces.

            @sa
            fastNlMeansDenoising
    """

def createGrayworldWB() -> retval:
    """
    @brief Creates an instance of GrayworldWB
    """

def createLearningBasedWB(path_to_model=...) -> retval:
    """
    @brief Creates an instance of LearningBasedWB

    @param path_to_model Path to a .yml file with the model. If not specified, the default model is used
    """

def createSimpleWB() -> retval:
    """
    @brief Creates an instance of SimpleWB
    """

def createTonemapDurand(gamma=..., contrast=..., saturation=..., sigma_color=..., sigma_space=...) -> retval:
    """
    @brief Creates TonemapDurand object

    You need to set the OPENCV_ENABLE_NONFREE option in cmake to use those. Use them at your own risk.

    @param gamma gamma value for gamma correction. See createTonemap
    @param contrast resulting contrast on logarithmic scale, i. e. log(max / min), where max and min
    are maximum and minimum luminance values of the resulting image.
    @param saturation saturation enhancement value. See createTonemapDrago
    @param sigma_color bilateral filter sigma in color space
    @param sigma_space bilateral filter sigma in coordinate space
    """

def dctDenoising(src, dst, sigma, psize=...) -> None:
    """
    @brief The function implements simple dct-based denoising

        <http://www.ipol.im/pub/art/2011/ys-dct/>.
        @param src source image
        @param dst destination image
        @param sigma expected noise standard deviation
        @param psize size of block side where dct is computed

        @sa
           fastNlMeansDenoising
    """

@overload
def inpaint(src, mask, dst, algorithmType) -> None:
    """
    @brief The function implements different single-image inpainting algorithms.
    """

@overload
def inpaint(src, mask, dst, algorithmType) -> None:
    """

    @param src source image
    - #INPAINT_SHIFTMAP: it could be of any type and any number of channels from 1 to 4. In case of
    3- and 4-channels images the function expect them in CIELab colorspace or similar one, where first
    """

@overload
def inpaint(src, mask, dst, algorithmType) -> None:
    """ """

@overload
def inpaint(src, mask, dst, algorithmType) -> None:
    """
    - #INPAINT_FSR_BEST or #INPAINT_FSR_FAST: 1-channel grayscale or 3-channel BGR image.
    @param mask mask (#CV_8UC1), where non-zero pixels indicate valid image area, while zero pixels
    """

@overload
def inpaint(src, mask, dst, algorithmType) -> None:
    """
    @param dst destination image
    @param algorithmType see xphoto::InpaintTypes
    """

@overload
def oilPainting(src, size, dynRatio, code, dst=...) -> dst:
    """
    @brief oilPainting
    See the book @cite Holzmann1988 for details.
    @param src Input three-channel or one channel image (either CV_8UC3 or CV_8UC1)
    @param dst Output image of the same size and type as src.
    @param size neighbouring size is 2-size+1
    @param dynRatio image is divided by dynRatio before histogram processing
    @param code color space conversion code(see ColorConversionCodes). Histogram will used only first plane
    """

@overload
def oilPainting(src, size, dynRatio, code, dst=...) -> dst:
    """
    @brief oilPainting
    See the book @cite Holzmann1988 for details.
    @param src Input three-channel or one channel image (either CV_8UC3 or CV_8UC1)
    @param dst Output image of the same size and type as src.
    @param size neighbouring size is 2-size+1
    @param dynRatio image is divided by dynRatio before histogram processing
    """

BM3D_STEP1: Final[int]
BM3D_STEP2: Final[int]
BM3D_STEPALL: Final[int]
HAAR: Final[int]
INPAINT_FSR_BEST: Final[int]
INPAINT_FSR_FAST: Final[int]
INPAINT_SHIFTMAP: Final[int]
