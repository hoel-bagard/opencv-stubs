from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

outputToneMappedImage: TypeAlias = Any
retinaOutput_parvo: TypeAlias = Any
retinaOutput_magno: TypeAlias = Any
transientAreas: TypeAlias = Any
retval: TypeAlias = Any

class Retina(cv2.Algorithm):
    def activateContoursProcessing(self, activate) -> None:
        """
        @brief Activate/desactivate the Parvocellular pathway processing (contours information extraction), by
        default, it is activated
        @param activate true if Parvocellular (contours information extraction) output should be activated, false if not... if activated, the Parvocellular output can be retrieved using the Retina::getParvo methods
        """

    def activateMovingContoursProcessing(self, activate) -> None:
        """
        @brief Activate/desactivate the Magnocellular pathway processing (motion information extraction), by
        default, it is activated
        @param activate true if Magnocellular output should be activated, false if not... if activated, the Magnocellular output can be retrieved using the **getMagno** methods
        """

    def applyFastToneMapping(self, inputImage, outputToneMappedImage=...) -> outputToneMappedImage:
        """
        @brief Method which processes an image in the aim to correct its luminance correct
        backlight problems, enhance details in shadows.

        This method is designed to perform High Dynamic Range image tone mapping (compress \>8bit/pixel
        images to 8bit/pixel). This is a simplified version of the Retina Parvocellular model
        (simplified version of the run/getParvo methods call) since it does not include the
        spatio-temporal filter modelling the Outer Plexiform Layer of the retina that performs spectral
        whitening and many other stuff. However, it works great for tone mapping and in a faster way.

        Check the demos and experiments section to see examples and the way to perform tone mapping
        using the original retina model and the method.

        @param inputImage the input image to process (should be coded in float format : CV_32F, CV_32FC1, CV_32F_C3, CV_32F_C4, the 4th channel won't be considered).
        @param outputToneMappedImage the output 8bit/channel tone mapped image (CV_8U or CV_8UC3 format).
        """

    def clearBuffers(self) -> None:
        """
        @brief Clears all retina buffers

        (equivalent to opening the eyes after a long period of eye close ;o) whatchout the temporal
        transition occuring just after this method call.
        """

    def getInputSize(self) -> retval:
        """
        @brief Retreive retina input buffer size
        @return the retina input buffer size
        """

    def getMagno(self, retinaOutput_magno=...) -> retinaOutput_magno:
        """
        @brief Accessor of the motion channel of the retina (models peripheral vision).

        Warning, getMagnoRAW methods return buffers that are not rescaled within range [0;255] while
        the non RAW method allows a normalized matrix to be retrieved.
        @param retinaOutput_magno the output buffer (reallocated if necessary), format can be : -   a Mat, this output is rescaled for standard 8bits image processing use in OpenCV -   RAW methods actually return a 1D matrix (encoding is M1, M2,... Mn), this output is the original retina filter model output, without any quantification or rescaling. @see getMagnoRAW
        """

    @overload
    def getMagnoRAW(self, retinaOutput_magno=...) -> retinaOutput_magno:
        """
        @brief Accessor of the motion channel of the retina (models peripheral vision).
        @see getMagno
        """

    @overload
    def getMagnoRAW(self) -> retval:
        """
        @overload
        """

    def getOutputSize(self) -> retval:
        """
        @brief Retreive retina output buffer size that can be different from the input if a spatial log
        transformation is applied
        @return the retina output buffer size
        """

    def getParvo(self, retinaOutput_parvo=...) -> retinaOutput_parvo:
        """
        @brief Accessor of the details channel of the retina (models foveal vision).

        Warning, getParvoRAW methods return buffers that are not rescaled within range [0;255] while
        the non RAW method allows a normalized matrix to be retrieved.

        @param retinaOutput_parvo the output buffer (reallocated if necessary), format can be : -   a Mat, this output is rescaled for standard 8bits image processing use in OpenCV -   RAW methods actually return a 1D matrix (encoding is R1, R2, ... Rn, G1, G2, ..., Gn, B1, B2, ...Bn), this output is the original retina filter model output, without any quantification or rescaling. @see getParvoRAW
        """

    @overload
    def getParvoRAW(self, retinaOutput_parvo=...) -> retinaOutput_parvo:
        """
        @brief Accessor of the details channel of the retina (models foveal vision).
        @see getParvo
        """

    @overload
    def getParvoRAW(self) -> retval:
        """
        @overload
        """

    def printSetup(self) -> retval:
        """
        @brief Outputs a string showing the used parameters setup
        @return a string which contains formated parameters information
        """

    def run(self, inputImage) -> None:
        """
        @brief Method which allows retina to be applied on an input image,

        after run, encapsulated retina module is ready to deliver its outputs using dedicated
        acccessors, see getParvo and getMagno methods
        @param inputImage the input Mat image to be processed, can be gray level or BGR coded in any format (from 8bit to 16bits)
        """

    def setColorSaturation(self, saturateColors=..., colorSaturationValue=...) -> None:
        """
        @brief Activate color saturation as the final step of the color demultiplexing process -\> this
        saturation is a sigmoide function applied to each channel of the demultiplexed image.
        @param saturateColors boolean that activates color saturation (if true) or desactivate (if false)
        @param colorSaturationValue the saturation factor : a simple factor applied on the chrominance buffers
        """

    def setup(self, retinaParameterFile=..., applyDefaultSetupOnFailure=...) -> None:
        """
        @brief Try to open an XML retina parameters file to adjust current retina instance setup

        - if the xml file does not exist, then default setup is applied
        - warning, Exceptions are thrown if read XML file is not valid
        @param retinaParameterFile the parameters filename
        @param applyDefaultSetupOnFailure set to true if an error must be thrown on error  You can retrieve the current parameters structure using the method Retina::getParameters and update it before running method Retina::setup.
        """

    def setupIPLMagnoChannel(self, normaliseOutput=..., parasolCells_beta=..., parasolCells_tau=..., parasolCells_k=..., amacrinCellsTemporalCutFrequency=..., V0CompressionParameter=..., localAdaptintegration_tau=..., localAdaptintegration_k=...) -> None:
        """
        @brief Set parameters values for the Inner Plexiform Layer (IPL) magnocellular channel

        this channel processes signals output from OPL processing stage in peripheral vision, it allows
        motion information enhancement. It is decorrelated from the details channel. See reference
        papers for more details.

        @param normaliseOutput specifies if (true) output is rescaled between 0 and 255 of not (false)
        @param parasolCells_beta the low pass filter gain used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), typical value is 0
        @param parasolCells_tau the low pass filter time constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is frame, typical value is 0 (immediate response)
        @param parasolCells_k the low pass filter spatial constant used for local contrast adaptation at the IPL level of the retina (for ganglion cells local adaptation), unit is pixels, typical value is 5
        @param amacrinCellsTemporalCutFrequency the time constant of the first order high pass fiter of the magnocellular way (motion information channel), unit is frames, typical value is 1.2
        @param V0CompressionParameter the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.95
        @param localAdaptintegration_tau specifies the temporal constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
        @param localAdaptintegration_k specifies the spatial constant of the low pas filter involved in the computation of the local "motion mean" for the local adaptation computation
        """

    def setupOPLandIPLParvoChannel(self, colorMode=..., normaliseOutput=..., photoreceptorsLocalAdaptationSensitivity=..., photoreceptorsTemporalConstant=..., photoreceptorsSpatialConstant=..., horizontalCellsGain=..., HcellsTemporalConstant=..., HcellsSpatialConstant=..., ganglionCellsSensitivity=...) -> None:
        """
        @brief Setup the OPL and IPL parvo channels (see biologocal model)

        OPL is referred as Outer Plexiform Layer of the retina, it allows the spatio-temporal filtering
        which withens the spectrum and reduces spatio-temporal noise while attenuating global luminance
        (low frequency energy) IPL parvo is the OPL next processing stage, it refers to a part of the
        Inner Plexiform layer of the retina, it allows high contours sensitivity in foveal vision. See
        reference papers for more informations.
        for more informations, please have a look at the paper Benoit A., Caplier A., Durette B., Herault, J., "USING HUMAN VISUAL SYSTEM MODELING FOR BIO-INSPIRED LOW LEVEL IMAGE PROCESSING", Elsevier, Computer Vision and Image Understanding 114 (2010), pp. 758-773, DOI: http://dx.doi.org/10.1016/j.cviu.2010.01.011
        @param colorMode specifies if (true) color is processed of not (false) to then processing gray level image
        @param normaliseOutput specifies if (true) output is rescaled between 0 and 255 of not (false)
        @param photoreceptorsLocalAdaptationSensitivity the photoreceptors sensitivity renage is 0-1 (more log compression effect when value increases)
        @param photoreceptorsTemporalConstant the time constant of the first order low pass filter of the photoreceptors, use it to cut high temporal frequencies (noise or fast motion), unit is frames, typical value is 1 frame
        @param photoreceptorsSpatialConstant the spatial constant of the first order low pass filter of the photoreceptors, use it to cut high spatial frequencies (noise or thick contours), unit is pixels, typical value is 1 pixel
        @param horizontalCellsGain gain of the horizontal cells network, if 0, then the mean value of the output is zero, if the parameter is near 1, then, the luminance is not filtered and is still reachable at the output, typicall value is 0
        @param HcellsTemporalConstant the time constant of the first order low pass filter of the horizontal cells, use it to cut low temporal frequencies (local luminance variations), unit is frames, typical value is 1 frame, as the photoreceptors
        @param HcellsSpatialConstant the spatial constant of the first order low pass filter of the horizontal cells, use it to cut low spatial frequencies (local luminance), unit is pixels, typical value is 5 pixel, this value is also used for local contrast computing when computing the local contrast adaptation at the ganglion cells level (Inner Plexiform Layer parvocellular channel model)
        @param ganglionCellsSensitivity the compression strengh of the ganglion cells local adaptation output, set a value between 0.6 and 1 for best results, a high value increases more the low value sensitivity... and the output saturates faster, recommended value: 0.7
        """

    def write(self, fs) -> None:
        """
        @brief Write xml/yml formated parameters information
        @param fs the filename of the xml file that will be open and writen with formatted parameters information
        """

    @overload
    def create(self, inputSize) -> retval:
        """
        @overload
        """

    @overload
    def create(self, inputSize, colorMode, colorSamplingMethod=..., useRetinaLogSampling=..., reductionFactor=..., samplingStrength=...) -> retval:
        """
        @brief Constructors from standardized interfaces : retreive a smart pointer to a Retina instance

        @param inputSize the input frame size
        @param colorMode the chosen processing mode : with or without color processing
        @param colorSamplingMethod specifies which kind of color sampling will be used : -   cv::bioinspired::RETINA_COLOR_RANDOM: each pixel position is either R, G or B in a random choice -   cv::bioinspired::RETINA_COLOR_DIAGONAL: color sampling is RGBRGBRGB..., line 2 BRGBRGBRG..., line 3, GBRGBRGBR... -   cv::bioinspired::RETINA_COLOR_BAYER: standard bayer sampling
        @param useRetinaLogSampling activate retina log sampling, if true, the 2 following parameters can be used
        @param reductionFactor only usefull if param useRetinaLogSampling=true, specifies the reduction factor of the output frame (as the center (fovea) is high resolution and corners can be underscaled, then a reduction of the output is allowed without precision leak
        @param samplingStrength only usefull if param useRetinaLogSampling=true, specifies the strength of the log scale that is applied
        """

class RetinaFastToneMapping(cv2.Algorithm):
    def applyFastToneMapping(self, inputImage, outputToneMappedImage=...) -> outputToneMappedImage:
        """
        @brief applies a luminance correction (initially High Dynamic Range (HDR) tone mapping)

        using only the 2 local adaptation stages of the retina parvocellular channel : photoreceptors
        level and ganlion cells level. Spatio temporal filtering is applied but limited to temporal
        smoothing and eventually high frequencies attenuation. This is a lighter method than the one
        available using the regular retina::run method. It is then faster but it does not include
        complete temporal filtering nor retina spectral whitening. Then, it can have a more limited
        effect on images with a very high dynamic range. This is an adptation of the original still
        image HDR tone mapping algorithm of David Alleyson, Sabine Susstruck and Laurence Meylan's
        work, please cite: -> Meylan L., Alleysson D., and Susstrunk S., A Model of Retinal Local
        Adaptation for the Tone Mapping of Color Filter Array Images, Journal of Optical Society of
        America, A, Vol. 24, N 9, September, 1st, 2007, pp. 2807-2816

        @param inputImage the input image to process RGB or gray levels
        @param outputToneMappedImage the output tone mapped image
        """

    def setup(self, photoreceptorsNeighborhoodRadius=..., ganglioncellsNeighborhoodRadius=..., meanLuminanceModulatorK=...) -> None:
        """
        @brief updates tone mapping behaviors by adjusing the local luminance computation area

        @param photoreceptorsNeighborhoodRadius the first stage local adaptation area
        @param ganglioncellsNeighborhoodRadius the second stage local adaptation area
        @param meanLuminanceModulatorK the factor applied to modulate the meanLuminance information (default is 1, see reference paper)
        """

    def create(self, inputSize) -> retval:
        """"""

class TransientAreasSegmentationModule(cv2.Algorithm):
    def clearAllBuffers(self) -> None:
        """
        @brief cleans all the buffers of the instance
        """

    def getSegmentationPicture(self, transientAreas=...) -> transientAreas:
        """
        @brief access function
        return the last segmentation result: a boolean picture which is resampled between 0 and 255 for a display purpose
        """

    def getSize(self) -> retval:
        """
        @brief return the sze of the manage input and output images
        """

    def printSetup(self) -> retval:
        """
        @brief parameters setup display method
        @return a string which contains formatted parameters information
        """

    def run(self, inputToSegment, channelIndex=...) -> None:
        """
        @brief main processing method, get result using methods getSegmentationPicture()
        @param inputToSegment : the image to process, it must match the instance buffer size !
        @param channelIndex : the channel to process in case of multichannel images
        """

    def setup(self, segmentationParameterFile=..., applyDefaultSetupOnFailure=...) -> None:
        """
        @brief try to open an XML segmentation parameters file to adjust current segmentation instance setup

        - if the xml file does not exist, then default setup is applied
        - warning, Exceptions are thrown if read XML file is not valid
        @param segmentationParameterFile : the parameters filename
        @param applyDefaultSetupOnFailure : set to true if an error must be thrown on error
        """

    def write(self, fs) -> None:
        """
        @brief write xml/yml formated parameters information
        @param fs : the filename of the xml file that will be open and writen with formatted parameters information
        """

    def create(self, inputSize) -> retval:
        """
        @brief allocator
        @param inputSize : size of the images input to segment (output will be the same size)
        """

def RetinaFastToneMapping_create(inputSize) -> retval:
    """
    .
    """

@overload
def Retina_create(inputSize) -> retval:
    """
    @overload
    """

@overload
def Retina_create(inputSize) -> retval:
    """
    @brief Constructors from standardized interfaces : retreive a smart pointer to a Retina instance

        @param inputSize the input frame size
        @param colorMode the chosen processing mode : with or without color processing
        @param colorSamplingMethod specifies which kind of color sampling will be used :
        -   cv::bioinspired::RETINA_COLOR_RANDOM: each pixel position is either R, G or B in a random choice
        -   cv::bioinspired::RETINA_COLOR_DIAGONAL: color sampling is RGBRGBRGB..., line 2 BRGBRGBRG..., line 3, GBRGBRGBR...
        -   cv::bioinspired::RETINA_COLOR_BAYER: standard bayer sampling
        @param useRetinaLogSampling activate retina log sampling, if true, the 2 following parameters can
    """

@overload
def Retina_create(inputSize) -> retval:
    """
    @param reductionFactor only usefull if param useRetinaLogSampling=true, specifies the reduction
    """

@overload
def Retina_create(inputSize) -> retval:
    """ """

@overload
def Retina_create(inputSize) -> retval:
    """
    @param samplingStrength only usefull if param useRetinaLogSampling=true, specifies the strength of
    """

@overload
def Retina_create(inputSize) -> retval:
    """ """

def TransientAreasSegmentationModule_create(inputSize) -> retval:
    """
    @brief allocator
        @param inputSize : size of the images input to segment (output will be the same size)
    """

RETINA_COLOR_BAYER: Final[int]
RETINA_COLOR_DIAGONAL: Final[int]
RETINA_COLOR_RANDOM: Final[int]
