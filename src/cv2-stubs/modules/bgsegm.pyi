import builtins
from typing import Any, Final, TypeAlias

from .. import functions as cv2

gtMask: TypeAlias = Any
fgmask: TypeAlias = Any
backgroundImage: TypeAlias = Any
frame: TypeAlias = Any
retval: TypeAlias = Any

class BackgroundSubtractorCNT(cv2.BackgroundSubtractor):
    def apply(self, image, fgmask=..., learningRate=...) -> fgmask:
        """"""

    def getBackgroundImage(self, backgroundImage=...) -> backgroundImage:
        """"""

    def getIsParallel(self) -> retval:
        """
        @brief Returns if we're parallelizing the algorithm.
        """

    def getMaxPixelStability(self) -> retval:
        """
        @brief Returns maximum allowed credit for a pixel in history.
        """

    def getMinPixelStability(self) -> retval:
        """
        @brief Returns number of frames with same pixel color to consider stable.
        """

    def getUseHistory(self) -> retval:
        """
        @brief Returns if we're giving a pixel credit for being stable for a long time.
        """

    def setIsParallel(self, value) -> None:
        """
        @brief Sets if we're parallelizing the algorithm.
        """

    def setMaxPixelStability(self, value) -> None:
        """
        @brief Sets the maximum allowed credit for a pixel in history.
        """

    def setMinPixelStability(self, value) -> None:
        """
        @brief Sets the number of frames with same pixel color to consider stable.
        """

    def setUseHistory(self, value) -> None:
        """
        @brief Sets if we're giving a pixel credit for being stable for a long time.
        """

class BackgroundSubtractorGMG(cv2.BackgroundSubtractor):
    def getBackgroundPrior(self) -> retval:
        """
        @brief Returns the prior probability that each individual pixel is a background pixel.
        """

    def getDecisionThreshold(self) -> retval:
        """
        @brief Returns the value of decision threshold.

        Decision value is the value above which pixel is determined to be FG.
        """

    def getDefaultLearningRate(self) -> retval:
        """
        @brief Returns the learning rate of the algorithm.

        It lies between 0.0 and 1.0. It determines how quickly features are "forgotten" from
        histograms.
        """

    def getMaxFeatures(self) -> retval:
        """
        @brief Returns total number of distinct colors to maintain in histogram.
        """

    def getMaxVal(self) -> retval:
        """
        @brief Returns the maximum value taken on by pixels in image sequence. e.g. 1.0 or 255.
        """

    def getMinVal(self) -> retval:
        """
        @brief Returns the minimum value taken on by pixels in image sequence. Usually 0.
        """

    def getNumFrames(self) -> retval:
        """
        @brief Returns the number of frames used to initialize background model.
        """

    def getQuantizationLevels(self) -> retval:
        """
        @brief Returns the parameter used for quantization of color-space.

        It is the number of discrete levels in each channel to be used in histograms.
        """

    def getSmoothingRadius(self) -> retval:
        """
        @brief Returns the kernel radius used for morphological operations
        """

    def getUpdateBackgroundModel(self) -> retval:
        """
        @brief Returns the status of background model update
        """

    def setBackgroundPrior(self, bgprior) -> None:
        """
        @brief Sets the prior probability that each individual pixel is a background pixel.
        """

    def setDecisionThreshold(self, thresh) -> None:
        """
        @brief Sets the value of decision threshold.
        """

    def setDefaultLearningRate(self, lr) -> None:
        """
        @brief Sets the learning rate of the algorithm.
        """

    def setMaxFeatures(self, maxFeatures) -> None:
        """
        @brief Sets total number of distinct colors to maintain in histogram.
        """

    def setMaxVal(self, val) -> None:
        """
        @brief Sets the maximum value taken on by pixels in image sequence.
        """

    def setMinVal(self, val) -> None:
        """
        @brief Sets the minimum value taken on by pixels in image sequence.
        """

    def setNumFrames(self, nframes) -> None:
        """
        @brief Sets the number of frames used to initialize background model.
        """

    def setQuantizationLevels(self, nlevels) -> None:
        """
        @brief Sets the parameter used for quantization of color-space
        """

    def setSmoothingRadius(self, radius) -> None:
        """
        @brief Sets the kernel radius used for morphological operations
        """

    def setUpdateBackgroundModel(self, update) -> None:
        """
        @brief Sets the status of background model update
        """

class BackgroundSubtractorGSOC(cv2.BackgroundSubtractor):
    def apply(self, image, fgmask=..., learningRate=...) -> fgmask:
        """"""

    def getBackgroundImage(self, backgroundImage=...) -> backgroundImage:
        """"""

class BackgroundSubtractorLSBP(cv2.BackgroundSubtractor):
    def apply(self, image, fgmask=..., learningRate=...) -> fgmask:
        """"""

    def getBackgroundImage(self, backgroundImage=...) -> backgroundImage:
        """"""

class BackgroundSubtractorLSBPDesc(builtins.object): ...

class BackgroundSubtractorMOG(cv2.BackgroundSubtractor):
    def getBackgroundRatio(self) -> retval:
        """"""

    def getHistory(self) -> retval:
        """"""

    def getNMixtures(self) -> retval:
        """"""

    def getNoiseSigma(self) -> retval:
        """"""

    def setBackgroundRatio(self, backgroundRatio) -> None:
        """"""

    def setHistory(self, nframes) -> None:
        """"""

    def setNMixtures(self, nmix) -> None:
        """"""

    def setNoiseSigma(self, noiseSigma) -> None:
        """"""

class SyntheticSequenceGenerator(cv2.Algorithm):
    def getNextFrame(self, frame=..., gtMask=...) -> tuple[frame, gtMask]:
        """
        @brief Obtain the next frame in the sequence.

        @param frame Output frame.
        @param gtMask Output ground-truth (reference) segmentation mask object/background.
        """

def createBackgroundSubtractorCNT(minPixelStability=..., useHistory=..., maxPixelStability=..., isParallel=...) -> retval:
    """
    @brief Creates a CNT Background Subtractor

    @param minPixelStability number of frames with same pixel color to consider stable
    @param useHistory determines if we're giving a pixel credit for being stable for a long time
    @param maxPixelStability maximum allowed credit for a pixel in history
    @param isParallel determines if we're parallelizing the algorithm
    """

def createBackgroundSubtractorGMG(initializationFrames=..., decisionThreshold=...) -> retval:
    """
    @brief Creates a GMG Background Subtractor

    @param initializationFrames number of frames used to initialize the background models.
    @param decisionThreshold Threshold value, above which it is marked foreground, else background.
    """

def createBackgroundSubtractorGSOC(mc=..., nSamples=..., replaceRate=..., propagationRate=..., hitsThreshold=..., alpha=..., beta=..., blinkingSupressionDecay=..., blinkingSupressionMultiplier=..., noiseRemovalThresholdFacBG=..., noiseRemovalThresholdFacFG=...) -> retval:
    """
    @brief Creates an instance of BackgroundSubtractorGSOC algorithm.

    Implementation of the different yet better algorithm which is called GSOC, as it was implemented during GSOC and was not originated from any paper.

    @param mc Whether to use camera motion compensation.
    @param nSamples Number of samples to maintain at each point of the frame.
    @param replaceRate Probability of replacing the old sample - how fast the model will update itself.
    @param propagationRate Probability of propagating to neighbors.
    @param hitsThreshold How many positives the sample must get before it will be considered as a possible replacement.
    @param alpha Scale coefficient for threshold.
    @param beta Bias coefficient for threshold.
    @param blinkingSupressionDecay Blinking supression decay factor.
    @param blinkingSupressionMultiplier Blinking supression multiplier.
    @param noiseRemovalThresholdFacBG Strength of the noise removal for background points.
    @param noiseRemovalThresholdFacFG Strength of the noise removal for foreground points.
    """

def createBackgroundSubtractorLSBP(mc=..., nSamples=..., LSBPRadius=..., Tlower=..., Tupper=..., Tinc=..., Tdec=..., Rscale=..., Rincdec=..., noiseRemovalThresholdFacBG=..., noiseRemovalThresholdFacFG=..., LSBPthreshold=..., minCount=...) -> retval:
    """
    @brief Creates an instance of BackgroundSubtractorLSBP algorithm.

    Background Subtraction using Local SVD Binary Pattern. More details about the algorithm can be found at @cite LGuo2016

    @param mc Whether to use camera motion compensation.
    @param nSamples Number of samples to maintain at each point of the frame.
    @param LSBPRadius LSBP descriptor radius.
    @param Tlower Lower bound for T-values. See @cite LGuo2016 for details.
    @param Tupper Upper bound for T-values. See @cite LGuo2016 for details.
    @param Tinc Increase step for T-values. See @cite LGuo2016 for details.
    @param Tdec Decrease step for T-values. See @cite LGuo2016 for details.
    @param Rscale Scale coefficient for threshold values.
    @param Rincdec Increase/Decrease step for threshold values.
    @param noiseRemovalThresholdFacBG Strength of the noise removal for background points.
    @param noiseRemovalThresholdFacFG Strength of the noise removal for foreground points.
    @param LSBPthreshold Threshold for LSBP binary string.
    @param minCount Minimal number of matches for sample to be considered as foreground.
    """

def createBackgroundSubtractorMOG(history=..., nmixtures=..., backgroundRatio=..., noiseSigma=...) -> retval:
    """
    @brief Creates mixture-of-gaussian background subtractor

    @param history Length of the history.
    @param nmixtures Number of Gaussian mixtures.
    @param backgroundRatio Background ratio.
    @param noiseSigma Noise strength (standard deviation of the brightness or each color channel). 0
    means some automatic value.
    """

def createSyntheticSequenceGenerator(background, object, amplitude=..., wavelength=..., wavespeed=..., objspeed=...) -> retval:
    """
    @brief Creates an instance of SyntheticSequenceGenerator.

    @param background Background image for object.
    @param object Object image which will move slowly over the background.
    @param amplitude Amplitude of wave distortion applied to background.
    @param wavelength Length of waves in distortion applied to background.
    @param wavespeed How fast waves will move.
    @param objspeed How fast object will fly over background.
    """

LSBP_CAMERA_MOTION_COMPENSATION_LK: Final[int]
LSBP_CAMERA_MOTION_COMPENSATION_NONE: Final[int]
