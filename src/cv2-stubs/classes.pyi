
class AKAZE(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getDescriptorChannels(self) -> retval:
        """"""

    def getDescriptorSize(self) -> retval:
        """"""

    def getDescriptorType(self) -> retval:
        """"""

    def getDiffusivity(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setDescriptorChannels(self, dch) -> None:
        """"""

    def setDescriptorSize(self, dsize) -> None:
        """"""

    def setDescriptorType(self, dtype) -> None:
        """"""

    def setDiffusivity(self, diff) -> None:
        """"""

    def setNOctaveLayers(self, octaveLayers) -> None:
        """"""

    def setNOctaves(self, octaves) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def create(self, descriptor_type = ..., descriptor_size = ..., descriptor_channels = ..., threshold = ..., nOctaves = ..., nOctaveLayers = ..., diffusivity = ...) -> retval:
        """
        @brief The AKAZE constructor

        @param descriptor_type Type of the extracted descriptor: DESCRIPTOR_KAZE, DESCRIPTOR_KAZE_UPRIGHT, DESCRIPTOR_MLDB or DESCRIPTOR_MLDB_UPRIGHT.
        @param descriptor_size Size of the descriptor in bits. 0 -\> Full size
        @param descriptor_channels Number of channels in the descriptor (1, 2, 3)
        @param threshold Detector response threshold to accept point
        @param nOctaves Maximum octave evolution of the image
        @param nOctaveLayers Default number of sublevels per scale level
        @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        """


class AffineFeature(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getViewParams(self, tilts, rolls) -> None:
        """"""

    def setViewParams(self, tilts, rolls) -> None:
        """"""

    def create(self, backend, maxTilt = ..., minTilt = ..., tiltStep = ..., rotateStepBase = ...) -> retval:
        """
        @param backend The detector/extractor you want to use as backend.
        @param maxTilt The highest power index of tilt factor. 5 is used in the paper as tilt sampling range n.
        @param minTilt The lowest power index of tilt factor. 0 is used in the paper.
        @param tiltStep Tilt sampling step \f$\delta_t\f$ in Algorithm 1 in the paper.
        @param rotateStepBase Rotation sampling step factor b in Algorithm 1 in the paper.
        """


class AgastFeatureDetector(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getNonmaxSuppression(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getType(self) -> retval:
        """"""

    def setNonmaxSuppression(self, f) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setType(self, type) -> None:
        """"""

    def create(self, threshold = ..., nonmaxSuppression = ..., type = ...) -> retval:
        """"""


class Algorithm(builtins.object):
    def clear(self) -> None:
        """
        @brief Clears the algorithm state
        """

    def empty(self) -> retval:
        """
        @brief Returns true if the Algorithm is empty (e.g. in the very beginning or after unsuccessful read
        """

    def getDefaultName(self) -> retval:
        """
        Returns the algorithm string identifier.
        This string is used as top level xml/yml node tag when the object is saved to a file or string.
        """

    def read(self, fn) -> None:
        """
        @brief Reads algorithm parameters from a file storage
        """

    def save(self, filename) -> None:
        """
        Saves the algorithm to a file.
        In order to make this method work, the derived class must implement Algorithm::write(FileStorage& fs).
        """

    def write(self, fs) -> None:
        """
        @brief Stores algorithm parameters in a file storage
        """

    @overload
    def write(self, fs, name) -> None:
        """
        """


class AlignExposures(Algorithm):
    def process(self, src, dst, times, response) -> None:
        """
        @brief Aligns images

        @param src vector of input images
        @param dst vector of aligned images
        @param times vector of exposure time values for each image
        @param response 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.
        """


class AlignMTB(AlignExposures):
    def calculateShift(self, img0, img1) -> retval:
        """
        @brief Calculates shift between two images, i. e. how to shift the second image to correspond it with the
        first.

        @param img0 first image
        @param img1 second image
        """

    def computeBitmaps(self, img, tb = ..., eb = ...) -> tuple[tb, eb]:
        """
        @brief Computes median threshold and exclude bitmaps of given image.

        @param img input image
        @param tb median threshold bitmap
        @param eb exclude bitmap
        """

    def getCut(self) -> retval:
        """"""

    def getExcludeRange(self) -> retval:
        """"""

    def getMaxBits(self) -> retval:
        """"""

    def process(self, src, dst, times, response) -> None:
        """"""

    def process(self, src, dst) -> None:
        """
        @brief Short version of process, that doesn't take extra arguments.

        @param src vector of input images
        @param dst vector of aligned images
        """

    def setCut(self, value) -> None:
        """"""

    def setExcludeRange(self, exclude_range) -> None:
        """"""

    def setMaxBits(self, max_bits) -> None:
        """"""

    def shiftMat(self, src, shift, dst = ...) -> dst:
        """
        @brief Helper function, that shift Mat filling new regions with zeros.

        @param src input image
        @param dst result image
        @param shift shift value
        """


class AsyncArray(builtins.object):
    def get(self, dst = ...) -> dst:
        """
        Fetch the result.
        @param[out] dst destination array  Waits for result until container has valid result. Throws exception if exception was stored as a result.  Throws exception on invalid container state.  @note Result or stored exception can be fetched only once.
        """

    def get(self, timeoutNs, dst = ...) -> tuple[retval, dst]:
        """
        Retrieving the result with timeout
        @param[out] dst destination array
        @param[in] timeoutNs timeout in nanoseconds, -1 for infinite wait  @returns true if result is ready, false if the timeout has expired  @note Result or stored exception can be fetched only once.
        """

    def release(self) -> None:
        """"""

    def valid(self) -> retval:
        """"""

    def wait_for(self, timeoutNs) -> retval:
        """"""


class BFMatcher(DescriptorMatcher):
    def create(self, normType = ..., crossCheck = ...) -> retval:
        """
        @brief Brute-force matcher create method.
        @param normType One of NORM_L1, NORM_L2, NORM_HAMMING, NORM_HAMMING2. L1 and L2 norms are preferable choices for SIFT and SURF descriptors, NORM_HAMMING should be used with ORB, BRISK and BRIEF, NORM_HAMMING2 should be used with ORB when WTA_K==3 or 4 (see ORB::ORB constructor description).
        @param crossCheck If it is false, this is will be default BFMatcher behaviour when it finds the k nearest neighbors for each query descriptor. If crossCheck==true, then the knnMatch() method with k=1 will only return pairs (i,j) such that for i-th query descriptor the j-th descriptor in the matcher's collection is the nearest and vice versa, i.e. the BFMatcher will only return consistent pairs. Such technique usually produces best results with minimal number of outliers when there are enough matches. This is alternative to the ratio test, used by D. Lowe in SIFT paper.
        """


class BOWImgDescriptorExtractor(builtins.object):
    @overload
    def compute(self, image, keypoints, imgDescriptor = ...) -> imgDescriptor:
        """
        @overload
        @param keypointDescriptors Computed descriptors to match with vocabulary.
        @param imgDescriptor Computed output image descriptor.
        @param pointIdxsOfClusters Indices of keypoints that belong to the cluster. This means that pointIdxsOfClusters[i] are keypoint indices that belong to the i -th cluster (word of vocabulary) returned if it is non-zero.
        """

    def descriptorSize(self) -> retval:
        """
        @brief Returns an image descriptor size if the vocabulary is set. Otherwise, it returns 0.
        """

    def descriptorType(self) -> retval:
        """
        @brief Returns an image descriptor type.
        """

    def getVocabulary(self) -> retval:
        """
        @brief Returns the set vocabulary.
        """

    def setVocabulary(self, vocabulary) -> None:
        """
        @brief Sets a visual vocabulary.

        @param vocabulary Vocabulary (can be trained using the inheritor of BOWTrainer ). Each row of the vocabulary is a visual word (cluster center).
        """


class BOWKMeansTrainer(BOWTrainer):
    def cluster(self) -> retval:
        """"""

    def cluster(self, descriptors) -> retval:
        """"""


class BOWTrainer(builtins.object):
    def add(self, descriptors) -> None:
        """
        @brief Adds descriptors to a training set.

        @param descriptors Descriptors to add to a training set. Each row of the descriptors matrix is a descriptor.  The training set is clustered using clustermethod to construct the vocabulary.
        """

    def clear(self) -> None:
        """"""

    @overload
    def cluster(self) -> retval:
        """
        @overload
        """

    def cluster(self, descriptors) -> retval:
        """
        @brief Clusters train descriptors.

        @param descriptors Descriptors to cluster. Each row of the descriptors matrix is a descriptor. Descriptors are not added to the inner train descriptor set.  The vocabulary consists of cluster centers. So, this method returns the vocabulary. In the first variant of the method, train descriptors stored in the object are clustered. In the second variant, input descriptors are clustered.
        """

    def descriptorsCount(self) -> retval:
        """
        @brief Returns the count of all descriptors stored in the training set.
        """

    def getDescriptors(self) -> retval:
        """
        @brief Returns a training set of descriptors.
        """


class BRISK(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getOctaves(self) -> retval:
        """"""

    def getPatternScale(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setOctaves(self, octaves) -> None:
        """
        @brief Set detection octaves.
        @param octaves detection octaves. Use 0 to do single scale.
        """

    def setPatternScale(self, patternScale) -> None:
        """
        @brief Set detection patternScale.
        @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        """

    def setThreshold(self, threshold) -> None:
        """
        @brief Set detection threshold.
        @param threshold AGAST detection threshold score.
        """

    def create(self, thresh = ..., octaves = ..., patternScale = ...) -> retval:
        """
        @brief The BRISK constructor

        @param thresh AGAST detection threshold score.
        @param octaves detection octaves. Use 0 to do single scale.
        @param patternScale apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        """

    def create(self, radiusList, numberList, dMax = ..., dMin = ..., indexChange = ...) -> retval:
        """
        @brief The BRISK constructor for a custom pattern

        @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).
        @param numberList defines the number of sampling points on the sampling circle. Must be the same size as radiusList..
        @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).
        @param dMin threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).
        @param indexChange index remapping of the bits.
        """

    def create(self, thresh, octaves, radiusList, numberList, dMax = ..., dMin = ..., indexChange = ...) -> retval:
        """
        @brief The BRISK constructor for a custom pattern, detection threshold and octaves

        @param thresh AGAST detection threshold score.
        @param octaves detection octaves. Use 0 to do single scale.
        @param radiusList defines the radii (in pixels) where the samples around a keypoint are taken (for keypoint scale 1).
        @param numberList defines the number of sampling points on the sampling circle. Must be the same size as radiusList..
        @param dMax threshold for the short pairings used for descriptor formation (in pixels for keypoint scale 1).
        @param dMin threshold for the long pairings used for orientation determination (in pixels for keypoint scale 1).
        @param indexChange index remapping of the bits.
        """


class BackgroundSubtractor(Algorithm):
    def apply(self, image, fgmask = ..., learningRate = ...) -> fgmask:
        """
        @brief Computes a foreground mask.

        @param image Next video frame.
        @param fgmask The output foreground mask as an 8-bit binary image.
        @param learningRate The value between 0 and 1 that indicates how fast the background model is learnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
        """

    def getBackgroundImage(self, backgroundImage = ...) -> backgroundImage:
        """
        @brief Computes a background image.

        @param backgroundImage The output background image.  @note Sometimes the background image can be very blurry, as it contain the average background statistics.
        """


class BackgroundSubtractorKNN(BackgroundSubtractor):
    def getDetectShadows(self) -> retval:
        """
        @brief Returns the shadow detection flag

        If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorKNN for
        details.
        """

    def getDist2Threshold(self) -> retval:
        """
        @brief Returns the threshold on the squared distance between the pixel and the sample

        The threshold on the squared distance between the pixel and the sample to decide whether a pixel is
        close to a data sample.
        """

    def getHistory(self) -> retval:
        """
        @brief Returns the number of last frames that affect the background model
        """

    def getNSamples(self) -> retval:
        """
        @brief Returns the number of data samples in the background model
        """

    def getShadowThreshold(self) -> retval:
        """
        @brief Returns the shadow threshold

        A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
        the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
        is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara,
        *Detecting Moving Shadows...*, IEEE PAMI,2003.
        """

    def getShadowValue(self) -> retval:
        """
        @brief Returns the shadow value

        Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
        in the mask always means background, 255 means foreground.
        """

    def getkNNSamples(self) -> retval:
        """
        @brief Returns the number of neighbours, the k in the kNN.

        K is the number of samples that need to be within dist2Threshold in order to decide that that
        pixel is matching the kNN background model.
        """

    def setDetectShadows(self, detectShadows) -> None:
        """
        @brief Enables or disables shadow detection
        """

    def setDist2Threshold(self, _dist2Threshold) -> None:
        """
        @brief Sets the threshold on the squared distance
        """

    def setHistory(self, history) -> None:
        """
        @brief Sets the number of last frames that affect the background model
        """

    def setNSamples(self, _nN) -> None:
        """
        @brief Sets the number of data samples in the background model.

        The model needs to be reinitalized to reserve memory.
        """

    def setShadowThreshold(self, threshold) -> None:
        """
        @brief Sets the shadow threshold
        """

    def setShadowValue(self, value) -> None:
        """
        @brief Sets the shadow value
        """

    def setkNNSamples(self, _nkNN) -> None:
        """
        @brief Sets the k in the kNN. How many nearest neighbours need to match.
        """


class BackgroundSubtractorMOG2(BackgroundSubtractor):
    def apply(self, image, fgmask = ..., learningRate = ...) -> fgmask:
        """
        @brief Computes a foreground mask.

        @param image Next video frame. Floating point frame will be used without scaling and should be in range \f$[0,255]\f$.
        @param fgmask The output foreground mask as an 8-bit binary image.
        @param learningRate The value between 0 and 1 that indicates how fast the background model is learnt. Negative parameter value makes the algorithm to use some automatically chosen learning rate. 0 means that the background model is not updated at all, 1 means that the background model is completely reinitialized from the last frame.
        """

    def getBackgroundRatio(self) -> retval:
        """
        @brief Returns the "background ratio" parameter of the algorithm

        If a foreground pixel keeps semi-constant value for about backgroundRatio\*history frames, it's
        considered background and added to the model as a center of a new component. It corresponds to TB
        parameter in the paper.
        """

    def getComplexityReductionThreshold(self) -> retval:
        """
        @brief Returns the complexity reduction threshold

        This parameter defines the number of samples needed to accept to prove the component exists. CT=0.05
        is a default value for all the samples. By setting CT=0 you get an algorithm very similar to the
        standard Stauffer&Grimson algorithm.
        """

    def getDetectShadows(self) -> retval:
        """
        @brief Returns the shadow detection flag

        If true, the algorithm detects shadows and marks them. See createBackgroundSubtractorMOG2 for
        details.
        """

    def getHistory(self) -> retval:
        """
        @brief Returns the number of last frames that affect the background model
        """

    def getNMixtures(self) -> retval:
        """
        @brief Returns the number of gaussian components in the background model
        """

    def getShadowThreshold(self) -> retval:
        """
        @brief Returns the shadow threshold

        A shadow is detected if pixel is a darker version of the background. The shadow threshold (Tau in
        the paper) is a threshold defining how much darker the shadow can be. Tau= 0.5 means that if a pixel
        is more than twice darker then it is not shadow. See Prati, Mikic, Trivedi and Cucchiara,
        *Detecting Moving Shadows...*, IEEE PAMI,2003.
        """

    def getShadowValue(self) -> retval:
        """
        @brief Returns the shadow value

        Shadow value is the value used to mark shadows in the foreground mask. Default value is 127. Value 0
        in the mask always means background, 255 means foreground.
        """

    def getVarInit(self) -> retval:
        """
        @brief Returns the initial variance of each gaussian component
        """

    def getVarMax(self) -> retval:
        """"""

    def getVarMin(self) -> retval:
        """"""

    def getVarThreshold(self) -> retval:
        """
        @brief Returns the variance threshold for the pixel-model match

        The main threshold on the squared Mahalanobis distance to decide if the sample is well described by
        the background model or not. Related to Cthr from the paper.
        """

    def getVarThresholdGen(self) -> retval:
        """
        @brief Returns the variance threshold for the pixel-model match used for new mixture component generation

        Threshold for the squared Mahalanobis distance that helps decide when a sample is close to the
        existing components (corresponds to Tg in the paper). If a pixel is not close to any component, it
        is considered foreground or added as a new component. 3 sigma =\> Tg=3\*3=9 is default. A smaller Tg
        value generates more components. A higher Tg value may result in a small number of components but
        they can grow too large.
        """

    def setBackgroundRatio(self, ratio) -> None:
        """
        @brief Sets the "background ratio" parameter of the algorithm
        """

    def setComplexityReductionThreshold(self, ct) -> None:
        """
        @brief Sets the complexity reduction threshold
        """

    def setDetectShadows(self, detectShadows) -> None:
        """
        @brief Enables or disables shadow detection
        """

    def setHistory(self, history) -> None:
        """
        @brief Sets the number of last frames that affect the background model
        """

    def setNMixtures(self, nmixtures) -> None:
        """
        @brief Sets the number of gaussian components in the background model.

        The model needs to be reinitalized to reserve memory.
        """

    def setShadowThreshold(self, threshold) -> None:
        """
        @brief Sets the shadow threshold
        """

    def setShadowValue(self, value) -> None:
        """
        @brief Sets the shadow value
        """

    def setVarInit(self, varInit) -> None:
        """
        @brief Sets the initial variance of each gaussian component
        """

    def setVarMax(self, varMax) -> None:
        """"""

    def setVarMin(self, varMin) -> None:
        """"""

    def setVarThreshold(self, varThreshold) -> None:
        """
        @brief Sets the variance threshold for the pixel-model match
        """

    def setVarThresholdGen(self, varThresholdGen) -> None:
        """
        @brief Sets the variance threshold for the pixel-model match used for new mixture component generation
        """


class BaseCascadeClassifier(Algorithm):
    ...


class CLAHE(Algorithm):
    def apply(self, src, dst = ...) -> dst:
        """
        @brief Equalizes the histogram of a grayscale image using Contrast Limited Adaptive Histogram Equalization.

        @param src Source image of type CV_8UC1 or CV_16UC1.
        @param dst Destination image.
        """

    def collectGarbage(self) -> None:
        """"""

    def getClipLimit(self) -> retval:
        """"""

    def getTilesGridSize(self) -> retval:
        """"""

    def setClipLimit(self, clipLimit) -> None:
        """
        @brief Sets threshold for contrast limiting.

        @param clipLimit threshold value.
        """

    def setTilesGridSize(self, tileGridSize) -> None:
        """
        @brief Sets size of grid for histogram equalization. Input image will be divided into
        equally sized rectangular tiles.

        @param tileGridSize defines the number of tiles in row and column.
        """


class CalibrateCRF(Algorithm):
    def process(self, src, times, dst = ...) -> dst:
        """
        @brief Recovers inverse camera response.

        @param src vector of input images
        @param dst 256x1 matrix with inverse camera response function
        @param times vector of exposure time values for each image
        """


class CalibrateDebevec(CalibrateCRF):
    def getLambda(self) -> retval:
        """"""

    def getRandom(self) -> retval:
        """"""

    def getSamples(self) -> retval:
        """"""

    def setLambda(self, lambda_) -> None:
        """"""

    def setRandom(self, random) -> None:
        """"""

    def setSamples(self, samples) -> None:
        """"""


class CalibrateRobertson(CalibrateCRF):
    def getMaxIter(self) -> retval:
        """"""

    def getRadiance(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def setMaxIter(self, max_iter) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""


class CascadeClassifier(builtins.object):
    def detectMultiScale(self, image, scaleFactor = ..., minNeighbors = ..., flags = ..., minSize = ..., maxSize = ...) -> objects:
        """
        @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
        of rectangles.

        @param image Matrix of the type CV_8U containing an image where objects are detected.
        @param objects Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
        @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        @param flags Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
        """

    @overload
    def detectMultiScale2(self, image, scaleFactor = ..., minNeighbors = ..., flags = ..., minSize = ..., maxSize = ...) -> tuple[objects, numDetections]:
        """
        @overload
        @param image Matrix of the type CV_8U containing an image where objects are detected.
        @param objects Vector of rectangles where each rectangle contains the detected object, the rectangles may be partially outside the original image.
        @param numDetections Vector of detection numbers for the corresponding objects. An object's number of detections is the number of neighboring positively classified rectangles that were joined together to form the object.
        @param scaleFactor Parameter specifying how much the image size is reduced at each image scale.
        @param minNeighbors Parameter specifying how many neighbors each candidate rectangle should have to retain it.
        @param flags Parameter with the same meaning for an old cascade as in the function cvHaarDetectObjects. It is not used for a new cascade.
        @param minSize Minimum possible object size. Objects smaller than that are ignored.
        @param maxSize Maximum possible object size. Objects larger than that are ignored. If `maxSize == minSize` model is evaluated on single scale.
        """

    @overload
    def detectMultiScale3(self, image, scaleFactor = ..., minNeighbors = ..., flags = ..., minSize = ..., maxSize = ..., outputRejectLevels = ...) -> tuple[objects, rejectLevels, levelWeights]:
        """
        @overload
        This function allows you to retrieve the final stage decision certainty of classification.
        For this, one needs to set `outputRejectLevels` on true and provide the `rejectLevels` and `levelWeights` parameter.
        For each resulting detection, `levelWeights` will then contain the certainty of classification at the final stage.
        This value can then be used to separate strong from weaker classifications.

        A code sample on how to use it efficiently can be found below:
        @code
        Mat img;
        vector<double> weights;
        vector<int> levels;
        vector<Rect> detections;
        CascadeClassifier model("/path/to/your/model.xml");
        model.detectMultiScale(img, detections, levels, weights, 1.1, 3, 0, Size(), Size(), true);
        cerr << "Detection " << detections[0] << " with weight " << weights[0] << endl;
        @endcode
        """

    def empty(self) -> retval:
        """
        @brief Checks whether the classifier has been loaded.
        """

    def getFeatureType(self) -> retval:
        """"""

    def getOriginalWindowSize(self) -> retval:
        """"""

    def isOldFormatCascade(self) -> retval:
        """"""

    def load(self, filename) -> retval:
        """
        @brief Loads a classifier from a file.

        @param filename Name of the file from which the classifier is loaded. The file may contain an old HAAR classifier trained by the haartraining application or a new cascade classifier trained by the traincascade application.
        """

    def read(self, node) -> retval:
        """
        @brief Reads a classifier from a FileStorage node.

        @note The file may contain a new cascade classifier (trained by the traincascade application) only.
        """

    def convert(self, oldcascade, newcascade) -> retval:
        """"""


class CirclesGridFinderParameters(builtins.object):
    ...


class DISOpticalFlow(DenseOpticalFlow):
    def getFinestScale(self) -> retval:
        """
        @brief Finest level of the Gaussian pyramid on which the flow is computed (zero level
        corresponds to the original image resolution). The final flow is obtained by bilinear upscaling.
        @see setFinestScale
        """

    def getGradientDescentIterations(self) -> retval:
        """
        @brief Maximum number of gradient descent iterations in the patch inverse search stage. Higher values
        may improve quality in some cases.
        @see setGradientDescentIterations
        """

    def getPatchSize(self) -> retval:
        """
        @brief Size of an image patch for matching (in pixels). Normally, default 8x8 patches work well
        enough in most cases.
        @see setPatchSize
        """

    def getPatchStride(self) -> retval:
        """
        @brief Stride between neighbor patches. Must be less than patch size. Lower values correspond
        to higher flow quality.
        @see setPatchStride
        """

    def getUseMeanNormalization(self) -> retval:
        """
        @brief Whether to use mean-normalization of patches when computing patch distance. It is turned on
        by default as it typically provides a noticeable quality boost because of increased robustness to
        illumination variations. Turn it off if you are certain that your sequence doesn't contain any changes
        in illumination.
        @see setUseMeanNormalization
        """

    def getUseSpatialPropagation(self) -> retval:
        """
        @brief Whether to use spatial propagation of good optical flow vectors. This option is turned on by
        default, as it tends to work better on average and can sometimes help recover from major errors
        introduced by the coarse-to-fine scheme employed by the DIS optical flow algorithm. Turning this
        option off can make the output flow field a bit smoother, however.
        @see setUseSpatialPropagation
        """

    def getVariationalRefinementAlpha(self) -> retval:
        """
        @brief Weight of the smoothness term
        @see setVariationalRefinementAlpha
        """

    def getVariationalRefinementDelta(self) -> retval:
        """
        @brief Weight of the color constancy term
        @see setVariationalRefinementDelta
        """

    def getVariationalRefinementGamma(self) -> retval:
        """
        @brief Weight of the gradient constancy term
        @see setVariationalRefinementGamma
        """

    def getVariationalRefinementIterations(self) -> retval:
        """
        @brief Number of fixed point iterations of variational refinement per scale. Set to zero to
        disable variational refinement completely. Higher values will typically result in more smooth and
        high-quality flow.
        @see setGradientDescentIterations
        """

    def setFinestScale(self, val) -> None:
        """
        @copybrief getFinestScale @see getFinestScale
        """

    def setGradientDescentIterations(self, val) -> None:
        """
        @copybrief getGradientDescentIterations @see getGradientDescentIterations
        """

    def setPatchSize(self, val) -> None:
        """
        @copybrief getPatchSize @see getPatchSize
        """

    def setPatchStride(self, val) -> None:
        """
        @copybrief getPatchStride @see getPatchStride
        """

    def setUseMeanNormalization(self, val) -> None:
        """
        @copybrief getUseMeanNormalization @see getUseMeanNormalization
        """

    def setUseSpatialPropagation(self, val) -> None:
        """
        @copybrief getUseSpatialPropagation @see getUseSpatialPropagation
        """

    def setVariationalRefinementAlpha(self, val) -> None:
        """
        @copybrief getVariationalRefinementAlpha @see getVariationalRefinementAlpha
        """

    def setVariationalRefinementDelta(self, val) -> None:
        """
        @copybrief getVariationalRefinementDelta @see getVariationalRefinementDelta
        """

    def setVariationalRefinementGamma(self, val) -> None:
        """
        @copybrief getVariationalRefinementGamma @see getVariationalRefinementGamma
        """

    def setVariationalRefinementIterations(self, val) -> None:
        """
        @copybrief getGradientDescentIterations @see getGradientDescentIterations
        """

    def create(self, preset = ...) -> retval:
        """
        @brief Creates an instance of DISOpticalFlow

        @param preset one of PRESET_ULTRAFAST, PRESET_FAST and PRESET_MEDIUM
        """


class DMatch(builtins.object):
    ...


class DenseOpticalFlow(Algorithm):
    def calc(self, I0, I1, flow) -> flow:
        """
        @brief Calculates an optical flow.

        @param I0 first 8-bit single-channel input image.
        @param I1 second input image of the same size and the same type as prev.
        @param flow computed flow image that has the same size as prev and type CV_32FC2.
        """

    def collectGarbage(self) -> None:
        """
        @brief Releases all inner buffers.
        """


class DescriptorMatcher(Algorithm):
    def add(self, descriptors) -> None:
        """
        @brief Adds descriptors to train a CPU(trainDescCollectionis) or GPU(utrainDescCollectionis) descriptor
        collection.

        If the collection is not empty, the new descriptors are added to existing train descriptors.

        @param descriptors Descriptors to add. Each descriptors[i] is a set of descriptors from the same train image.
        """

    def clear(self) -> None:
        """
        @brief Clears the train descriptor collections.
        """

    def clone(self, emptyTrainData = ...) -> retval:
        """
        @brief Clones the matcher.

        @param emptyTrainData If emptyTrainData is false, the method creates a deep copy of the object, that is, copies both parameters and train data. If emptyTrainData is true, the method creates an object copy with the current parameters but with empty train data.
        """

    def empty(self) -> retval:
        """
        @brief Returns true if there are no train descriptors in the both collections.
        """

    def getTrainDescriptors(self) -> retval:
        """
        @brief Returns a constant link to the train descriptor collection trainDescCollection .
        """

    def isMaskSupported(self) -> retval:
        """
        @brief Returns true if the descriptor matcher supports masking permissible matches.
        """

    def knnMatch(self, queryDescriptors, trainDescriptors, k, mask = ..., compactResult = ...) -> matches:
        """
        @brief Finds the k best matches for each descriptor from a query set.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.
        @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
        @param k Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.  These extended variants of DescriptorMatcher::match methods find several best matches for each query descriptor. The matches are returned in the distance increasing order. See DescriptorMatcher::match for the details about query and train descriptors.
        """

    @overload
    def knnMatch(self, queryDescriptors, k, masks = ..., compactResult = ...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Matches. Each matches[i] is k or less matches for the same query descriptor.
        @param k Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total.
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        """

    def match(self, queryDescriptors, trainDescriptors, mask = ...) -> matches:
        """
        @brief Finds the best match for each descriptor from a query set.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param matches Matches. If a query descriptor is masked out in mask , no match is added for this descriptor. So, matches size may be smaller than the query descriptors count.
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.  In the first variant of this method, the train descriptors are passed as an input argument. In the second variant of the method, train descriptors collection that was set by DescriptorMatcher::add is used. Optional mask (or masks) can be passed to specify which query and training descriptors can be matched. Namely, queryDescriptors[i] can be matched with trainDescriptors[j] only if mask.at\<uchar\>(i,j) is non-zero.
        """

    @overload
    def match(self, queryDescriptors, masks = ...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Matches. If a query descriptor is masked out in mask , no match is added for this descriptor. So, matches size may be smaller than the query descriptors count.
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        """

    def radiusMatch(self, queryDescriptors, trainDescriptors, maxDistance, mask = ..., compactResult = ...) -> matches:
        """
        @brief For each query descriptor, finds the training descriptors not farther than the specified distance.

        @param queryDescriptors Query set of descriptors.
        @param trainDescriptors Train set of descriptors. This set is not added to the train descriptors collection stored in the class object.
        @param matches Found matches.
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        @param maxDistance Threshold for the distance between matched descriptors. Distance means here metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        @param mask Mask specifying permissible matches between an input query and train matrices of descriptors.  For each query descriptor, the methods find such training descriptors that the distance between the query descriptor and the training descriptor is equal or smaller than maxDistance. Found matches are returned in the distance increasing order.
        """

    @overload
    def radiusMatch(self, queryDescriptors, maxDistance, masks = ..., compactResult = ...) -> matches:
        """
        @overload
        @param queryDescriptors Query set of descriptors.
        @param matches Found matches.
        @param maxDistance Threshold for the distance between matched descriptors. Distance means here metric distance (e.g. Hamming distance), not the distance between coordinates (which is measured in Pixels)!
        @param masks Set of masks. Each masks[i] specifies permissible matches between the input query descriptors and stored train descriptors from the i-th image trainDescCollection[i].
        @param compactResult Parameter used when the mask (or masks) is not empty. If compactResult is false, the matches vector has the same size as queryDescriptors rows. If compactResult is true, the matches vector does not contain matches for fully masked-out query descriptors.
        """

    def read(self, fileName) -> None:
        """"""

    def read(self, arg1) -> None:
        """"""

    def train(self) -> None:
        """
        @brief Trains a descriptor matcher

        Trains a descriptor matcher (for example, the flann index). In all methods to match, the method
        train() is run every time before matching. Some descriptor matchers (for example, BruteForceMatcher)
        have an empty implementation of this method. Other matchers really train their inner structures (for
        example, FlannBasedMatcher trains flann::Index ).
        """

    def write(self, fileName) -> None:
        """"""

    def write(self, fs, name) -> None:
        """"""

    def create(self, descriptorMatcherType) -> retval:
        """
        @brief Creates a descriptor matcher of a given type with the default parameters (using default
        constructor).

        @param descriptorMatcherType Descriptor matcher type. Now the following matcher types are supported: -   `BruteForce` (it uses L2 ) -   `BruteForce-L1` -   `BruteForce-Hamming` -   `BruteForce-Hamming(2)` -   `FlannBased`
        """

    def create(self, matcherType) -> retval:
        """"""


class FaceDetectorYN(builtins.object):
    def detect(self, image, faces = ...) -> tuple[retval, faces]:
        """
        @brief A simple interface to detect face from given image
        *
        *  @param image an image to detect
        *  @param faces detection results stored in a cv::Mat
        """

    def getInputSize(self) -> retval:
        """"""

    def getNMSThreshold(self) -> retval:
        """"""

    def getScoreThreshold(self) -> retval:
        """"""

    def getTopK(self) -> retval:
        """"""

    def setInputSize(self, input_size) -> None:
        """
        @brief Set the size for the network input, which overwrites the input size of creating model. Call this method when the size of input image does not match the input size when creating model
        *
        * @param input_size the size of the input image
        """

    def setNMSThreshold(self, nms_threshold) -> None:
        """
        @brief Set the Non-maximum-suppression threshold to suppress bounding boxes that have IoU greater than the given value
        *
        * @param nms_threshold threshold for NMS operation
        """

    def setScoreThreshold(self, score_threshold) -> None:
        """
        @brief Set the score threshold to filter out bounding boxes of score less than the given value
        *
        * @param score_threshold threshold for filtering out bounding boxes
        """

    def setTopK(self, top_k) -> None:
        """
        @brief Set the number of bounding boxes preserved before NMS
        *
        * @param top_k the number of bounding boxes to preserve from top rank based on score
        """

    def create(self, model, config, input_size, score_threshold = ..., nms_threshold = ..., top_k = ..., backend_id = ..., target_id = ...) -> retval:
        """
        @brief Creates an instance of this class with given parameters
        *
        *  @param model the path to the requested model
        *  @param config the path to the config file for compability, which is not requested for ONNX models
        *  @param input_size the size of the input image
        *  @param score_threshold the threshold to filter out bounding boxes of score smaller than the given value
        *  @param nms_threshold the threshold to suppress bounding boxes of IoU bigger than the given value
        *  @param top_k keep top K bboxes before NMS
        *  @param backend_id the id of backend
        *  @param target_id the id of target device
        """


class FaceRecognizerSF(builtins.object):
    def alignCrop(self, src_img, face_box, aligned_img = ...) -> aligned_img:
        """
        @brief Aligning image to put face on the standard position
        *  @param src_img input image
        *  @param face_box the detection result used for indicate face in input image
        *  @param aligned_img output aligned image
        """

    def feature(self, aligned_img, face_feature = ...) -> face_feature:
        """
        @brief Extracting face feature from aligned image
        *  @param aligned_img input aligned image
        *  @param face_feature output face feature
        """

    def match(self, face_feature1, face_feature2, dis_type = ...) -> retval:
        """
        @brief Calculating the distance between two face features
        *  @param face_feature1 the first input feature
        *  @param face_feature2 the second input feature of the same size and the same type as face_feature1
        *  @param dis_type defining the similarity with optional values "FR_OSINE" or "FR_NORM_L2"
        """

    def create(self, model, config, backend_id = ..., target_id = ...) -> retval:
        """
        @brief Creates an instance of this class with given parameters
        *  @param model the path of the onnx model used for face recognition
        *  @param config the path to the config file for compability, which is not requested for ONNX models
        *  @param backend_id the id of backend
        *  @param target_id the id of target device
        """


class FarnebackOpticalFlow(DenseOpticalFlow):
    def getFastPyramids(self) -> retval:
        """"""

    def getFlags(self) -> retval:
        """"""

    def getNumIters(self) -> retval:
        """"""

    def getNumLevels(self) -> retval:
        """"""

    def getPolyN(self) -> retval:
        """"""

    def getPolySigma(self) -> retval:
        """"""

    def getPyrScale(self) -> retval:
        """"""

    def getWinSize(self) -> retval:
        """"""

    def setFastPyramids(self, fastPyramids) -> None:
        """"""

    def setFlags(self, flags) -> None:
        """"""

    def setNumIters(self, numIters) -> None:
        """"""

    def setNumLevels(self, numLevels) -> None:
        """"""

    def setPolyN(self, polyN) -> None:
        """"""

    def setPolySigma(self, polySigma) -> None:
        """"""

    def setPyrScale(self, pyrScale) -> None:
        """"""

    def setWinSize(self, winSize) -> None:
        """"""

    def create(self, numLevels = ..., pyrScale = ..., fastPyramids = ..., winSize = ..., numIters = ..., polyN = ..., polySigma = ..., flags = ...) -> retval:
        """"""


class FastFeatureDetector(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getNonmaxSuppression(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getType(self) -> retval:
        """"""

    def setNonmaxSuppression(self, f) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setType(self, type) -> None:
        """"""

    def create(self, threshold = ..., nonmaxSuppression = ..., type = ...) -> retval:
        """"""


class Feature2D(builtins.object):
    def compute(self, image, keypoints, descriptors = ...) -> tuple[keypoints, descriptors]:
        """
        @brief Computes the descriptors for a set of keypoints detected in an image (first variant) or image set
        (second variant).

        @param image Image.
        @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
        @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
        """

    @overload
    def compute(self, images, keypoints, descriptors = ...) -> tuple[keypoints, descriptors]:
        """
        @overload

        @param images Image set.
        @param keypoints Input collection of keypoints. Keypoints for which a descriptor cannot be computed are removed. Sometimes new keypoints can be added, for example: SIFT duplicates keypoint with several dominant orientations (for each orientation).
        @param descriptors Computed descriptors. In the second variant of the method descriptors[i] are descriptors computed for a keypoints[i]. Row j is the keypoints (or keypoints[i]) is the descriptor for keypoint j-th keypoint.
        """

    def defaultNorm(self) -> retval:
        """"""

    def descriptorSize(self) -> retval:
        """"""

    def descriptorType(self) -> retval:
        """"""

    def detect(self, image, mask = ...) -> keypoints:
        """
        @brief Detects keypoints in an image (first variant) or image set (second variant).

        @param image Image.
        @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
        @param mask Mask specifying where to look for keypoints (optional). It must be a 8-bit integer matrix with non-zero values in the region of interest.
        """

    @overload
    def detect(self, images, masks = ...) -> keypoints:
        """
        @overload
        @param images Image set.
        @param keypoints The detected keypoints. In the second variant of the method keypoints[i] is a set of keypoints detected in images[i] .
        @param masks Masks for each input image specifying where to look for keypoints (optional). masks[i] is a mask for images[i].
        """

    def detectAndCompute(self, image, mask, descriptors = ..., useProvidedKeypoints = ...) -> tuple[keypoints, descriptors]:
        """
        Detects keypoints and computes the descriptors
        """

    def empty(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def read(self, fileName) -> None:
        """"""

    def read(self, arg1) -> None:
        """"""

    def write(self, fileName) -> None:
        """"""

    def write(self, fs, name) -> None:
        """"""


class FileNode(builtins.object):
    @overload
    def at(self, i) -> retval:
        """
        @overload
        @param i Index of an element in the sequence node.
        """

    def empty(self) -> retval:
        """"""

    @overload
    def getNode(self, nodename) -> retval:
        """
        @overload
        @param nodename Name of an element in the mapping node.
        """

    def isInt(self) -> retval:
        """"""

    def isMap(self) -> retval:
        """"""

    def isNamed(self) -> retval:
        """"""

    def isNone(self) -> retval:
        """"""

    def isReal(self) -> retval:
        """"""

    def isSeq(self) -> retval:
        """"""

    def isString(self) -> retval:
        """"""

    def keys(self) -> retval:
        """
        @brief Returns keys of a mapping node.
        @returns Keys of a mapping node.
        """

    def mat(self) -> retval:
        """"""

    def name(self) -> retval:
        """"""

    def rawSize(self) -> retval:
        """"""

    def real(self) -> retval:
        """
        Internal method used when reading FileStorage.
        Sets the type (int, real or string) and value of the previously created node.
        """

    def size(self) -> retval:
        """"""

    def string(self) -> retval:
        """"""

    def type(self) -> retval:
        """
        @brief Returns type of the node.
        @returns Type of the node. See FileNode::Type
        """


class FileStorage(builtins.object):
    def endWriteStruct(self) -> None:
        """
        @brief Finishes writing nested structure (should pair startWriteStruct())
        """

    def getFirstTopLevelNode(self) -> retval:
        """
        @brief Returns the first element of the top-level mapping.
        @returns The first element of the top-level mapping.
        """

    def getFormat(self) -> retval:
        """
        @brief Returns the current format.
        * @returns The current format, see FileStorage::Mode
        """

    @overload
    def getNode(self, nodename) -> retval:
        """
        @overload
        """

    def isOpened(self) -> retval:
        """
        @brief Checks whether the file is opened.

        @returns true if the object is associated with the current file and false otherwise. It is a
        good practice to call this method after you tried to open a file.
        """

    def open(self, filename, flags, encoding = ...) -> retval:
        """
        @brief Opens a file.

        See description of parameters in FileStorage::FileStorage. The method calls FileStorage::release
        before opening the file.
        @param filename Name of the file to open or the text string to read the data from. Extension of the file (.xml, .yml/.yaml or .json) determines its format (XML, YAML or JSON respectively). Also you can append .gz to work with compressed files, for example myHugeMatrix.xml.gz. If both FileStorage::WRITE and FileStorage::MEMORY flags are specified, source is used just to specify the output file format (e.g. mydata.xml, .yml etc.). A file name can also contain parameters. You can use this format, "*?base64" (e.g. "file.json?base64" (case sensitive)), as an alternative to FileStorage::BASE64 flag.
        @param flags Mode of operation. One of FileStorage::Mode
        @param encoding Encoding of the file. Note that UTF-16 XML encoding is not supported currently and you should use 8-bit encoding instead of it.
        """

    def release(self) -> None:
        """
        @brief Closes the file and releases all the memory buffers.

        Call this method after all I/O operations with the storage are finished.
        """

    def releaseAndGetString(self) -> retval:
        """
        @brief Closes the file and releases all the memory buffers.

        Call this method after all I/O operations with the storage are finished. If the storage was
        opened for writing data and FileStorage::WRITE was specified
        """

    def root(self, streamidx = ...) -> retval:
        """
        @brief Returns the top-level mapping
        @param streamidx Zero-based index of the stream. In most cases there is only one stream in the file. However, YAML supports multiple streams and so there can be several. @returns The top-level mapping.
        """

    def startWriteStruct(self, name, flags, typeName = ...) -> None:
        """
        @brief Starts to write a nested structure (sequence or a mapping).
        @param name name of the structure. When writing to sequences (a.k.a. "arrays"), pass an empty string.
        @param flags type of the structure (FileNode::MAP or FileNode::SEQ (both with optional FileNode::FLOW)).
        @param typeName optional name of the type you store. The effect of setting this depends on the storage format. I.e. if the format has a specification for storing type information, this parameter is used.
        """

    def write(self, name, val) -> None:
        """
        * @brief Simplified writing API to use with bindings.
        * @param name Name of the written object. When writing to sequences (a.k.a. "arrays"), pass an empty string.
        * @param val Value of the written object.
        """

    def writeComment(self, comment, append = ...) -> None:
        """
        @brief Writes a comment.

        The function writes a comment into file storage. The comments are skipped when the storage is read.
        @param comment The written comment, single-line or multi-line
        @param append If true, the function tries to put the comment at the end of current line. Else if the comment is multi-line, or if it does not fit at the end of the current line, the comment starts a new line.
        """


class FlannBasedMatcher(DescriptorMatcher):
    def create(self) -> retval:
        """"""


class GArray(builtins.object):
    ...


class GArrayDesc(builtins.object):
    ...


class GArrayT(builtins.object):
    def type(self) -> retval:
        """"""


class GCompileArg(builtins.object):
    ...


class GComputation(builtins.object):
    def apply(self, callback, args = ...) -> retval:
        """
        * @brief Compile graph on-the-fly and immediately execute it on
        * the inputs data vectors.
        *
        * Number of input/output data objects must match GComputation's
        * protocol, also types of host data objects (cv::Mat, cv::Scalar)
        * must match the shapes of data objects from protocol (cv::GMat,
        * cv::GScalar). If there's a mismatch, a run-time exception will
        * be generated.
        *
        * Internally, a cv::GCompiled object is created for the given
        * input format configuration, which then is executed on the input
        * data immediately. cv::GComputation caches compiled objects
        * produced within apply() -- if this method would be called next
        * time with the same input parameters (image formats, image
        * resolution, etc), the underlying compiled graph will be reused
        * without recompilation. If new metadata doesn't match the cached
        * one, the underlying compiled graph is regenerated.
        *
        * @note compile() always triggers a compilation process and
        * produces a new GCompiled object regardless if a similar one has
        * been cached via apply() or not.
        *
        * @param ins vector of input data to process. Don't create * GRunArgs object manually, use cv::gin() wrapper instead.
        * @param outs vector of output data to fill results in. cv::Mat * objects may be empty in this vector, G-API will automatically * initialize it with the required format & dimensions. Don't * create GRunArgsP object manually, use cv::gout() wrapper instead.
        * @param args a list of compilation arguments to pass to the * underlying compilation process. Don't create GCompileArgs * object manually, use cv::compile_args() wrapper instead. * * @sa @ref gapi_data_objects, @ref gapi_compile_args
        """

    def compileStreaming(self, in_metas, args = ...) -> retval:
        """
        * @brief Compile the computation for streaming mode.
        *
        * This method triggers compilation process and produces a new
        * GStreamingCompiled object which then can process video stream
        * data of the given format. Passing a stream in a different
        * format to the compiled computation will generate a run-time
        * exception.
        *
        * @param in_metas vector of input metadata configuration. Grab * metadata from real data objects (like cv::Mat or cv::Scalar) * using cv::descr_of(), or create it on your own. *
        * @param args compilation arguments for this compilation * process. Compilation arguments directly affect what kind of * executable object would be produced, e.g. which kernels (and * thus, devices) would be used to execute computation. * * @return GStreamingCompiled, a streaming-oriented executable * computation compiled specifically for the given input * parameters. * * @sa @ref gapi_compile_args
        """

    def compileStreaming(self, args = ...) -> retval:
        """
        * @brief Compile the computation for streaming mode.
        *
        * This method triggers compilation process and produces a new
        * GStreamingCompiled object which then can process video stream
        * data in any format. Underlying mechanisms will be adjusted to
        * every new input video stream automatically, but please note that
        * _not all_ existing backends support this (see reshape()).
        *
        * @param args compilation arguments for this compilation * process. Compilation arguments directly affect what kind of * executable object would be produced, e.g. which kernels (and * thus, devices) would be used to execute computation. * * @return GStreamingCompiled, a streaming-oriented executable * computation compiled for any input image format. * * @sa @ref gapi_compile_args
        """

    def compileStreaming(self, callback, args = ...) -> retval:
        """"""


class GFTTDetector(Feature2D):
    def getBlockSize(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getGradientSize(self) -> retval:
        """"""

    def getHarrisDetector(self) -> retval:
        """"""

    def getK(self) -> retval:
        """"""

    def getMaxFeatures(self) -> retval:
        """"""

    def getMinDistance(self) -> retval:
        """"""

    def getQualityLevel(self) -> retval:
        """"""

    def setBlockSize(self, blockSize) -> None:
        """"""

    def setGradientSize(self, gradientSize_) -> None:
        """"""

    def setHarrisDetector(self, val) -> None:
        """"""

    def setK(self, k) -> None:
        """"""

    def setMaxFeatures(self, maxFeatures) -> None:
        """"""

    def setMinDistance(self, minDistance) -> None:
        """"""

    def setQualityLevel(self, qlevel) -> None:
        """"""

    def create(self, maxCorners = ..., qualityLevel = ..., minDistance = ..., blockSize = ..., useHarrisDetector = ..., k = ...) -> retval:
        """"""

    def create(self, maxCorners, qualityLevel, minDistance, blockSize, gradiantSize, useHarrisDetector = ..., k = ...) -> retval:
        """"""


class GFrame(builtins.object):
    ...


class GInferInputs(builtins.object):
    def setInput(self, name, value) -> retval:
        """"""


class GInferListInputs(builtins.object):
    def setInput(self, name, value) -> retval:
        """"""


class GInferListOutputs(builtins.object):
    def at(self, name) -> retval:
        """"""


class GInferOutputs(builtins.object):
    def at(self, name) -> retval:
        """"""


class GKernelPackage(builtins.object):
    ...


class GMat(builtins.object):
    ...


class GMatDesc(builtins.object):
    def asInterleaved(self) -> retval:
        """"""

    def asPlanar(self) -> retval:
        """"""

    def asPlanar(self, planes) -> retval:
        """"""

    def withDepth(self, ddepth) -> retval:
        """"""

    def withSize(self, sz) -> retval:
        """"""

    def withSizeDelta(self, delta) -> retval:
        """"""

    def withSizeDelta(self, dx, dy) -> retval:
        """"""

    def withType(self, ddepth, dchan) -> retval:
        """"""


class GOpaque(builtins.object):
    ...


class GOpaqueDesc(builtins.object):
    ...


class GOpaqueT(builtins.object):
    def type(self) -> retval:
        """"""


class GScalar(builtins.object):
    ...


class GScalarDesc(builtins.object):
    ...


class GStreamingCompiled(builtins.object):
    def pull(self) -> retval:
        """
        * @brief Get the next processed frame from the pipeline.
        *
        * Use gout() to create an output parameter vector.
        *
        * Output vectors must have the same number of elements as defined
        * in the cv::GComputation protocol (at the moment of its
        * construction). Shapes of elements also must conform to protocol
        * (e.g. cv::Mat needs to be passed where cv::GMat has been
        * declared as output, and so on). Run-time exception is generated
        * on type mismatch.
        *
        * This method writes new data into objects passed via output
        * vector.  If there is no data ready yet, this method blocks. Use
        * try_pull() if you need a non-blocking version.
        *
        * @param outs vector of output parameters to obtain. * @return true if next result has been obtained, *    false marks end of the stream.
        """

    def running(self) -> retval:
        """
        * @brief Test if the pipeline is running.
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        *
        * @return true if the current stream is not over yet.
        """

    def setSource(self, callback) -> None:
        """
        * @brief Specify the input data to GStreamingCompiled for
        * processing, a generic version.
        *
        * Use gin() to create an input parameter vector.
        *
        * Input vectors must have the same number of elements as defined
        * in the cv::GComputation protocol (at the moment of its
        * construction). Shapes of elements also must conform to protocol
        * (e.g. cv::Mat needs to be passed where cv::GMat has been
        * declared as input, and so on). Run-time exception is generated
        * on type mismatch.
        *
        * In contrast with regular GCompiled, user can also pass an
        * object of type GVideoCapture for a GMat parameter of the parent
        * GComputation.  The compiled pipeline will start fetching data
        * from that GVideoCapture and feeding it into the
        * pipeline. Pipeline stops when a GVideoCapture marks end of the
        * stream (or when stop() is called).
        *
        * Passing a regular Mat for a GMat parameter makes it "infinite"
        * source -- pipeline may run forever feeding with this Mat until
        * stopped explicitly.
        *
        * Currently only a single GVideoCapture is supported as input. If
        * the parent GComputation is declared with multiple input GMat's,
        * one of those can be specified as GVideoCapture but all others
        * must be regular Mat objects.
        *
        * Throws if pipeline is already running. Use stop() and then
        * setSource() to run the graph on a new video stream.
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        *
        * @param ins vector of inputs to process. * @sa gin
        """

    def start(self) -> None:
        """
        * @brief Start the pipeline execution.
        *
        * Use pull()/try_pull() to obtain data. Throws an exception if
        * a video source was not specified.
        *
        * setSource() must be called first, even if the pipeline has been
        * working already and then stopped (explicitly via stop() or due
        * stream completion)
        *
        * @note This method is not thread-safe (with respect to the user
        * side) at the moment. Protect the access if
        * start()/stop()/setSource() may be called on the same object in
        * multiple threads in your application.
        """

    def stop(self) -> None:
        """
        * @brief Stop (abort) processing the pipeline.
        *
        * Note - it is not pause but a complete stop. Calling start()
        * will cause G-API to start processing the stream from the early beginning.
        *
        * Throws if the pipeline is not running.
        """


class GeneralizedHough(Algorithm):
    def detect(self, image, positions = ..., votes = ...) -> tuple[positions, votes]:
        """"""

    def detect(self, edges, dx, dy, positions = ..., votes = ...) -> tuple[positions, votes]:
        """"""

    def getCannyHighThresh(self) -> retval:
        """"""

    def getCannyLowThresh(self) -> retval:
        """"""

    def getDp(self) -> retval:
        """"""

    def getMaxBufferSize(self) -> retval:
        """"""

    def getMinDist(self) -> retval:
        """"""

    def setCannyHighThresh(self, cannyHighThresh) -> None:
        """"""

    def setCannyLowThresh(self, cannyLowThresh) -> None:
        """"""

    def setDp(self, dp) -> None:
        """"""

    def setMaxBufferSize(self, maxBufferSize) -> None:
        """"""

    def setMinDist(self, minDist) -> None:
        """"""

    def setTemplate(self, templ, templCenter = ...) -> None:
        """"""

    def setTemplate(self, edges, dx, dy, templCenter = ...) -> None:
        """"""


class GeneralizedHoughBallard(GeneralizedHough):
    def getLevels(self) -> retval:
        """"""

    def getVotesThreshold(self) -> retval:
        """"""

    def setLevels(self, levels) -> None:
        """"""

    def setVotesThreshold(self, votesThreshold) -> None:
        """"""


class GeneralizedHoughGuil(GeneralizedHough):
    def getAngleEpsilon(self) -> retval:
        """"""

    def getAngleStep(self) -> retval:
        """"""

    def getAngleThresh(self) -> retval:
        """"""

    def getLevels(self) -> retval:
        """"""

    def getMaxAngle(self) -> retval:
        """"""

    def getMaxScale(self) -> retval:
        """"""

    def getMinAngle(self) -> retval:
        """"""

    def getMinScale(self) -> retval:
        """"""

    def getPosThresh(self) -> retval:
        """"""

    def getScaleStep(self) -> retval:
        """"""

    def getScaleThresh(self) -> retval:
        """"""

    def getXi(self) -> retval:
        """"""

    def setAngleEpsilon(self, angleEpsilon) -> None:
        """"""

    def setAngleStep(self, angleStep) -> None:
        """"""

    def setAngleThresh(self, angleThresh) -> None:
        """"""

    def setLevels(self, levels) -> None:
        """"""

    def setMaxAngle(self, maxAngle) -> None:
        """"""

    def setMaxScale(self, maxScale) -> None:
        """"""

    def setMinAngle(self, minAngle) -> None:
        """"""

    def setMinScale(self, minScale) -> None:
        """"""

    def setPosThresh(self, posThresh) -> None:
        """"""

    def setScaleStep(self, scaleStep) -> None:
        """"""

    def setScaleThresh(self, scaleThresh) -> None:
        """"""

    def setXi(self, xi) -> None:
        """"""


class HOGDescriptor(builtins.object):
    def checkDetectorSize(self) -> retval:
        """
        @brief Checks if detector size equal to descriptor size.
        """

    def compute(self, img, winStride = ..., padding = ..., locations = ...) -> descriptors:
        """
        @brief Computes HOG descriptors of given image.
        @param img Matrix of the type CV_8U containing an image where HOG features will be calculated.
        @param descriptors Matrix of the type CV_32F
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param locations Vector of Point
        """

    def computeGradient(self, img, grad, angleOfs, paddingTL = ..., paddingBR = ...) -> tuple[grad, angleOfs]:
        """
        @brief  Computes gradients and quantized gradient orientations.
        @param img Matrix contains the image to be computed
        @param grad Matrix of type CV_32FC2 contains computed gradients
        @param angleOfs Matrix of type CV_8UC2 contains quantized gradient orientations
        @param paddingTL Padding from top-left
        @param paddingBR Padding from bottom-right
        """

    def detect(self, img, hitThreshold = ..., winStride = ..., padding = ..., searchLocations = ...) -> tuple[foundLocations, weights]:
        """
        @brief Performs object detection without a multi-scale window.
        @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
        @param foundLocations Vector of point where each point contains left-top corner point of detected object boundaries.
        @param weights Vector that will contain confidence values for each detected object.
        @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param searchLocations Vector of Point includes set of requested locations to be evaluated.
        """

    def detectMultiScale(self, img, hitThreshold = ..., winStride = ..., padding = ..., scale = ..., groupThreshold = ..., useMeanshiftGrouping = ...) -> tuple[foundLocations, foundWeights]:
        """
        @brief Detects objects of different sizes in the input image. The detected objects are returned as a list
        of rectangles.
        @param img Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
        @param foundLocations Vector of rectangles where each rectangle contains the detected object.
        @param foundWeights Vector that will contain confidence values for each detected object.
        @param hitThreshold Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
        @param winStride Window stride. It must be a multiple of block stride.
        @param padding Padding
        @param scale Coefficient of the detection window increase.
        @param groupThreshold Coefficient to regulate the similarity threshold. When detected, some objects can be covered by many rectangles. 0 means not to perform grouping.
        @param useMeanshiftGrouping indicates grouping algorithm
        """

    def getDescriptorSize(self) -> retval:
        """
        @brief Returns the number of coefficients required for the classification.
        """

    def getWinSigma(self) -> retval:
        """
        @brief Returns winSigma value
        """

    def load(self, filename, objname = ...) -> retval:
        """
        @brief loads HOGDescriptor parameters and coefficients for the linear SVM classifier from a file
        @param filename Name of the file to read.
        @param objname The optional name of the node to read (if empty, the first top-level node will be used).
        """

    def save(self, filename, objname = ...) -> None:
        """
        @brief saves HOGDescriptor parameters and coefficients for the linear SVM classifier to a file
        @param filename File name
        @param objname Object name
        """

    def setSVMDetector(self, svmdetector) -> None:
        """
        @brief Sets coefficients for the linear SVM classifier.
        @param svmdetector coefficients for the linear SVM classifier.
        """

    def getDaimlerPeopleDetector(self) -> retval:
        """
        @brief Returns coefficients of the classifier trained for people detection (for 48x96 windows).
        """

    def getDefaultPeopleDetector(self) -> retval:
        """
        @brief Returns coefficients of the classifier trained for people detection (for 64x128 windows).
        """


class KAZE(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getDiffusivity(self) -> retval:
        """"""

    def getExtended(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getNOctaves(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """"""

    def getUpright(self) -> retval:
        """"""

    def setDiffusivity(self, diff) -> None:
        """"""

    def setExtended(self, extended) -> None:
        """"""

    def setNOctaveLayers(self, octaveLayers) -> None:
        """"""

    def setNOctaves(self, octaves) -> None:
        """"""

    def setThreshold(self, threshold) -> None:
        """"""

    def setUpright(self, upright) -> None:
        """"""

    def create(self, extended = ..., upright = ..., threshold = ..., nOctaves = ..., nOctaveLayers = ..., diffusivity = ...) -> retval:
        """
        @brief The KAZE constructor

        @param extended Set to enable extraction of extended (128-byte) descriptor.
        @param upright Set to enable use of upright descriptors (non rotation-invariant).
        @param threshold Detector response threshold to accept point
        @param nOctaves Maximum octave evolution of the image
        @param nOctaveLayers Default number of sublevels per scale level
        @param diffusivity Diffusivity type. DIFF_PM_G1, DIFF_PM_G2, DIFF_WEICKERT or DIFF_CHARBONNIER
        """


class KalmanFilter(builtins.object):
    def correct(self, measurement) -> retval:
        """
        @brief Updates the predicted state from the measurement.

        @param measurement The measured system parameters
        """

    def predict(self, control = ...) -> retval:
        """
        @brief Computes a predicted state.

        @param control The optional input control
        """


class KeyPoint(builtins.object):
    def convert(self, keypoints, keypointIndexes = ...) -> points2f:
        """
        This method converts vector of keypoints to vector of points or the reverse, where each keypoint is
        assigned the same size and the same orientation.

        @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
        @param points2f Array of (x,y) coordinates of each keypoint
        @param keypointIndexes Array of indexes of keypoints to be converted to points. (Acts like a mask to convert only specified keypoints)
        """

    @overload
    def convert(self, points2f, size = ..., response = ..., octave = ..., class_id = ...) -> keypoints:
        """
        @overload
        @param points2f Array of (x,y) coordinates of each keypoint
        @param keypoints Keypoints obtained from any feature detection algorithm like SIFT/SURF/ORB
        @param size keypoint diameter
        @param response keypoint detector response on the keypoint (that is, strength of the keypoint)
        @param octave pyramid octave in which the keypoint has been detected
        @param class_id object id
        """

    def overlap(self, kp1, kp2) -> retval:
        """
        This method computes overlap for pair of keypoints. Overlap is the ratio between area of keypoint
        regions' intersection and area of keypoint regions' union (considering keypoint region as circle).
        If they don't overlap, we get zero. If they coincide at same location with same size, we get 1.
        @param kp1 First keypoint
        @param kp2 Second keypoint
        """


class LineSegmentDetector(Algorithm):
    def compareSegments(self, size, lines1, lines2, image = ...) -> tuple[retval, image]:
        """
        @brief Draws two groups of lines in blue and red, counting the non overlapping (mismatching) pixels.

        @param size The size of the image, where lines1 and lines2 were found.
        @param lines1 The first group of lines that needs to be drawn. It is visualized in blue color.
        @param lines2 The second group of lines. They visualized in red color.
        @param image Optional image, where the lines will be drawn. The image should be color(3-channel) in order for lines1 and lines2 to be drawn in the above mentioned colors.
        """

    def detect(self, image, lines = ..., width = ..., prec = ..., nfa = ...) -> tuple[lines, width, prec, nfa]:
        """
        @brief Finds lines in the input image.

        This is the output of the default parameters of the algorithm on the above shown image.

        ![image](pics/building_lsd.png)

        @param image A grayscale (CV_8UC1) input image. If only a roi needs to be selected, use: `lsd_ptr-\>detect(image(roi), lines, ...); lines += Scalar(roi.x, roi.y, roi.x, roi.y);`
        @param lines A vector of Vec4f elements specifying the beginning and ending point of a line. Where Vec4f is (x1, y1, x2, y2), point 1 is the start, point 2 - end. Returned lines are strictly oriented depending on the gradient.
        @param width Vector of widths of the regions, where the lines are found. E.g. Width of line.
        @param prec Vector of precisions with which the lines are found.
        @param nfa Vector containing number of false alarms in the line region, with precision of 10%. The bigger the value, logarithmically better the detection. - -1 corresponds to 10 mean false alarms - 0 corresponds to 1 mean false alarm - 1 corresponds to 0.1 mean false alarms This vector will be calculated only when the objects type is #LSD_REFINE_ADV.
        """

    def drawSegments(self, image, lines) -> image:
        """
        @brief Draws the line segments on a given image.
        @param image The image, where the lines will be drawn. Should be bigger or equal to the image, where the lines were found.
        @param lines A vector of the lines that needed to be drawn.
        """


class MSER(Feature2D):
    def detectRegions(self, image) -> tuple[msers, bboxes]:
        """
        @brief Detect %MSER regions

        @param image input image (8UC1, 8UC3 or 8UC4, must be greater or equal than 3x3)
        @param msers resulting list of point sets
        @param bboxes resulting bounding boxes
        """

    def getAreaThreshold(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getDelta(self) -> retval:
        """"""

    def getEdgeBlurSize(self) -> retval:
        """"""

    def getMaxArea(self) -> retval:
        """"""

    def getMaxEvolution(self) -> retval:
        """"""

    def getMaxVariation(self) -> retval:
        """"""

    def getMinArea(self) -> retval:
        """"""

    def getMinDiversity(self) -> retval:
        """"""

    def getMinMargin(self) -> retval:
        """"""

    def getPass2Only(self) -> retval:
        """"""

    def setAreaThreshold(self, areaThreshold) -> None:
        """"""

    def setDelta(self, delta) -> None:
        """"""

    def setEdgeBlurSize(self, edge_blur_size) -> None:
        """"""

    def setMaxArea(self, maxArea) -> None:
        """"""

    def setMaxEvolution(self, maxEvolution) -> None:
        """"""

    def setMaxVariation(self, maxVariation) -> None:
        """"""

    def setMinArea(self, minArea) -> None:
        """"""

    def setMinDiversity(self, minDiversity) -> None:
        """"""

    def setMinMargin(self, min_margin) -> None:
        """"""

    def setPass2Only(self, f) -> None:
        """"""

    def create(self, delta = ..., min_area = ..., max_area = ..., max_variation = ..., min_diversity = ..., max_evolution = ..., area_threshold = ..., min_margin = ..., edge_blur_size = ...) -> retval:
        """
        @brief Full constructor for %MSER detector

        @param delta it compares \f$(size_{i}-size_{i-delta})/size_{i-delta}\f$
        @param min_area prune the area which smaller than minArea
        @param max_area prune the area which bigger than maxArea
        @param max_variation prune the area have similar size to its children
        @param min_diversity for color image, trace back to cut off mser with diversity less than min_diversity
        @param max_evolution  for color image, the evolution steps
        @param area_threshold for color image, the area threshold to cause re-initialize
        @param min_margin for color image, ignore too small margin
        @param edge_blur_size for color image, the aperture size for edge blur
        """


class Mat(numpy.ndarray):
    ...


class MergeDebevec(MergeExposures):
    def process(self, src, times, response, dst = ...) -> dst:
        """"""

    def process(self, src, times, dst = ...) -> dst:
        """"""


class MergeExposures(Algorithm):
    def process(self, src, times, response, dst = ...) -> dst:
        """
        @brief Merges images.

        @param src vector of input images
        @param dst result image
        @param times vector of exposure time values for each image
        @param response 256x1 matrix with inverse camera response function for each pixel value, it should have the same number of channels as images.
        """


class MergeMertens(MergeExposures):
    def getContrastWeight(self) -> retval:
        """"""

    def getExposureWeight(self) -> retval:
        """"""

    def getSaturationWeight(self) -> retval:
        """"""

    def process(self, src, times, response, dst = ...) -> dst:
        """"""

    def process(self, src, dst = ...) -> dst:
        """
        @brief Short version of process, that doesn't take extra arguments.

        @param src vector of input images
        @param dst result image
        """

    def setContrastWeight(self, contrast_weiht) -> None:
        """"""

    def setExposureWeight(self, exposure_weight) -> None:
        """"""

    def setSaturationWeight(self, saturation_weight) -> None:
        """"""


class MergeRobertson(MergeExposures):
    def process(self, src, times, response, dst = ...) -> dst:
        """"""

    def process(self, src, times, dst = ...) -> dst:
        """"""


class ORB(Feature2D):
    def getDefaultName(self) -> retval:
        """"""

    def getEdgeThreshold(self) -> retval:
        """"""

    def getFastThreshold(self) -> retval:
        """"""

    def getFirstLevel(self) -> retval:
        """"""

    def getMaxFeatures(self) -> retval:
        """"""

    def getNLevels(self) -> retval:
        """"""

    def getPatchSize(self) -> retval:
        """"""

    def getScaleFactor(self) -> retval:
        """"""

    def getScoreType(self) -> retval:
        """"""

    def getWTA_K(self) -> retval:
        """"""

    def setEdgeThreshold(self, edgeThreshold) -> None:
        """"""

    def setFastThreshold(self, fastThreshold) -> None:
        """"""

    def setFirstLevel(self, firstLevel) -> None:
        """"""

    def setMaxFeatures(self, maxFeatures) -> None:
        """"""

    def setNLevels(self, nlevels) -> None:
        """"""

    def setPatchSize(self, patchSize) -> None:
        """"""

    def setScaleFactor(self, scaleFactor) -> None:
        """"""

    def setScoreType(self, scoreType) -> None:
        """"""

    def setWTA_K(self, wta_k) -> None:
        """"""

    def create(self, nfeatures = ..., scaleFactor = ..., nlevels = ..., edgeThreshold = ..., firstLevel = ..., WTA_K = ..., scoreType = ..., patchSize = ..., fastThreshold = ...) -> retval:
        """
        @brief The ORB constructor

        @param nfeatures The maximum number of features to retain.
        @param scaleFactor Pyramid decimation ratio, greater than 1. scaleFactor==2 means the classical pyramid, where each next level has 4x less pixels than the previous, but such a big scale factor will degrade feature matching scores dramatically. On the other hand, too close to 1 scale factor will mean that to cover certain scale range you will need more pyramid levels and so the speed will suffer.
        @param nlevels The number of pyramid levels. The smallest level will have linear size equal to input_image_linear_size/pow(scaleFactor, nlevels - firstLevel).
        @param edgeThreshold This is size of the border where the features are not detected. It should roughly match the patchSize parameter.
        @param firstLevel The level of pyramid to put source image to. Previous layers are filled with upscaled source image.
        @param WTA_K The number of points that produce each element of the oriented BRIEF descriptor. The default value 2 means the BRIEF where we take a random point pair and compare their brightnesses, so we get 0/1 response. Other possible values are 3 and 4. For example, 3 means that we take 3 random points (of course, those point coordinates are random, but they are generated from the pre-defined seed, so each element of BRIEF descriptor is computed deterministically from the pixel rectangle), find point of maximum brightness and output index of the winner (0, 1 or 2). Such output will occupy 2 bits, and therefore it will need a special variant of Hamming distance, denoted as NORM_HAMMING2 (2 bits per bin). When WTA_K=4, we take 4 random points to compute each bin (that will also occupy 2 bits with possible values 0, 1, 2 or 3).
        @param scoreType The default HARRIS_SCORE means that Harris algorithm is used to rank features (the score is written to KeyPoint::score and is used to retain best nfeatures features); FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints, but it is a little faster to compute.
        @param patchSize size of the patch used by the oriented BRIEF descriptor. Of course, on smaller pyramid layers the perceived image area covered by a feature will be larger.
        @param fastThreshold the fast threshold
        """


class PyRotationWarper(builtins.object):
    def buildMaps(self, src_size, K, R, xmap = ..., ymap = ...) -> tuple[retval, xmap, ymap]:
        """
        @brief Builds the projection maps according to the given camera data.

        @param src_size Source image size
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param xmap Projection map for the x axis
        @param ymap Projection map for the y axis @return Projected image minimum bounding box
        """

    def getScale(self) -> retval:
        """"""

    def setScale(self, arg1) -> None:
        """"""

    def warp(self, src, K, R, interp_mode, border_mode, dst = ...) -> tuple[retval, dst]:
        """
        @brief Projects the image.

        @param src Source image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst Projected image @return Project image top-left corner
        """

    def warpBackward(self, src, K, R, interp_mode, border_mode, dst_size, dst = ...) -> dst:
        """
        @brief Projects the image backward.

        @param src Projected image
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix
        @param interp_mode Interpolation mode
        @param border_mode Border extrapolation mode
        @param dst_size Backward-projected image size
        @param dst Backward-projected image
        """

    def warpPoint(self, pt, K, R) -> retval:
        """
        @brief Projects the image point.

        @param pt Source point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Projected point
        """

    def warpPointBackward(self, pt, K, R) -> retval:
        """
        @brief Projects the image point backward.

        @param pt Projected point
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Backward-projected point
        """

    def warpRoi(self, src_size, K, R) -> retval:
        """
        @param src_size Source image bounding box
        @param K Camera intrinsic parameters
        @param R Camera rotation matrix @return Projected image minimum bounding box
        """


class QRCodeDetector(builtins.object):
    def decode(self, img, points, straight_qrcode = ...) -> tuple[retval, straight_qrcode]:
        """
        @brief Decodes QR code in image once it's found by the detect() method.

        Returns UTF8-encoded output string or empty string if the code cannot be decoded.
        @param img grayscale or color (BGR) image containing QR code.
        @param points Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def decodeCurved(self, img, points, straight_qrcode = ...) -> tuple[retval, straight_qrcode]:
        """
        @brief Decodes QR code on a curved surface in image once it's found by the detect() method.

        Returns UTF8-encoded output string or empty string if the code cannot be decoded.
        @param img grayscale or color (BGR) image containing QR code.
        @param points Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def decodeMulti(self, img, points, straight_qrcode = ...) -> tuple[retval, decoded_info, straight_qrcode]:
        """
        @brief Decodes QR codes in image once it's found by the detect() method.
        @param img grayscale or color (BGR) image containing QR codes.
        @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
        @param points vector of Quadrangle vertices found by detect() method (or some other algorithm).
        @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
        """

    def detect(self, img, points = ...) -> tuple[retval, points]:
        """
        @brief Detects QR code in image and returns the quadrangle containing the code.
        @param img grayscale or color (BGR) image containing (or not) QR code.
        @param points Output vector of vertices of the minimum-area quadrangle containing the code.
        """

    def detectAndDecode(self, img, points = ..., straight_qrcode = ...) -> tuple[retval, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR code

        @param img grayscale or color (BGR) image containing QR code.
        @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def detectAndDecodeCurved(self, img, points = ..., straight_qrcode = ...) -> tuple[retval, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR code on a curved surface

        @param img grayscale or color (BGR) image containing QR code.
        @param points optional output array of vertices of the found QR code quadrangle. Will be empty if not found.
        @param straight_qrcode The optional output image containing rectified and binarized QR code
        """

    def detectAndDecodeMulti(self, img, points = ..., straight_qrcode = ...) -> tuple[retval, decoded_info, points, straight_qrcode]:
        """
        @brief Both detects and decodes QR codes
        @param img grayscale or color (BGR) image containing QR codes.
        @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
        @param points optional output vector of vertices of the found QR code quadrangles. Will be empty if not found.
        @param straight_qrcode The optional output vector of images containing rectified and binarized QR codes
        """

    def detectMulti(self, img, points = ...) -> tuple[retval, points]:
        """
        @brief Detects QR codes in image and returns the vector of the quadrangles containing the codes.
        @param img grayscale or color (BGR) image containing (or not) QR codes.
        @param points Output vector of vector of vertices of the minimum-area quadrangle containing the codes.
        """

    def setEpsX(self, epsX) -> None:
        """
        @brief sets the epsilon used during the horizontal scan of QR code stop marker detection.
        @param epsX Epsilon neighborhood, which allows you to determine the horizontal pattern of the scheme 1:1:3:1:1 according to QR code standard.
        """

    def setEpsY(self, epsY) -> None:
        """
        @brief sets the epsilon used during the vertical scan of QR code stop marker detection.
        @param epsY Epsilon neighborhood, which allows you to determine the vertical pattern of the scheme 1:1:3:1:1 according to QR code standard.
        """

    def setUseAlignmentMarkers(self, useAlignmentMarkers) -> None:
        """
        @brief use markers to improve the position of the corners of the QR code
        *
        * alignmentMarkers using by default
        """


class QRCodeEncoder(builtins.object):
    def encode(self, encoded_info, qrcode = ...) -> qrcode:
        """
        @brief Generates QR code from input string.
        @param encoded_info Input string to encode.
        @param qrcode Generated QR code.
        """

    def encodeStructuredAppend(self, encoded_info, qrcodes = ...) -> qrcodes:
        """
        @brief Generates QR code from input string in Structured Append mode. The encoded message is splitting over a number of QR codes.
        @param encoded_info Input string to encode.
        @param qrcodes Vector of generated QR codes.
        """

    def create(self, parameters = ...) -> retval:
        """
        @brief Constructor
        @param parameters QR code encoder parameters QRCodeEncoder::Params
        """


class Params(builtins.object):
    ...


class SIFT(Feature2D):
    def getContrastThreshold(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getEdgeThreshold(self) -> retval:
        """"""

    def getNFeatures(self) -> retval:
        """"""

    def getNOctaveLayers(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def setContrastThreshold(self, contrastThreshold) -> None:
        """"""

    def setEdgeThreshold(self, edgeThreshold) -> None:
        """"""

    def setNFeatures(self, maxFeatures) -> None:
        """"""

    def setNOctaveLayers(self, nOctaveLayers) -> None:
        """"""

    def setSigma(self, sigma) -> None:
        """"""

    def create(self, nfeatures = ..., nOctaveLayers = ..., contrastThreshold = ..., edgeThreshold = ..., sigma = ...) -> retval:
        """
        @param nfeatures The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast) 
        @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution. 
        @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.  @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
        @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained). 
        @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number.
        """

    def create(self, nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma, descriptorType) -> retval:
        """
        @brief Create SIFT with specified descriptorType.
        @param nfeatures The number of best features to retain. The features are ranked by their scores (measured in SIFT algorithm as the local contrast) 
        @param nOctaveLayers The number of layers in each octave. 3 is the value used in D. Lowe paper. The number of octaves is computed automatically from the image resolution. 
        @param contrastThreshold The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions. The larger the threshold, the less features are produced by the detector.  @note The contrast threshold will be divided by nOctaveLayers when the filtering is applied. When nOctaveLayers is set to default and if you want to use the value used in D. Lowe paper, 0.03, set this argument to 0.09. 
        @param edgeThreshold The threshold used to filter out edge-like features. Note that the its meaning is different from the contrastThreshold, i.e. the larger the edgeThreshold, the less features are filtered out (more features are retained). 
        @param sigma The sigma of the Gaussian applied to the input image at the octave \#0. If your image is captured with a weak camera with soft lenses, you might want to reduce the number. 
        @param descriptorType The type of descriptors. Only CV_32F and CV_8U are supported.
        """


class SimpleBlobDetector(Feature2D):
    def getBlobContours(self) -> retval:
        """"""

    def getDefaultName(self) -> retval:
        """"""

    def getParams(self) -> retval:
        """"""

    def setParams(self, params) -> None:
        """"""

    def create(self, parameters = ...) -> retval:
        """"""


class Params(builtins.object):
    ...


class SparseOpticalFlow(Algorithm):
    def calc(self, prevImg, nextImg, prevPts, nextPts, status = ..., err = ...) -> tuple[nextPts, status, err]:
        """
        @brief Calculates a sparse optical flow.

        @param prevImg First input image.
        @param nextImg Second input image of the same size and the same type as prevImg.
        @param prevPts Vector of 2D points for which the flow needs to be found.
        @param nextPts Output vector of 2D points containing the calculated new positions of input features in the second image.
        @param status Output status vector. Each element of the vector is set to 1 if the flow for the corresponding features has been found. Otherwise, it is set to 0.
        @param err Optional output vector that contains error response for each point (inverse confidence).
        """


class SparsePyrLKOpticalFlow(SparseOpticalFlow):
    def getFlags(self) -> retval:
        """"""

    def getMaxLevel(self) -> retval:
        """"""

    def getMinEigThreshold(self) -> retval:
        """"""

    def getTermCriteria(self) -> retval:
        """"""

    def getWinSize(self) -> retval:
        """"""

    def setFlags(self, flags) -> None:
        """"""

    def setMaxLevel(self, maxLevel) -> None:
        """"""

    def setMinEigThreshold(self, minEigThreshold) -> None:
        """"""

    def setTermCriteria(self, crit) -> None:
        """"""

    def setWinSize(self, winSize) -> None:
        """"""

    def create(self, winSize = ..., maxLevel = ..., crit = ..., flags = ..., minEigThreshold = ...) -> retval:
        """"""


class StereoBM(StereoMatcher):
    def getPreFilterCap(self) -> retval:
        """"""

    def getPreFilterSize(self) -> retval:
        """"""

    def getPreFilterType(self) -> retval:
        """"""

    def getROI1(self) -> retval:
        """"""

    def getROI2(self) -> retval:
        """"""

    def getSmallerBlockSize(self) -> retval:
        """"""

    def getTextureThreshold(self) -> retval:
        """"""

    def getUniquenessRatio(self) -> retval:
        """"""

    def setPreFilterCap(self, preFilterCap) -> None:
        """"""

    def setPreFilterSize(self, preFilterSize) -> None:
        """"""

    def setPreFilterType(self, preFilterType) -> None:
        """"""

    def setROI1(self, roi1) -> None:
        """"""

    def setROI2(self, roi2) -> None:
        """"""

    def setSmallerBlockSize(self, blockSize) -> None:
        """"""

    def setTextureThreshold(self, textureThreshold) -> None:
        """"""

    def setUniquenessRatio(self, uniquenessRatio) -> None:
        """"""

    def create(self, numDisparities = ..., blockSize = ...) -> retval:
        """
        @brief Creates StereoBM object

        @param numDisparities the disparity search range. For each pixel algorithm will find the best disparity from 0 (default minimum disparity) to numDisparities. The search range can then be shifted by changing the minimum disparity.
        @param blockSize the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, but there is higher chance for algorithm to find a wrong correspondence.  The function create StereoBM object. You can then call StereoBM::compute() to compute disparity for a specific stereo pair.
        """


class StereoMatcher(Algorithm):
    def compute(self, left, right, disparity = ...) -> disparity:
        """
        @brief Computes disparity map for the specified stereo pair

        @param left Left 8-bit single-channel image.
        @param right Right image of the same size and the same type as the left one.
        @param disparity Output disparity map. It has the same size as the input images. Some algorithms, like StereoBM or StereoSGBM compute 16-bit fixed-point disparity map (where each disparity value has 4 fractional bits), whereas other algorithms output 32-bit floating-point disparity map.
        """

    def getBlockSize(self) -> retval:
        """"""

    def getDisp12MaxDiff(self) -> retval:
        """"""

    def getMinDisparity(self) -> retval:
        """"""

    def getNumDisparities(self) -> retval:
        """"""

    def getSpeckleRange(self) -> retval:
        """"""

    def getSpeckleWindowSize(self) -> retval:
        """"""

    def setBlockSize(self, blockSize) -> None:
        """"""

    def setDisp12MaxDiff(self, disp12MaxDiff) -> None:
        """"""

    def setMinDisparity(self, minDisparity) -> None:
        """"""

    def setNumDisparities(self, numDisparities) -> None:
        """"""

    def setSpeckleRange(self, speckleRange) -> None:
        """"""

    def setSpeckleWindowSize(self, speckleWindowSize) -> None:
        """"""


class StereoSGBM(StereoMatcher):
    def getMode(self) -> retval:
        """"""

    def getP1(self) -> retval:
        """"""

    def getP2(self) -> retval:
        """"""

    def getPreFilterCap(self) -> retval:
        """"""

    def getUniquenessRatio(self) -> retval:
        """"""

    def setMode(self, mode) -> None:
        """"""

    def setP1(self, P1) -> None:
        """"""

    def setP2(self, P2) -> None:
        """"""

    def setPreFilterCap(self, preFilterCap) -> None:
        """"""

    def setUniquenessRatio(self, uniquenessRatio) -> None:
        """"""

    def create(self, minDisparity = ..., numDisparities = ..., blockSize = ..., P1 = ..., P2 = ..., disp12MaxDiff = ..., preFilterCap = ..., uniquenessRatio = ..., speckleWindowSize = ..., speckleRange = ..., mode = ...) -> retval:
        """
        @brief Creates StereoSGBM object

        @param minDisparity Minimum possible disparity value. Normally, it is zero but sometimes rectification algorithms can shift images, so this parameter needs to be adjusted accordingly.
        @param numDisparities Maximum disparity minus minimum disparity. The value is always greater than zero. In the current implementation, this parameter must be divisible by 16.
        @param blockSize Matched block size. It must be an odd number \>=1 . Normally, it should be somewhere in the 3..11 range.
        @param P1 The first parameter controlling the disparity smoothness. See below.
        @param P2 The second parameter controlling the disparity smoothness. The larger the values are, the smoother the disparity is. P1 is the penalty on the disparity change by plus or minus 1 between neighbor pixels. P2 is the penalty on the disparity change by more than 1 between neighbor pixels. The algorithm requires P2 \> P1 . See stereo_match.cpp sample where some reasonably good P1 and P2 values are shown (like 8\*number_of_image_channels\*blockSize\*blockSize and 32\*number_of_image_channels\*blockSize\*blockSize , respectively).
        @param disp12MaxDiff Maximum allowed difference (in integer pixel units) in the left-right disparity check. Set it to a non-positive value to disable the check.
        @param preFilterCap Truncation value for the prefiltered image pixels. The algorithm first computes x-derivative at each pixel and clips its value by [-preFilterCap, preFilterCap] interval. The result values are passed to the Birchfield-Tomasi pixel cost function.
        @param uniquenessRatio Margin in percentage by which the best (minimum) computed cost function value should "win" the second best value to consider the found match correct. Normally, a value within the 5-15 range is good enough.
        @param speckleWindowSize Maximum size of smooth disparity regions to consider their noise speckles and invalidate. Set it to 0 to disable speckle filtering. Otherwise, set it somewhere in the 50-200 range.
        @param speckleRange Maximum disparity variation within each connected component. If you do speckle filtering, set the parameter to a positive value, it will be implicitly multiplied by 16. Normally, 1 or 2 is good enough.
        @param mode Set it to StereoSGBM::MODE_HH to run the full-scale two-pass dynamic programming algorithm. It will consume O(W\*H\*numDisparities) bytes, which is large for 640x480 stereo and huge for HD-size pictures. By default, it is set to false .  The first constructor initializes StereoSGBM with all the default parameters. So, you only have to set StereoSGBM::numDisparities at minimum. The second constructor enables you to set each parameter to a custom value.
        """


class Stitcher(builtins.object):
    @overload
    def composePanorama(self, pano = ...) -> tuple[retval, pano]:
        """
        @overload
        """

    def composePanorama(self, images, pano = ...) -> tuple[retval, pano]:
        """
        @brief These functions try to compose the given images (or images stored internally from the other function
        calls) into the final pano under the assumption that the image transformations were estimated
        before.

        @note Use the functions only if you're aware of the stitching pipeline, otherwise use
        Stitcher::stitch.

        @param images Input images.
        @param pano Final pano. @return Status code.
        """

    def compositingResol(self) -> retval:
        """"""

    def estimateTransform(self, images, masks = ...) -> retval:
        """
        @brief These functions try to match the given images and to estimate rotations of each camera.

        @note Use the functions only if you're aware of the stitching pipeline, otherwise use
        Stitcher::stitch.

        @param images Input images.
        @param masks Masks for each input image specifying where to look for keypoints (optional). @return Status code.
        """

    def interpolationFlags(self) -> retval:
        """"""

    def panoConfidenceThresh(self) -> retval:
        """"""

    def registrationResol(self) -> retval:
        """"""

    def seamEstimationResol(self) -> retval:
        """"""

    def setCompositingResol(self, resol_mpx) -> None:
        """"""

    def setInterpolationFlags(self, interp_flags) -> None:
        """"""

    def setPanoConfidenceThresh(self, conf_thresh) -> None:
        """"""

    def setRegistrationResol(self, resol_mpx) -> None:
        """"""

    def setSeamEstimationResol(self, resol_mpx) -> None:
        """"""

    def setWaveCorrection(self, flag) -> None:
        """"""

    @overload
    def stitch(self, images, pano = ...) -> tuple[retval, pano]:
        """
        @overload
        """

    def stitch(self, images, masks, pano = ...) -> tuple[retval, pano]:
        """
        @brief These functions try to stitch the given images.

        @param images Input images.
        @param masks Masks for each input image specifying where to look for keypoints (optional).
        @param pano Final pano. @return Status code.
        """

    def waveCorrection(self) -> retval:
        """"""

    def workScale(self) -> retval:
        """"""

    def create(self, mode = ...) -> retval:
        """
        @brief Creates a Stitcher configured in one of the stitching modes.

        @param mode Scenario for stitcher operation. This is usually determined by source of images to stitch and their transformation. Default parameters will be chosen for operation in given scenario. @return Stitcher class instance.
        """


class Subdiv2D(builtins.object):
    def edgeDst(self, edge) -> tuple[retval, dstpt]:
        """
        @brief Returns the edge destination.

        @param edge Subdivision edge ID.
        @param dstpt Output vertex location.  @returns vertex ID.
        """

    def edgeOrg(self, edge) -> tuple[retval, orgpt]:
        """
        @brief Returns the edge origin.

        @param edge Subdivision edge ID.
        @param orgpt Output vertex location.  @returns vertex ID.
        """

    def findNearest(self, pt) -> tuple[retval, nearestPt]:
        """
        @brief Finds the subdivision vertex closest to the given point.

        @param pt Input point.
        @param nearestPt Output subdivision vertex point.  The function is another function that locates the input point within the subdivision. It finds the subdivision vertex that is the closest to the input point. It is not necessarily one of vertices of the facet containing the input point, though the facet (located using locate() ) is used as a starting point.  @returns vertex ID.
        """

    def getEdge(self, edge, nextEdgeType) -> retval:
        """
        @brief Returns one of the edges related to the given edge.

        @param edge Subdivision edge ID.
        @param nextEdgeType Parameter specifying which of the related edges to return. The following values are possible: -   NEXT_AROUND_ORG next around the edge origin ( eOnext on the picture below if e is the input edge) -   NEXT_AROUND_DST next around the edge vertex ( eDnext ) -   PREV_AROUND_ORG previous around the edge origin (reversed eRnext ) -   PREV_AROUND_DST previous around the edge destination (reversed eLnext ) -   NEXT_AROUND_LEFT next around the left facet ( eLnext ) -   NEXT_AROUND_RIGHT next around the right facet ( eRnext ) -   PREV_AROUND_LEFT previous around the left facet (reversed eOnext ) -   PREV_AROUND_RIGHT previous around the right facet (reversed eDnext )  ![sample output](pics/quadedge.png)  @returns edge ID related to the input edge.
        """

    def getEdgeList(self) -> edgeList:
        """
        @brief Returns a list of all edges.

        @param edgeList Output vector.  The function gives each edge as a 4 numbers vector, where each two are one of the edge vertices. i.e. org_x = v[0], org_y = v[1], dst_x = v[2], dst_y = v[3].
        """

    def getLeadingEdgeList(self) -> leadingEdgeList:
        """
        @brief Returns a list of the leading edge ID connected to each triangle.

        @param leadingEdgeList Output vector.  The function gives one edge ID for each triangle.
        """

    def getTriangleList(self) -> triangleList:
        """
        @brief Returns a list of all triangles.

        @param triangleList Output vector.  The function gives each triangle as a 6 numbers vector, where each two are one of the triangle vertices. i.e. p1_x = v[0], p1_y = v[1], p2_x = v[2], p2_y = v[3], p3_x = v[4], p3_y = v[5].
        """

    def getVertex(self, vertex) -> tuple[retval, firstEdge]:
        """
        @brief Returns vertex location from vertex ID.

        @param vertex vertex ID.
        @param firstEdge Optional. The first edge ID which is connected to the vertex. @returns vertex (x,y)
        """

    def getVoronoiFacetList(self, idx) -> tuple[facetList, facetCenters]:
        """
        @brief Returns a list of all Voronoi facets.

        @param idx Vector of vertices IDs to consider. For all vertices you can pass empty vector.
        @param facetList Output vector of the Voronoi facets.
        @param facetCenters Output vector of the Voronoi facets center points.
        """

    def initDelaunay(self, rect) -> None:
        """
        @brief Creates a new empty Delaunay subdivision

        @param rect Rectangle that includes all of the 2D points that are to be added to the subdivision.
        """

    def insert(self, pt) -> retval:
        """
        @brief Insert a single point into a Delaunay triangulation.

        @param pt Point to insert.  The function inserts a single point into a subdivision and modifies the subdivision topology appropriately. If a point with the same coordinates exists already, no new point is added. @returns the ID of the point.  @note If the point is outside of the triangulation specified rect a runtime error is raised.
        """

    def insert(self, ptvec) -> None:
        """
        @brief Insert multiple points into a Delaunay triangulation.

        @param ptvec Points to insert.  The function inserts a vector of points into a subdivision and modifies the subdivision topology appropriately.
        """

    def locate(self, pt) -> tuple[retval, edge, vertex]:
        """
        @brief Returns the location of a point within a Delaunay triangulation.

        @param pt Point to locate.
        @param edge Output edge that the point belongs to or is located to the right of it.
        @param vertex Optional output vertex the input point coincides with.  The function locates the input point within the subdivision and gives one of the triangle edges or vertices.  @returns an integer which specify one of the following five cases for point location: -  The point falls into some facet. The function returns #PTLOC_INSIDE and edge will contain one of edges of the facet. -  The point falls onto the edge. The function returns #PTLOC_ON_EDGE and edge will contain this edge. -  The point coincides with one of the subdivision vertices. The function returns #PTLOC_VERTEX and vertex will contain a pointer to the vertex. -  The point is outside the subdivision reference rectangle. The function returns #PTLOC_OUTSIDE_RECT and no pointers are filled. -  One of input arguments is invalid. A runtime error is raised or, if silent or "parent" error processing mode is selected, #PTLOC_ERROR is returned.
        """

    def nextEdge(self, edge) -> retval:
        """
        @brief Returns next edge around the edge origin.

        @param edge Subdivision edge ID.  @returns an integer which is next edge ID around the edge origin: eOnext on the picture above if e is the input edge).
        """

    def rotateEdge(self, edge, rotate) -> retval:
        """
        @brief Returns another edge of the same quad-edge.

        @param edge Subdivision edge ID.
        @param rotate Parameter specifying which of the edges of the same quad-edge as the input one to return. The following values are possible: -   0 - the input edge ( e on the picture below if e is the input edge) -   1 - the rotated edge ( eRot ) -   2 - the reversed edge (reversed e (in green)) -   3 - the reversed rotated edge (reversed eRot (in green))  @returns one of the edges ID of the same quad-edge as the input edge.
        """

    def symEdge(self, edge) -> retval:
        """"""


class TickMeter(builtins.object):
    def getAvgTimeMilli(self) -> retval:
        """"""

    def getAvgTimeSec(self) -> retval:
        """"""

    def getCounter(self) -> retval:
        """"""

    def getFPS(self) -> retval:
        """"""

    def getTimeMicro(self) -> retval:
        """"""

    def getTimeMilli(self) -> retval:
        """"""

    def getTimeSec(self) -> retval:
        """"""

    def getTimeTicks(self) -> retval:
        """"""

    def reset(self) -> None:
        """"""

    def start(self) -> None:
        """"""

    def stop(self) -> None:
        """"""


class Tonemap(Algorithm):
    def getGamma(self) -> retval:
        """"""

    def process(self, src, dst = ...) -> dst:
        """
        @brief Tonemaps image

        @param src source image - CV_32FC3 Mat (float 32 bits 3 channels)
        @param dst destination image - CV_32FC3 Mat with values in [0, 1] range
        """

    def setGamma(self, gamma) -> None:
        """"""


class TonemapDrago(Tonemap):
    def getBias(self) -> retval:
        """"""

    def getSaturation(self) -> retval:
        """"""

    def setBias(self, bias) -> None:
        """"""

    def setSaturation(self, saturation) -> None:
        """"""


class TonemapMantiuk(Tonemap):
    def getSaturation(self) -> retval:
        """"""

    def getScale(self) -> retval:
        """"""

    def setSaturation(self, saturation) -> None:
        """"""

    def setScale(self, scale) -> None:
        """"""


class TonemapReinhard(Tonemap):
    def getColorAdaptation(self) -> retval:
        """"""

    def getIntensity(self) -> retval:
        """"""

    def getLightAdaptation(self) -> retval:
        """"""

    def setColorAdaptation(self, color_adapt) -> None:
        """"""

    def setIntensity(self, intensity) -> None:
        """"""

    def setLightAdaptation(self, light_adapt) -> None:
        """"""


class Tracker(builtins.object):
    def init(self, image, boundingBox) -> None:
        """
        @brief Initialize the tracker with a known bounding box that surrounded the target
        @param image The initial frame
        @param boundingBox The initial bounding box
        """

    def update(self, image) -> tuple[retval, boundingBox]:
        """
        @brief Update the tracker, find the new most likely bounding box for the target
        @param image The current frame
        @param boundingBox The bounding box that represent the new target location, if true was returned, not modified otherwise  @return True means that target was located and false means that tracker cannot locate target in current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight)
        """


class TrackerDaSiamRPN(Tracker):
    def getTrackingScore(self) -> retval:
        """
        @brief Return tracking score
        """

    def create(self, parameters = ...) -> retval:
        """
        @brief Constructor
        @param parameters DaSiamRPN parameters TrackerDaSiamRPN::Params
        """


class Params(builtins.object):
    ...


class TrackerGOTURN(Tracker):
    def create(self, parameters = ...) -> retval:
        """
        @brief Constructor
        @param parameters GOTURN parameters TrackerGOTURN::Params
        """


class Params(builtins.object):
    ...


class TrackerMIL(Tracker):
    def create(self, parameters = ...) -> retval:
        """
        @brief Create MIL tracker instance
        *  @param parameters MIL parameters TrackerMIL::Params
        """


class Params(builtins.object):
    ...


class TrackerNano(Tracker):
    def getTrackingScore(self) -> retval:
        """
        @brief Return tracking score
        """

    def create(self, parameters = ...) -> retval:
        """
        @brief Constructor
        @param parameters NanoTrack parameters TrackerNano::Params
        """


class Params(builtins.object):
    ...


class UMat(builtins.object):
    def get(self) -> retval:
        """"""

    def handle(self, accessFlags) -> retval:
        """"""

    def isContinuous(self) -> retval:
        """"""

    def isSubmatrix(self) -> retval:
        """"""

    def context(self) -> retval:
        """"""

    def queue(self) -> retval:
        """"""


class UsacParams(builtins.object):
    ...


class VariationalRefinement(DenseOpticalFlow):
    def calcUV(self, I0, I1, flow_u, flow_v) -> tuple[flow_u, flow_v]:
        """
        @brief @ref calc function overload to handle separate horizontal (u) and vertical (v) flow components
        (to avoid extra splits/merges)
        """

    def getAlpha(self) -> retval:
        """
        @brief Weight of the smoothness term
        @see setAlpha
        """

    def getDelta(self) -> retval:
        """
        @brief Weight of the color constancy term
        @see setDelta
        """

    def getFixedPointIterations(self) -> retval:
        """
        @brief Number of outer (fixed-point) iterations in the minimization procedure.
        @see setFixedPointIterations
        """

    def getGamma(self) -> retval:
        """
        @brief Weight of the gradient constancy term
        @see setGamma
        """

    def getOmega(self) -> retval:
        """
        @brief Relaxation factor in SOR
        @see setOmega
        """

    def getSorIterations(self) -> retval:
        """
        @brief Number of inner successive over-relaxation (SOR) iterations
        in the minimization procedure to solve the respective linear system.
        @see setSorIterations
        """

    def setAlpha(self, val) -> None:
        """
        @copybrief getAlpha @see getAlpha
        """

    def setDelta(self, val) -> None:
        """
        @copybrief getDelta @see getDelta
        """

    def setFixedPointIterations(self, val) -> None:
        """
        @copybrief getFixedPointIterations @see getFixedPointIterations
        """

    def setGamma(self, val) -> None:
        """
        @copybrief getGamma @see getGamma
        """

    def setOmega(self, val) -> None:
        """
        @copybrief getOmega @see getOmega
        """

    def setSorIterations(self, val) -> None:
        """
        @copybrief getSorIterations @see getSorIterations
        """

    def create(self) -> retval:
        """
        @brief Creates an instance of VariationalRefinement
        """


class VideoCapture(builtins.object):
    def get(self, propId) -> retval:
        """
        @brief Returns the specified VideoCapture property

        @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) or one from @ref videoio_flags_others @return Value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoCapture instance.  @note Reading / writing properties involves many layers. Some unexpected result might happens along this chain. @code{.txt} VideoCapture -> API Backend -> Operating System -> Device Driver -> Device Hardware @endcode The returned value might be different from what really used by the device or it could be encoded using device dependent rules (eg. steps or percentage). Effective behaviour depends from device driver and API Backend
        """

    def getBackendName(self) -> retval:
        """
        @brief Returns used backend API name

        @note Stream should be opened.
        """

    def getExceptionMode(self) -> retval:
        """"""

    def grab(self) -> retval:
        """
        @brief Grabs the next frame from video file or capturing device.

        @return `true` (non-zero) in the case of success.

        The method/function grabs the next frame from video file or camera and returns true (non-zero) in
        the case of success.

        The primary use of the function is in multi-camera environments, especially when the cameras do not
        have hardware synchronization. That is, you call VideoCapture::grab() for each camera and after that
        call the slower method VideoCapture::retrieve() to decode and get frame from each camera. This way
        the overhead on demosaicing or motion jpeg decompression etc. is eliminated and the retrieved frames
        from different cameras will be closer in time.

        Also, when a connected camera is multi-head (for example, a stereo camera or a Kinect device), the
        correct way of retrieving data from it is to call VideoCapture::grab() first and then call
        VideoCapture::retrieve() one or more times with different values of the channel parameter.

        @ref tutorial_kinect_openni
        """

    def isOpened(self) -> retval:
        """
        @brief Returns true if video capturing has been initialized already.

        If the previous call to VideoCapture constructor or VideoCapture::open() succeeded, the method returns
        true.
        """

    def open(self, filename, apiPreference = ...) -> retval:
        """
    @overload
        @brief  Opens a video file or a capturing device or an IP video stream for video capturing.

        @overload

        Parameters are same as the constructor VideoCapture(const String& filename, int apiPreference = CAP_ANY)
        @return `true` if the file has been successfully opened

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    def open(self, filename, apiPreference, params) -> retval:
        """
    @overload
        @brief  Opens a video file or a capturing device or an IP video stream for video capturing with API Preference and parameters

        @overload

        The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`.
        See cv::VideoCaptureProperties

        @return `true` if the file has been successfully opened

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    def open(self, index, apiPreference = ...) -> retval:
        """
    @overload
        @brief  Opens a camera for video capturing

        @overload

        Parameters are same as the constructor VideoCapture(int index, int apiPreference = CAP_ANY)
        @return `true` if the camera has been successfully opened.

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    def open(self, index, apiPreference, params) -> retval:
        """
    @overload
        @brief  Opens a camera for video capturing with API Preference and parameters

        @overload

        The `params` parameter allows to specify extra parameters encoded as pairs `(paramId_1, paramValue_1, paramId_2, paramValue_2, ...)`.
        See cv::VideoCaptureProperties

        @return `true` if the camera has been successfully opened.

        The method first calls VideoCapture::release to close the already opened file or camera.
        """

    def read(self, image = ...) -> tuple[retval, image]:
        """
        @brief Grabs, decodes and returns the next video frame.

        @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty. @return `false` if no frames has been grabbed  The method/function combines VideoCapture::grab() and VideoCapture::retrieve() in one call. This is the most convenient method for reading video files or capturing data from decode and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns empty image (with %cv::Mat, test it with Mat::empty()).  @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy.
        """

    def release(self) -> None:
        """
        @brief Closes video file or capturing device.

        The method is automatically called by subsequent VideoCapture::open and by VideoCapture
        destructor.

        The C function also deallocates memory and clears \*capture pointer.
        """

    def retrieve(self, image = ..., flag = ...) -> tuple[retval, image]:
        """
        @brief Decodes and returns the grabbed video frame.

        @param [out] image the video frame is returned here. If no frames has been grabbed the image will be empty.
        @param flag it could be a frame index or a driver specific flag @return `false` if no frames has been grabbed  The method decodes and returns the just grabbed frame. If no frames has been grabbed (camera has been disconnected, or there are no more frames in video file), the method returns false and the function returns an empty image (with %cv::Mat, test it with Mat::empty()).  @sa read()  @note In @ref videoio_c "C API", functions cvRetrieveFrame() and cv.RetrieveFrame() return image stored inside the video capturing structure. It is not allowed to modify or release the image! You can copy the frame using cvCloneImage and then do whatever you want with the copy.
        """

    def set(self, propId, value) -> retval:
        """
        @brief Sets a property in the VideoCapture.

        @param propId Property identifier from cv::VideoCaptureProperties (eg. cv::CAP_PROP_POS_MSEC, cv::CAP_PROP_POS_FRAMES, ...) or one from @ref videoio_flags_others
        @param value Value of the property. @return `true` if the property is supported by backend used by the VideoCapture instance. @note Even if it returns `true` this doesn't ensure that the property value has been accepted by the capture device. See note in VideoCapture::get()
        """

    def setExceptionMode(self, enable) -> None:
        """
        Switches exceptions mode
        *
        * methods raise exceptions if not successful instead of returning an error code
        """

    def waitAny(self, streams, timeoutNs = ...) -> tuple[retval, readyIndex]:
        """
        @brief Wait for ready frames from VideoCapture.

        @param streams input video streams
        @param readyIndex stream indexes with grabbed frames (ready to use .retrieve() to fetch actual frame)
        @param timeoutNs number of nanoseconds (0 - infinite) @return `true` if streamReady is not empty  @throws Exception %Exception on stream errors (check .isOpened() to filter out malformed streams) or VideoCapture type is not supported  The primary use of the function is in multi-camera environments. The method fills the ready state vector, grabs video frame, if camera is ready.  After this call use VideoCapture::retrieve() to decode and fetch frame data.
        """


class VideoWriter(builtins.object):
    def get(self, propId) -> retval:
        """
        @brief Returns the specified VideoWriter property

        @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY) or one of @ref videoio_flags_others  @return Value for the specified property. Value 0 is returned when querying a property that is not supported by the backend used by the VideoWriter instance.
        """

    def getBackendName(self) -> retval:
        """
        @brief Returns used backend API name

        @note Stream should be opened.
        """

    def isOpened(self) -> retval:
        """
        @brief Returns true if video writer has been successfully initialized.
        """

    def open(self, filename, fourcc, fps, frameSize, isColor = ...) -> retval:
        """
        @brief Initializes or reinitializes video writer.

        The method opens video writer. Parameters are the same as in the constructor
        VideoWriter::VideoWriter.
        @return `true` if video writer has been successfully initialized

        The method first calls VideoWriter::release to close the already opened file.
        """

    @overload
    def open(self, filename, apiPreference, fourcc, fps, frameSize, isColor = ...) -> retval:
        """
        @overload
        """

    @overload
    def open(self, filename, fourcc, fps, frameSize, params) -> retval:
        """
        @overload
        """

    @overload
    def open(self, filename, apiPreference, fourcc, fps, frameSize, params) -> retval:
        """
        @overload
        """

    def release(self) -> None:
        """
        @brief Closes the video writer.

        The method is automatically called by subsequent VideoWriter::open and by the VideoWriter
        destructor.
        """

    def set(self, propId, value) -> retval:
        """
        @brief Sets a property in the VideoWriter.

        @param propId Property identifier from cv::VideoWriterProperties (eg. cv::VIDEOWRITER_PROP_QUALITY) or one of @ref videoio_flags_others 
        @param value Value of the property. @return  `true` if the property is supported by the backend used by the VideoWriter instance.
        """

    def write(self, image) -> None:
        """
        @brief Writes the next video frame

        @param image The written frame. In general, color images are expected in BGR format.  The function/method writes the specified image to video file. It must have the same size as has been specified when opening the video writer.
        """

    def fourcc(self, c1, c2, c3, c4) -> retval:
        """
        @brief Concatenates 4 chars to a fourcc code

        @return a fourcc code

        This static method constructs the fourcc code of the codec to be used in the constructor
        VideoWriter::VideoWriter or VideoWriter::open.
        """


class WarperCreator(builtins.object):
    ...


class ArucoDetector(cv2.Algorithm):
    def detectMarkers(self, image, corners = ..., ids = ..., rejectedImgPoints = ...) -> tuple[corners, ids, rejectedImgPoints]:
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

    def refineDetectedMarkers(self, image, board, detectedCorners, detectedIds, rejectedCorners, cameraMatrix = ..., distCoeffs = ..., recoveredIdxs = ...) -> tuple[detectedCorners, detectedIds, rejectedCorners, recoveredIdxs]:
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
    def generateImage(self, outSize, img = ..., marginSize = ..., borderBits = ...) -> img:
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

    def matchImagePoints(self, detectedCorners, detectedIds, objPoints = ..., imgPoints = ...) -> tuple[objPoints, imgPoints]:
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
    def detectBoard(self, image, charucoCorners = ..., charucoIds = ..., markerCorners = ..., markerIds = ...) -> tuple[charucoCorners, charucoIds, markerCorners, markerIds]:
        """
        * @brief detect aruco markers and interpolate position of ChArUco board corners
        * @param image input image necesary for corner refinement. Note that markers are not detected and * should be sent in corners and ids parameters.
        * @param charucoCorners interpolated chessboard corners.
        * @param charucoIds interpolated chessboard corners identifiers.
        * @param markerCorners vector of already detected markers corners. For each marker, its four * corners are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the * dimensions of this array should be Nx4. The order of the corners should be clockwise. * If markerCorners and markerCorners are empty, the function detect aruco markers and ids.
        * @param markerIds list of identifiers for each marker in corners. *  If markerCorners and markerCorners are empty, the function detect aruco markers and ids. * * This function receives the detected markers and returns the 2D position of the chessboard corners * from a ChArUco board using the detected Aruco markers. * * If markerCorners and markerCorners are empty, the detectMarkers() will run and detect aruco markers and ids. * * If camera parameters are provided, the process is based in an approximated pose estimation, else it is based on local homography. * Only visible corners are returned. For each corner, its corresponding identifier is also returned in charucoIds. * @sa findChessboardCorners
        """

    def detectDiamonds(self, image, diamondCorners = ..., diamondIds = ..., markerCorners = ..., markerIds = ...) -> tuple[diamondCorners, diamondIds, markerCorners, markerIds]:
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


class CharucoParameters(builtins.object):
    ...


class DetectorParameters(builtins.object):
    def readDetectorParameters(self, fn) -> retval:
        """
        @brief Read a new set of DetectorParameters from FileNode (use FileStorage.root()).
        """

    def writeDetectorParameters(self, fs, name = ...) -> retval:
        """
        @brief Write a set of DetectorParameters to FileStorage
        """


class Dictionary(builtins.object):
    def generateImageMarker(self, id, sidePixels, _img = ..., borderBits = ...) -> _img:
        """
        @brief Generate a canonical marker image
        """

    def getDistanceToId(self, bits, id, allRotations = ...) -> retval:
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

    def writeDictionary(self, fs, name = ...) -> None:
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

    def writeRefineParameters(self, fs, name = ...) -> retval:
        """
        @brief Write a set of RefineParameters to FileStorage
        """


class BufferPool(builtins.object):
    def getAllocator(self) -> retval:
        """"""

    def getBuffer(self, rows, cols, type) -> retval:
        """"""

    def getBuffer(self, size, type) -> retval:
        """"""


class DeviceInfo(builtins.object):
    def ECCEnabled(self) -> retval:
        """

        See help(type(self)) for accurate signature.

        """

    def asyncEngineCount(self) -> retval:
        """"""

    def canMapHostMemory(self) -> retval:
        """"""

    def clockRate(self) -> retval:
        """"""

    def computeMode(self) -> retval:
        """"""

    def concurrentKernels(self) -> retval:
        """"""

    def deviceID(self) -> retval:
        """
        @brief Returns system index of the CUDA device starting with 0.
        """

    def freeMemory(self) -> retval:
        """"""

    def integrated(self) -> retval:
        """"""

    def isCompatible(self) -> retval:
        """
        @brief Checks the CUDA module and device compatibility.

        This function returns true if the CUDA module can be run on the specified device. Otherwise, it
        returns false .
        """

    def kernelExecTimeoutEnabled(self) -> retval:
        """"""

    def l2CacheSize(self) -> retval:
        """"""

    def majorVersion(self) -> retval:
        """"""

    def maxGridSize(self) -> retval:
        """"""

    def maxSurface1D(self) -> retval:
        """"""

    def maxSurface1DLayered(self) -> retval:
        """"""

    def maxSurface2D(self) -> retval:
        """"""

    def maxSurface2DLayered(self) -> retval:
        """"""

    def maxSurface3D(self) -> retval:
        """"""

    def maxSurfaceCubemap(self) -> retval:
        """"""

    def maxSurfaceCubemapLayered(self) -> retval:
        """"""

    def maxTexture1D(self) -> retval:
        """"""

    def maxTexture1DLayered(self) -> retval:
        """"""

    def maxTexture1DLinear(self) -> retval:
        """"""

    def maxTexture1DMipmap(self) -> retval:
        """"""

    def maxTexture2D(self) -> retval:
        """"""

    def maxTexture2DGather(self) -> retval:
        """"""

    def maxTexture2DLayered(self) -> retval:
        """"""

    def maxTexture2DLinear(self) -> retval:
        """"""

    def maxTexture2DMipmap(self) -> retval:
        """"""

    def maxTexture3D(self) -> retval:
        """"""

    def maxTextureCubemap(self) -> retval:
        """"""

    def maxTextureCubemapLayered(self) -> retval:
        """"""

    def maxThreadsDim(self) -> retval:
        """"""

    def maxThreadsPerBlock(self) -> retval:
        """"""

    def maxThreadsPerMultiProcessor(self) -> retval:
        """"""

    def memPitch(self) -> retval:
        """"""

    def memoryBusWidth(self) -> retval:
        """"""

    def memoryClockRate(self) -> retval:
        """"""

    def minorVersion(self) -> retval:
        """"""

    def multiProcessorCount(self) -> retval:
        """"""

    def pciBusID(self) -> retval:
        """"""

    def pciDeviceID(self) -> retval:
        """"""

    def pciDomainID(self) -> retval:
        """"""

    def queryMemory(self, totalMemory, freeMemory) -> None:
        """"""

    def regsPerBlock(self) -> retval:
        """"""

    def sharedMemPerBlock(self) -> retval:
        """"""

    def surfaceAlignment(self) -> retval:
        """"""

    def tccDriver(self) -> retval:
        """"""

    def textureAlignment(self) -> retval:
        """"""

    def texturePitchAlignment(self) -> retval:
        """"""

    def totalConstMem(self) -> retval:
        """"""

    def totalGlobalMem(self) -> retval:
        """"""

    def totalMemory(self) -> retval:
        """"""

    def unifiedAddressing(self) -> retval:
        """"""

    def warpSize(self) -> retval:
        """"""


class Event(builtins.object):
    def queryIfComplete(self) -> retval:
        """"""

    def record(self, stream = ...) -> None:
        """"""

    def waitForCompletion(self) -> None:
        """"""

    def elapsedTime(self, start, end) -> retval:
        """"""


class GpuData(builtins.object):
    ...


class GpuMat(builtins.object):
    def adjustROI(self, dtop, dbottom, dleft, dright) -> retval:
        """"""

    def assignTo(self, m, type = ...) -> None:
        """"""

    def channels(self) -> retval:
        """"""

    def clone(self) -> retval:
        """"""

    def col(self, x) -> retval:
        """"""

    def colRange(self, startcol, endcol) -> retval:
        """"""

    def colRange(self, r) -> retval:
        """"""

    def convertTo(self, rtype, dst = ...) -> dst:
        """"""

    def convertTo(self, rtype, stream, dst = ...) -> dst:
        """"""

    def convertTo(self, rtype, alpha, dst = ..., beta = ...) -> dst:
        """"""

    def convertTo(self, rtype, alpha, stream, dst = ...) -> dst:
        """"""

    def convertTo(self, rtype, alpha, beta, stream, dst = ...) -> dst:
        """"""

    def copyTo(self, dst = ...) -> dst:
        """"""

    def copyTo(self, stream, dst = ...) -> dst:
        """"""

    def copyTo(self, mask, dst = ...) -> dst:
        """"""

    def copyTo(self, mask, stream, dst = ...) -> dst:
        """"""

    def create(self, rows, cols, type) -> None:
        """"""

    def create(self, size, type) -> None:
        """"""

    def cudaPtr(self) -> retval:
        """"""

    def depth(self) -> retval:
        """"""

    def download(self, dst = ...) -> dst:
        """
        @brief Performs data download from GpuMat (Blocking call)

        This function copies data from device memory to host memory. As being a blocking call, it is
        guaranteed that the copy operation is finished when this function returns.
        """

    def download(self, stream, dst = ...) -> dst:
        """
        @brief Performs data download from GpuMat (Non-Blocking call)

        This function copies data from device memory to host memory. As being a non-blocking call, this
        function may return even if the copy operation is not finished.

        The copy operation may be overlapped with operations in other non-default streams if \p stream is
        not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
        """

    def elemSize(self) -> retval:
        """"""

    def elemSize1(self) -> retval:
        """"""

    def empty(self) -> retval:
        """"""

    def isContinuous(self) -> retval:
        """"""

    def locateROI(self, wholeSize, ofs) -> None:
        """"""

    def release(self) -> None:
        """"""

    def reshape(self, cn, rows = ...) -> retval:
        """"""

    def row(self, y) -> retval:
        """"""

    def rowRange(self, startrow, endrow) -> retval:
        """"""

    def rowRange(self, r) -> retval:
        """"""

    def setTo(self, s) -> retval:
        """"""

    def setTo(self, s, stream) -> retval:
        """"""

    def setTo(self, s, mask) -> retval:
        """"""

    def setTo(self, s, mask, stream) -> retval:
        """"""

    def size(self) -> retval:
        """"""

    def step1(self) -> retval:
        """"""

    def swap(self, mat) -> None:
        """"""

    def type(self) -> retval:
        """"""

    def updateContinuityFlag(self) -> None:
        """"""

    def upload(self, arr) -> None:
        """
        @brief Performs data upload to GpuMat (Blocking call)

        This function copies data from host memory to device memory. As being a blocking call, it is
        guaranteed that the copy operation is finished when this function returns.
        """

    def upload(self, arr, stream) -> None:
        """
        @brief Performs data upload to GpuMat (Non-Blocking call)

        This function copies data from host memory to device memory. As being a non-blocking call, this
        function may return even if the copy operation is not finished.

        The copy operation may be overlapped with operations in other non-default streams if \p stream is
        not the default stream and \p dst is HostMem allocated with HostMem::PAGE_LOCKED option.
        """

    def defaultAllocator(self) -> retval:
        """"""

    def setDefaultAllocator(self, allocator) -> None:
        """"""


class GpuMatND(builtins.object):
    ...


class Allocator(builtins.object):
    ...


class HostMem(builtins.object):
    def channels(self) -> retval:
        """"""

    def clone(self) -> retval:
        """"""

    def create(self, rows, cols, type) -> None:
        """"""

    def createMatHeader(self) -> retval:
        """"""

    def depth(self) -> retval:
        """"""

    def elemSize(self) -> retval:
        """"""

    def elemSize1(self) -> retval:
        """"""

    def empty(self) -> retval:
        """"""

    def isContinuous(self) -> retval:
        """
        @brief Maps CPU memory to GPU address space and creates the cuda::GpuMat header without reference counting
        for it.

        This can be done only if memory was allocated with the SHARED flag and if it is supported by the
        hardware. Laptops often share video and CPU memory, so address spaces can be mapped, which
        eliminates an extra copy.
        """

    def reshape(self, cn, rows = ...) -> retval:
        """"""

    def size(self) -> retval:
        """"""

    def step1(self) -> retval:
        """"""

    def swap(self, b) -> None:
        """"""

    def type(self) -> retval:
        """"""


class Stream(builtins.object):
    def cudaPtr(self) -> retval:
        """"""

    def queryIfComplete(self) -> retval:
        """
        @brief Returns true if the current stream queue is finished. Otherwise, it returns false.
        """

    def waitEvent(self, event) -> None:
        """
        @brief Makes a compute stream wait on an event.
        """

    def waitForCompletion(self) -> None:
        """
        @brief Blocks the current CPU thread until all operations in the stream are complete.
        """

    def Null(self) -> retval:
        """
        @brief Adds a callback to be called on the host after all currently enqueued items in the stream have
        completed.

        @note Callbacks must not make any CUDA API calls. Callbacks must not perform any synchronization
        that may depend on outstanding device work or other callbacks that are not mandated to run earlier.
        Callbacks without a mandated order (in independent streams) execute in undefined order and may be
        serialized.
        type
        See help(type) for accurate signature.
        """


class TargetArchs(builtins.object):
    def has(self, major, minor) -> retval:
        """
        @brief There is a set of methods to check whether the module contains intermediate (PTX) or binary CUDA
        code for the given architecture(s):

        @param major Major compute capability version.
        @param minor Minor compute capability version.
        """

    def hasBin(self, major, minor) -> retval:
        """"""

    def hasEqualOrGreater(self, major, minor) -> retval:
        """"""

    def hasEqualOrGreaterBin(self, major, minor) -> retval:
        """"""

    def hasEqualOrGreaterPtx(self, major, minor) -> retval:
        """"""

    def hasEqualOrLessPtx(self, major, minor) -> retval:
        """"""

    def hasPtx(self, major, minor) -> retval:
        """"""


class AffineBasedEstimator(Estimator):
    ...


class AffineBestOf2NearestMatcher(BestOf2NearestMatcher):
    ...


class BestOf2NearestMatcher(FeaturesMatcher):
    def collectGarbage(self) -> None:
        """"""

    def create(self, try_use_gpu = ..., match_conf = ..., num_matches_thresh1 = ..., num_matches_thresh2 = ..., matches_confindece_thresh = ...) -> retval:
        """"""


class BestOf2NearestRangeMatcher(BestOf2NearestMatcher):
    ...


class Blender(builtins.object):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """
        @brief Blends and returns the final pano.

        @param dst Final pano
        @param dst_mask Final pano mask
        """

    def feed(self, img, mask, tl) -> None:
        """
        @brief Processes the image.

        @param img Source image
        @param mask Source image mask
        @param tl Source image top-left corners
        """

    def prepare(self, corners, sizes) -> None:
        """
        @brief Prepares the blender for blending.

        @param corners Source images top-left corners
        @param sizes Source image sizes
        """

    @overload
    def prepare(self, dst_roi) -> None:
        """
        @overload
        """

    def createDefault(self, type, try_gpu = ...) -> retval:
        """"""


class BlocksChannelsCompensator(BlocksCompensator):
    ...


class BlocksCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getBlockSize(self) -> retval:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getNrGainsFilteringIterations(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setBlockSize(self, width, height) -> None:
        """"""

    def setBlockSize(self, size) -> None:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setNrGainsFilteringIterations(self, nr_iterations) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class BlocksGainCompensator(BlocksCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""


class BundleAdjusterAffine(BundleAdjusterBase):
    ...


class BundleAdjusterAffinePartial(BundleAdjusterBase):
    ...


class BundleAdjusterBase(Estimator):
    def confThresh(self) -> retval:
        """"""

    def refinementMask(self) -> retval:
        """"""

    def setConfThresh(self, conf_thresh) -> None:
        """"""

    def setRefinementMask(self, mask) -> None:
        """"""

    def setTermCriteria(self, term_criteria) -> None:
        """"""

    def termCriteria(self) -> retval:
        """"""


class BundleAdjusterRay(BundleAdjusterBase):
    ...


class BundleAdjusterReproj(BundleAdjusterBase):
    ...


class CameraParams(builtins.object):
    def K(self) -> retval:
        """

        
        """


class ChannelsCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class DpSeamFinder(SeamFinder):
    def setCostFunction(self, val) -> None:
        """"""


class Estimator(builtins.object):
    def apply(self, features, pairwise_matches, cameras) -> tuple[retval, cameras]:
        """
        @brief Estimates camera parameters.

        @param features Features of images
        @param pairwise_matches Pairwise matches of images
        @param cameras Estimated camera parameters @return True in case of success, false otherwise
        """


class ExposureCompensator(builtins.object):
    def apply(self, index, corner, image, mask) -> image:
        """
        @brief Compensate exposure in the specified image.

        @param index Image index
        @param corner Image top-left corner
        @param image Image to process
        @param mask Image mask
        """

    def feed(self, corners, images, masks) -> None:
        """
        @param corners Source image top-left corners
        @param images Source images
        @param masks Image masks to update (second value in pair specifies the value which should be used to detect where image is)
        """

    def getMatGains(self, arg1 = ...) -> arg1:
        """"""

    def getUpdateGain(self) -> retval:
        """"""

    def setMatGains(self, arg1) -> None:
        """"""

    def setUpdateGain(self, b) -> None:
        """"""

    def createDefault(self, type) -> retval:
        """"""


class FeatherBlender(Blender):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """"""

    def createWeightMaps(self, masks, corners, weight_maps) -> tuple[retval, weight_maps]:
        """"""

    def feed(self, img, mask, tl) -> None:
        """"""

    def prepare(self, dst_roi) -> None:
        """"""

    def setSharpness(self, val) -> None:
        """"""

    def sharpness(self) -> retval:
        """"""


class FeaturesMatcher(builtins.object):
    @overload
    def apply(self, features1, features2) -> matches_info:
        """
        @overload
        @param features1 First image features
        @param features2 Second image features
        @param matches_info Found matches
        """

    def apply2(self, features, mask = ...) -> pairwise_matches:
        """
        @brief Performs images matching.

        @param features Features of the source images
        @param pairwise_matches Found pairwise matches
        @param mask Mask indicating which image pairs must be matched  The function is parallelized with the TBB library.  @sa detail::MatchesInfo
        """

    def collectGarbage(self) -> None:
        """
        @brief Frees unused memory allocated before if there is any.
        """

    def isThreadSafe(self) -> retval:
        """
        @return True, if it's possible to use the same matcher instance in parallel, false otherwise
        """


class GainCompensator(ExposureCompensator):
    def apply(self, index, corner, image, mask) -> image:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def getNrFeeds(self) -> retval:
        """"""

    def getSimilarityThreshold(self) -> retval:
        """"""

    def setMatGains(self, umv) -> None:
        """"""

    def setNrFeeds(self, nr_feeds) -> None:
        """"""

    def setSimilarityThreshold(self, similarity_threshold) -> None:
        """"""


class GraphCutSeamFinder(builtins.object):
    def find(self, src, corners, masks) -> masks:
        """"""


class HomographyBasedEstimator(Estimator):
    ...


class ImageFeatures(builtins.object):
    def getKeypoints(self) -> retval:
        """"""


class MatchesInfo(builtins.object):
    def getInliers(self) -> retval:
        """"""

    def getMatches(self) -> retval:
        """"""


class MultiBandBlender(Blender):
    def blend(self, dst, dst_mask) -> tuple[dst, dst_mask]:
        """"""

    def feed(self, img, mask, tl) -> None:
        """"""

    def numBands(self) -> retval:
        """"""

    def prepare(self, dst_roi) -> None:
        """"""

    def setNumBands(self, val) -> None:
        """"""


class NoBundleAdjuster(BundleAdjusterBase):
    ...


class NoExposureCompensator(ExposureCompensator):
    def apply(self, arg1, arg2, arg3, arg4) -> arg3:
        """"""

    def getMatGains(self, umv = ...) -> umv:
        """"""

    def setMatGains(self, umv) -> None:
        """"""


class NoSeamFinder(SeamFinder):
    def find(self, arg1, arg2, arg3) -> arg3:
        """"""


class PairwiseSeamFinder(SeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""


class ProjectorBase(builtins.object):
    ...


class SeamFinder(builtins.object):
    def find(self, src, corners, masks) -> masks:
        """
        @brief Estimates seams.

        @param src Source images
        @param corners Source image top-left corners
        @param masks Source image masks to update
        """

    def createDefault(self, type) -> retval:
        """"""


class SphericalProjector(ProjectorBase):
    def mapBackward(self, u, v, x, y) -> None:
        """"""

    def mapForward(self, x, y, u, v) -> None:
        """"""


class Timelapser(builtins.object):
    def getDst(self) -> retval:
        """"""

    def initialize(self, corners, sizes) -> None:
        """"""

    def process(self, img, mask, tl) -> None:
        """"""

    def createDefault(self, type) -> retval:
        """"""


class TimelapserCrop(Timelapser):
    ...


class VoronoiSeamFinder(PairwiseSeamFinder):
    def find(self, src, corners, masks) -> masks:
        """"""


class ClassificationModel(Model):
    @overload
    def classify(self, frame) -> tuple[classId, conf]:
        """
        @overload
        """

    def getEnableSoftmaxPostProcessing(self) -> retval:
        """
        * @brief Get enable/disable softmax post processing option.
        *
        * This option defaults to false, softmax post processing is not applied within the classify() function.
        """

    def setEnableSoftmaxPostProcessing(self, enable) -> retval:
        """
        * @brief Set enable/disable softmax post processing option.
        *
        * If this option is true, softmax is applied after forward inference within the classify() function
        * to convert the confidences range to [0.0-1.0].
        * This function allows you to toggle this behavior.
        * Please turn true when not contain softmax layer in model.
        * @param[in] enable Set enable softmax post processing within the classify() function.
        """


class DetectionModel(Model):
    def detect(self, frame, confThreshold = ..., nmsThreshold = ...) -> tuple[classIds, confidences, boxes]:
        """
        @brief Given the @p input frame, create input blob, run net and return result detections.
        *  @param[in]  frame  The input image.
        *  @param[out] classIds Class indexes in result detection.
        *  @param[out] confidences A set of corresponding confidences.
        *  @param[out] boxes A set of bounding boxes.
        *  @param[in] confThreshold A threshold used to filter boxes by confidences.
        *  @param[in] nmsThreshold A threshold used in non maximum suppression.
        """

    def getNmsAcrossClasses(self) -> retval:
        """
        * @brief Getter for nmsAcrossClasses. This variable defaults to false,
        * such that when non max suppression is used during the detect() function, it will do so only per-class
        """

    def setNmsAcrossClasses(self, value) -> retval:
        """
        * @brief nmsAcrossClasses defaults to false,
        * such that when non max suppression is used during the detect() function, it will do so per-class.
        * This function allows you to toggle this behaviour.
        * @param[in] value The new value for nmsAcrossClasses
        """


class DictValue(builtins.object):
    def getIntValue(self, idx = ...) -> retval:
        """"""

    def getRealValue(self, idx = ...) -> retval:
        """"""

    def getStringValue(self, idx = ...) -> retval:
        """"""

    def isInt(self) -> retval:
        """"""

    def isReal(self) -> retval:
        """"""

    def isString(self) -> retval:
        """"""


class KeypointsModel(Model):
    def estimate(self, frame, thresh = ...) -> retval:
        """
        @brief Given the @p input frame, create input blob, run net
        *  @param[in]  frame  The input image.
        *  @param thresh minimum confidence threshold to select a keypoint *  @returns a vector holding the x and y coordinates of each detected keypoint *
        """


class Layer(cv2.Algorithm):
    def finalize(self, inputs, outputs = ...) -> outputs:
        """
        @brief Computes and sets internal parameters according to inputs, outputs and blobs.
        *  @param[in]  inputs  vector of already allocated input blobs
        *  @param[out] outputs vector of already allocated output blobs * * If this method is called after network has allocated all memory for input and output blobs * and before inferencing.
        """

    def outputNameToIndex(self, outputName) -> retval:
        """
        @brief Returns index of output blob in output array.
        *  @see inputNameToIndex()
        """

    def run(self, inputs, internals, outputs = ...) -> tuple[outputs, internals]:
        """
        @brief Allocates layer and computes output.
        *  @deprecated This method will be removed in the future release.
        """


class Model(builtins.object):
    def predict(self, frame, outs = ...) -> outs:
        """
        @brief Given the @p input frame, create input blob, run net and return the output @p blobs.
        *  @param[in]  frame  The input image.
        *  @param[out] outs Allocated output blobs, which will store results of the computation.
        """

    def setInputCrop(self, crop) -> retval:
        """
        @brief Set flag crop for frame.
        *  @param[in] crop Flag which indicates whether image will be cropped after resize or not.
        """

    def setInputMean(self, mean) -> retval:
        """
        @brief Set mean value for frame.
        *  @param[in] mean Scalar with mean values which are subtracted from channels.
        """

    def setInputParams(self, scale = ..., size = ..., mean = ..., swapRB = ..., crop = ...) -> None:
        """
        @brief Set preprocessing parameters for frame.
        *  @param[in] size New input size.
        *  @param[in] mean Scalar with mean values which are subtracted from channels.
        *  @param[in] scale Multiplier for frame values.
        *  @param[in] swapRB Flag which indicates that swap first and last channels.
        *  @param[in] crop Flag which indicates whether image will be cropped after resize or not. *  blob(n, c, y, x) = scale * resize( frame(y, x, c) ) - mean(c) )
        """

    def setInputScale(self, scale) -> retval:
        """
        @brief Set scalefactor value for frame.
        *  @param[in] scale Multiplier for frame values.
        """

    def setInputSize(self, size) -> retval:
        """
        @brief Set input size for frame.
        *  @param[in] size New input size. *  @note If shape of the new blob less than 0, then frame size not change.
        """

    @overload
    def setInputSize(self, width, height) -> retval:
        """
        @overload
        *  @param[in] width New input width.
        *  @param[in] height New input height.
        """

    def setInputSwapRB(self, swapRB) -> retval:
        """
        @brief Set flag swapRB for frame.
        *  @param[in] swapRB Flag which indicates that swap first and last channels.
        """

    def setPreferableBackend(self, backendId) -> retval:
        """"""

    def setPreferableTarget(self, targetId) -> retval:
        """"""


class Net(builtins.object):
    def connect(self, outPin, inpPin) -> None:
        """
        @brief Connects output of the first layer to input of the second layer.
        *  @param outPin descriptor of the first layer output.
        *  @param inpPin descriptor of the second layer input. * * Descriptors have the following template <DFN>&lt;layer_name&gt;[.input_number]</DFN>: * - the first part of the template <DFN>layer_name</DFN> is string name of the added layer. *   If this part is empty then the network input pseudo layer will be used; * - the second optional part of the template <DFN>input_number</DFN> *   is either number of the layer input, either label one. *   If this part is omitted then the first layer input will be used. * *  @see setNetInputs(), Layer::inputNameToIndex(), Layer::outputNameToIndex()
        """

    def dump(self) -> retval:
        """
        @brief Dump net to String
        *  @returns String with structure, hyperparameters, backend, target and fusion
        *  Call method after setInput(). To see correct backend, target and fusion run after forward().
        """

    def dumpToFile(self, path) -> None:
        """
        @brief Dump net structure, hyperparameters, backend, target and fusion to dot file
        *  @param path   path to output file with .dot extension *  @see dump()
        """

    def empty(self) -> retval:
        """
        Returns true if there are no layers in the network.
        """

    def enableFusion(self, fusion) -> None:
        """
        @brief Enables or disables layer fusion in the network.
        * @param fusion true to enable the fusion, false to disable. The fusion is enabled by default.
        """

    def enableWinograd(self, useWinograd) -> None:
        """
        @brief Enables or disables the Winograd compute branch. The Winograd compute branch can speed up
        * 3x3 Convolution at a small loss of accuracy.
        * @param useWinograd true to enable the Winograd compute branch. The default is true.
        """

    def forward(self, outputName = ...) -> retval:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputName name for layer which output is needed to get *  @return blob for first output of specified layer. *  @details By default runs forward pass for the whole network.
        """

    def forward(self, outputBlobs = ..., outputName = ...) -> outputBlobs:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputBlobs contains all output blobs for specified layer.
        *  @param outputName name for layer which output is needed to get *  @details If @p outputName is empty, runs forward pass for the whole network.
        """

    def forward(self, outBlobNames, outputBlobs = ...) -> outputBlobs:
        """
        @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
        *  @param outputBlobs contains blobs for first outputs of specified layers.
        *  @param outBlobNames names for layers which outputs are needed to get
        """

    def forwardAndRetrieve(self, outBlobNames) -> outputBlobs:
        """
        @brief Runs forward pass to compute outputs of layers listed in @p outBlobNames.
        *  @param outputBlobs contains all output blobs for each layer specified in @p outBlobNames.
        *  @param outBlobNames names for layers which outputs are needed to get
        """

    def forwardAsync(self, outputName = ...) -> retval:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputName name for layer which output is needed to get *  @details By default runs forward pass for the whole network. * *  This is an asynchronous version of forward(const String&). *  dnn::DNN_BACKEND_INFERENCE_ENGINE backend is required.
        """

    def getFLOPS(self, netInputShapes) -> retval:
        """
        @brief Computes FLOP for whole loaded model with specified input shapes.
        * @param netInputShapes vector of shapes for all net inputs. * @returns computed FLOP.
        """

    @overload
    def getFLOPS(self, netInputShape) -> retval:
        """
        @overload
        """

    @overload
    def getFLOPS(self, layerId, netInputShapes) -> retval:
        """
        @overload
        """

    @overload
    def getFLOPS(self, layerId, netInputShape) -> retval:
        """
        @overload
        """

    def getInputDetails(self) -> tuple[scales, zeropoints]:
        """
        @brief Returns input scale and zeropoint for a quantized Net.
        *  @param scales output parameter for returning input scales.
        *  @param zeropoints output parameter for returning input zeropoints.
        """

    def getLayer(self, layerId) -> retval:
        """
        @brief Returns pointer to layer with specified id or name which the network use.
        """

    @overload
    def getLayer(self, layerName) -> retval:
        """
        @overload
        *  @deprecated Use int getLayerId(const String &layer)
        """

    def getLayerId(self, layer) -> retval:
        """
        @brief Converts string name of the layer to the integer identifier.
        *  @returns id of the layer, or -1 if the layer wasn't found.
        """

    def getLayerNames(self) -> retval:
        """"""

    def getLayerTypes(self) -> layersTypes:
        """
        @brief Returns list of types for layer used in model.
        * @param layersTypes output parameter for returning types.
        """

    def getLayersCount(self, layerType) -> retval:
        """
        @brief Returns count of layers of specified type.
        * @param layerType type. * @returns count of layers
        """

    def getLayersShapes(self, netInputShapes) -> tuple[layersIds, inLayersShapes, outLayersShapes]:
        """
        @brief Returns input and output shapes for all layers in loaded model;
        *  preliminary inferencing isn't necessary.
        *  @param netInputShapes shapes for all input blobs in net input layer.
        *  @param layersIds output parameter for layer IDs.
        *  @param inLayersShapes output parameter for input layers shapes; * order is the same as in layersIds
        *  @param outLayersShapes output parameter for output layers shapes; * order is the same as in layersIds
        """

    @overload
    def getLayersShapes(self, netInputShape) -> tuple[layersIds, inLayersShapes, outLayersShapes]:
        """
        @overload
        """

    @overload
    def getMemoryConsumption(self, netInputShape) -> tuple[weights, blobs]:
        """
        @overload
        """

    @overload
    def getMemoryConsumption(self, layerId, netInputShapes) -> tuple[weights, blobs]:
        """
        @overload
        """

    @overload
    def getMemoryConsumption(self, layerId, netInputShape) -> tuple[weights, blobs]:
        """
        @overload
        """

    def getOutputDetails(self) -> tuple[scales, zeropoints]:
        """
        @brief Returns output scale and zeropoint for a quantized Net.
        *  @param scales output parameter for returning output scales.
        *  @param zeropoints output parameter for returning output zeropoints.
        """

    def getParam(self, layer, numParam = ...) -> retval:
        """
        @brief Returns parameter blob of the layer.
        *  @param layer name or id of the layer.
        *  @param numParam index of the layer parameter in the Layer::blobs array. *  @see Layer::blobs
        """

    def getParam(self, layerName, numParam = ...) -> retval:
        """"""

    def getPerfProfile(self) -> tuple[retval, timings]:
        """
        @brief Returns overall time for inference and timings (in ticks) for layers.
        *
        * Indexes in returned vector correspond to layers ids. Some layers can be fused with others,
        * in this case zero ticks count will be return for that skipped layers. Supported by DNN_BACKEND_OPENCV on DNN_TARGET_CPU only.
        *
        * @param[out] timings vector for tick timings for all layers. * @return overall ticks for model inference.
        """

    def getUnconnectedOutLayers(self) -> retval:
        """
        @brief Returns indexes of layers with unconnected outputs.
        *
        * FIXIT: Rework API to registerOutput() approach, deprecate this call
        """

    def getUnconnectedOutLayersNames(self) -> retval:
        """
        @brief Returns names of layers with unconnected outputs.
        *
        * FIXIT: Rework API to registerOutput() approach, deprecate this call
        """

    def quantize(self, calibData, inputsDtype, outputsDtype, perChannel = ...) -> retval:
        """
        @brief Returns a quantized Net from a floating-point Net.
        *  @param calibData Calibration data to compute the quantization parameters.
        *  @param inputsDtype Datatype of quantized net's inputs. Can be CV_32F or CV_8S.
        *  @param outputsDtype Datatype of quantized net's outputs. Can be CV_32F or CV_8S.
        *  @param perChannel Quantization granularity of quantized Net. The default is true, that means quantize model *  in per-channel way (channel-wise). Set it false to quantize model in per-tensor way (or tensor-wise).
        """

    def setHalideScheduler(self, scheduler) -> None:
        """
        * @brief Compile Halide layers.
        * @param[in] scheduler Path to YAML file with scheduling directives. * @see setPreferableBackend * * Schedule layers that support Halide backend. Then compile them for * specific target. For layers that not represented in scheduling file * or if no manual scheduling used at all, automatic scheduling will be applied.
        """

    def setInput(self, blob, name = ..., scalefactor = ..., mean = ...) -> None:
        """
        @brief Sets the new input value for the network
        *  @param blob        A new blob. Should have CV_32F or CV_8U depth.
        *  @param name        A name of input layer.
        *  @param scalefactor An optional normalization scale.
        *  @param mean        An optional mean subtraction values. *  @see connect(String, String) to know format of the descriptor. * *  If scale or mean values are specified, a final input blob is computed *  as: * \f[input(n,c,h,w) = scalefactor \times (blob(n,c,h,w) - mean_c)\f]
        """

    def setInputShape(self, inputName, shape) -> None:
        """
        @brief Specify shape of network input.
        """

    def setInputsNames(self, inputBlobNames) -> None:
        """
        @brief Sets outputs names of the network input pseudo layer.
        *
        * Each net always has special own the network input pseudo layer with id=0.
        * This layer stores the user blobs only and don't make any computations.
        * In fact, this layer provides the only way to pass user data into the network.
        * As any other layer, this layer can label its outputs and this function provides an easy way to do this.
        """

    def setParam(self, layer, numParam, blob) -> None:
        """
        @brief Sets the new value for the learned param of the layer.
        *  @param layer name or id of the layer.
        *  @param numParam index of the layer parameter in the Layer::blobs array.
        *  @param blob the new value. *  @see Layer::blobs *  @note If shape of the new blob differs from the previous shape, *  then the following forward pass may fail.
        """

    def setParam(self, layerName, numParam, blob) -> None:
        """"""

    def setPreferableBackend(self, backendId) -> None:
        """
        * @brief Ask network to use specific computation backend where it supported.
        * @param[in] backendId backend identifier. * @see Backend * * If OpenCV is compiled with Intel's Inference Engine library, DNN_BACKEND_DEFAULT * means DNN_BACKEND_INFERENCE_ENGINE. Otherwise it equals to DNN_BACKEND_OPENCV.
        """

    def setPreferableTarget(self, targetId) -> None:
        """
        * @brief Ask network to make computations on specific target device.
        * @param[in] targetId target identifier. * @see Target * * List of supported combinations backend / target: * |                        | DNN_BACKEND_OPENCV | DNN_BACKEND_INFERENCE_ENGINE | DNN_BACKEND_HALIDE |  DNN_BACKEND_CUDA | * |------------------------|--------------------|------------------------------|--------------------|-------------------| * | DNN_TARGET_CPU         |                  + |                            + |                  + |                   | * | DNN_TARGET_OPENCL      |                  + |                            + |                  + |                   | * | DNN_TARGET_OPENCL_FP16 |                  + |                            + |                    |                   | * | DNN_TARGET_MYRIAD      |                    |                            + |                    |                   | * | DNN_TARGET_FPGA        |                    |                            + |                    |                   | * | DNN_TARGET_CUDA        |                    |                              |                    |                 + | * | DNN_TARGET_CUDA_FP16   |                    |                              |                    |                 + | * | DNN_TARGET_HDDL        |                    |                            + |                    |                   |
        """

    def readFromModelOptimizer(self, xml, bin) -> retval:
        """
        @brief Create a network from Intel's Model Optimizer intermediate representation (IR).
        *  @param[in] xml XML configuration file with network's topology.
        *  @param[in] bin Binary file with trained weights. *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine *  backend.
        """

    def readFromModelOptimizer(self, bufferModelConfig, bufferWeights) -> retval:
        """
        @brief Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
        *  @param[in] bufferModelConfig buffer with model's configuration.
        *  @param[in] bufferWeights buffer with model's trained weights. *  @returns Net object.
        """


class SegmentationModel(Model):
    def segment(self, frame, mask = ...) -> mask:
        """
        @brief Given the @p input frame, create input blob, run net
        *  @param[in]  frame  The input image.
        *  @param[out] mask Allocated class prediction for each pixel
        """


class TextDetectionModel(Model):
    def detect(self, frame) -> tuple[detections, confidences]:
        """
        @brief Performs detection
        *
        * Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections.
        *
        * Each result is quadrangle's 4 points in this order:
        * - bottom-left
        * - top-left
        * - top-right
        * - bottom-right
        *
        * Use cv::getPerspectiveTransform function to retrieve image region without perspective transformations.
        *
        * @note If DL model doesn't support that kind of output then result may be derived from detectTextRectangles() output.
        *
        * @param[in] frame The input image
        * @param[out] detections array with detections' quadrangles (4 points per result)
        * @param[out] confidences array with detection confidences
        """

    @overload
    def detect(self, frame) -> detections:
        """
        @overload
        """

    def detectTextRectangles(self, frame) -> tuple[detections, confidences]:
        """
        @brief Performs detection
        *
        * Given the input @p frame, prepare network input, run network inference, post-process network output and return result detections.
        *
        * Each result is rotated rectangle.
        *
        * @note Result may be inaccurate in case of strong perspective transformations.
        *
        * @param[in] frame the input image
        * @param[out] detections array with detections' RotationRect results
        * @param[out] confidences array with detection confidences
        """

    @overload
    def detectTextRectangles(self, frame) -> detections:
        """
        @overload
        """


class TextDetectionModel_DB(TextDetectionModel):
    def getBinaryThreshold(self) -> retval:
        """"""

    def getMaxCandidates(self) -> retval:
        """"""

    def getPolygonThreshold(self) -> retval:
        """"""

    def getUnclipRatio(self) -> retval:
        """"""

    def setBinaryThreshold(self, binaryThreshold) -> retval:
        """"""

    def setMaxCandidates(self, maxCandidates) -> retval:
        """"""

    def setPolygonThreshold(self, polygonThreshold) -> retval:
        """"""

    def setUnclipRatio(self, unclipRatio) -> retval:
        """"""


class TextDetectionModel_EAST(TextDetectionModel):
    def getConfidenceThreshold(self) -> retval:
        """
        * @brief Get the detection confidence threshold
        """

    def getNMSThreshold(self) -> retval:
        """
        * @brief Get the detection confidence threshold
        """

    def setConfidenceThreshold(self, confThreshold) -> retval:
        """
        * @brief Set the detection confidence threshold
        * @param[in] confThreshold A threshold used to filter boxes by confidences
        """

    def setNMSThreshold(self, nmsThreshold) -> retval:
        """
        * @brief Set the detection NMS filter threshold
        * @param[in] nmsThreshold A threshold used in non maximum suppression
        """


class TextRecognitionModel(Model):
    def getDecodeType(self) -> retval:
        """
        * @brief Get the decoding method
        * @return the decoding method
        """

    def getVocabulary(self) -> retval:
        """
        * @brief Get the vocabulary for recognition.
        * @return vocabulary the associated vocabulary
        """

    def recognize(self, frame) -> retval:
        """
        * @brief Given the @p input frame, create input blob, run net and return recognition result
        * @param[in] frame The input image * @return The text recognition result
        """

    def recognize(self, frame, roiRects) -> results:
        """
        * @brief Given the @p input frame, create input blob, run net and return recognition result
        * @param[in] frame The input image
        * @param[in] roiRects List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
        * @param[out] results A set of text recognition results.
        """

    def setDecodeOptsCTCPrefixBeamSearch(self, beamSize, vocPruneSize = ...) -> retval:
        """
        * @brief Set the decoding method options for `"CTC-prefix-beam-search"` decode usage
        * @param[in] beamSize Beam size for search
        * @param[in] vocPruneSize Parameter to optimize big vocabulary search, * only take top @p vocPruneSize tokens in each search step, @p vocPruneSize <= 0 stands for disable this prune.
        """

    def setDecodeType(self, decodeType) -> retval:
        """
        * @brief Set the decoding method of translating the network output into string
        * @param[in] decodeType The decoding method of translating the network output into string, currently supported type: *    - `"CTC-greedy"` greedy decoding for the output of CTC-based methods *    - `"CTC-prefix-beam-search"` Prefix beam search decoding for the output of CTC-based methods
        """

    def setVocabulary(self, vocabulary) -> retval:
        """
        * @brief Set the vocabulary for recognition.
        * @param[in] vocabulary the associated vocabulary of the network.
        """


class error(builtins.Exception):
    ...


class Index(builtins.object):
    def build(self, features, params, distType = ...) -> None:
        """"""

    def getAlgorithm(self) -> retval:
        """"""

    def getDistance(self) -> retval:
        """"""

    def knnSearch(self, query, knn, indices = ..., dists = ..., params = ...) -> tuple[indices, dists]:
        """"""

    def load(self, features, filename) -> retval:
        """"""

    def radiusSearch(self, query, radius, maxResults, indices = ..., dists = ..., params = ...) -> tuple[retval, indices, dists]:
        """"""

    def release(self) -> None:
        """"""

    def save(self, filename) -> None:
        """"""


class GNetPackage(builtins.object):
    ...


class GNetParam(builtins.object):
    ...


class PyParams(builtins.object):
    def cfgBatchSize(self, size) -> retval:
        """"""

    def cfgNumRequests(self, nireq) -> retval:
        """"""

    def constInput(self, layer_name, data, hint = ...) -> retval:
        """"""


class PyParams(builtins.object):
    ...


class queue_capacity(builtins.object):
    ...


class GOutputs(builtins.object):
    def getGArray(self, type) -> retval:
        """"""

    def getGMat(self) -> retval:
        """"""

    def getGOpaque(self, type) -> retval:
        """"""

    def getGScalar(self) -> retval:
        """"""


class IStreamSource(builtins.object):
    ...


class Circle(builtins.object):
    ...


class Image(builtins.object):
    ...


class Line(builtins.object):
    ...


class Mosaic(builtins.object):
    ...


class Poly(builtins.object):
    ...


class Rect(builtins.object):
    ...


class Text(builtins.object):
    ...


class GStreamerPipeline(builtins.object):
    ...


class ANN_MLP(StatModel):
    def getAnnealCoolingRatio(self) -> retval:
        """
        @see setAnnealCoolingRatio
        """

    def getAnnealFinalT(self) -> retval:
        """
        @see setAnnealFinalT
        """

    def getAnnealInitialT(self) -> retval:
        """
        @see setAnnealInitialT
        """

    def getAnnealItePerStep(self) -> retval:
        """
        @see setAnnealItePerStep
        """

    def getBackpropMomentumScale(self) -> retval:
        """
        @see setBackpropMomentumScale
        """

    def getBackpropWeightScale(self) -> retval:
        """
        @see setBackpropWeightScale
        """

    def getLayerSizes(self) -> retval:
        """
        Integer vector specifying the number of neurons in each layer including the input and output layers.
        The very first element specifies the number of elements in the input layer.
        The last element - number of elements in the output layer.
        @sa setLayerSizes
        """

    def getRpropDW0(self) -> retval:
        """
        @see setRpropDW0
        """

    def getRpropDWMax(self) -> retval:
        """
        @see setRpropDWMax
        """

    def getRpropDWMin(self) -> retval:
        """
        @see setRpropDWMin
        """

    def getRpropDWMinus(self) -> retval:
        """
        @see setRpropDWMinus
        """

    def getRpropDWPlus(self) -> retval:
        """
        @see setRpropDWPlus
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getTrainMethod(self) -> retval:
        """
        Returns current training method
        """

    def getWeights(self, layerIdx) -> retval:
        """"""

    def setActivationFunction(self, type, param1 = ..., param2 = ...) -> None:
        """
        Initialize the activation function for each neuron.
        Currently the default and the only fully supported activation function is ANN_MLP::SIGMOID_SYM.
        @param type The type of activation function. See ANN_MLP::ActivationFunctions.
        @param param1 The first parameter of the activation function, \f$\alpha\f$. Default value is 0.
        @param param2 The second parameter of the activation function, \f$\beta\f$. Default value is 0.
        """

    def setAnnealCoolingRatio(self, val) -> None:
        """
        @copybrief getAnnealCoolingRatio @see getAnnealCoolingRatio
        """

    def setAnnealFinalT(self, val) -> None:
        """
        @copybrief getAnnealFinalT @see getAnnealFinalT
        """

    def setAnnealInitialT(self, val) -> None:
        """
        @copybrief getAnnealInitialT @see getAnnealInitialT
        """

    def setAnnealItePerStep(self, val) -> None:
        """
        @copybrief getAnnealItePerStep @see getAnnealItePerStep
        """

    def setBackpropMomentumScale(self, val) -> None:
        """
        @copybrief getBackpropMomentumScale @see getBackpropMomentumScale
        """

    def setBackpropWeightScale(self, val) -> None:
        """
        @copybrief getBackpropWeightScale @see getBackpropWeightScale
        """

    def setLayerSizes(self, _layer_sizes) -> None:
        """
        Integer vector specifying the number of neurons in each layer including the input and output layers.
        The very first element specifies the number of elements in the input layer.
        The last element - number of elements in the output layer. Default value is empty Mat.
        @sa getLayerSizes
        """

    def setRpropDW0(self, val) -> None:
        """
        @copybrief getRpropDW0 @see getRpropDW0
        """

    def setRpropDWMax(self, val) -> None:
        """
        @copybrief getRpropDWMax @see getRpropDWMax
        """

    def setRpropDWMin(self, val) -> None:
        """
        @copybrief getRpropDWMin @see getRpropDWMin
        """

    def setRpropDWMinus(self, val) -> None:
        """
        @copybrief getRpropDWMinus @see getRpropDWMinus
        """

    def setRpropDWPlus(self, val) -> None:
        """
        @copybrief getRpropDWPlus @see getRpropDWPlus
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def setTrainMethod(self, method, param1 = ..., param2 = ...) -> None:
        """
        Sets training method and common parameters.
        @param method Default value is ANN_MLP::RPROP. See ANN_MLP::TrainingMethods.
        @param param1 passed to setRpropDW0 for ANN_MLP::RPROP and to setBackpropWeightScale for ANN_MLP::BACKPROP and to initialT for ANN_MLP::ANNEAL.
        @param param2 passed to setRpropDWMin for ANN_MLP::RPROP and to setBackpropMomentumScale for ANN_MLP::BACKPROP and to finalT for ANN_MLP::ANNEAL.
        """

    def create(self) -> retval:
        """
        @brief Creates empty model

        Use StatModel::train to train the model, Algorithm::load\<ANN_MLP\>(filename) to load the pre-trained model.
        Note that the train method has optional flags: ANN_MLP::TrainFlags.
        """

    def load(self, filepath) -> retval:
        """
        @brief Loads and creates a serialized ANN from a file
        *
        * Use ANN::save to serialize and store an ANN to disk.
        * Load the ANN from this file again, by calling this function with the path to the file.
        *
        * @param filepath path to serialized ANN
        """


class Boost(DTrees):
    def getBoostType(self) -> retval:
        """
        @see setBoostType
        """

    def getWeakCount(self) -> retval:
        """
        @see setWeakCount
        """

    def getWeightTrimRate(self) -> retval:
        """
        @see setWeightTrimRate
        """

    def setBoostType(self, val) -> None:
        """
        @copybrief getBoostType @see getBoostType
        """

    def setWeakCount(self, val) -> None:
        """
        @copybrief getWeakCount @see getWeakCount
        """

    def setWeightTrimRate(self, val) -> None:
        """
        @copybrief getWeightTrimRate @see getWeightTrimRate
        """

    def create(self) -> retval:
        """
        Creates the empty model.
        Use StatModel::train to train the model, Algorithm::load\<Boost\>(filename) to load the pre-trained model.
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized Boost from a file
        *
        * Use Boost::save to serialize and store an RTree to disk.
        * Load the Boost from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized Boost
        * @param nodeName name of node containing the classifier
        """


class DTrees(StatModel):
    def getCVFolds(self) -> retval:
        """
        @see setCVFolds
        """

    def getMaxCategories(self) -> retval:
        """
        @see setMaxCategories
        """

    def getMaxDepth(self) -> retval:
        """
        @see setMaxDepth
        """

    def getMinSampleCount(self) -> retval:
        """
        @see setMinSampleCount
        """

    def getPriors(self) -> retval:
        """
        @see setPriors
        """

    def getRegressionAccuracy(self) -> retval:
        """
        @see setRegressionAccuracy
        """

    def getTruncatePrunedTree(self) -> retval:
        """
        @see setTruncatePrunedTree
        """

    def getUse1SERule(self) -> retval:
        """
        @see setUse1SERule
        """

    def getUseSurrogates(self) -> retval:
        """
        @see setUseSurrogates
        """

    def setCVFolds(self, val) -> None:
        """
        @copybrief getCVFolds @see getCVFolds
        """

    def setMaxCategories(self, val) -> None:
        """
        @copybrief getMaxCategories @see getMaxCategories
        """

    def setMaxDepth(self, val) -> None:
        """
        @copybrief getMaxDepth @see getMaxDepth
        """

    def setMinSampleCount(self, val) -> None:
        """
        @copybrief getMinSampleCount @see getMinSampleCount
        """

    def setPriors(self, val) -> None:
        """
        @copybrief getPriors @see getPriors
        """

    def setRegressionAccuracy(self, val) -> None:
        """
        @copybrief getRegressionAccuracy @see getRegressionAccuracy
        """

    def setTruncatePrunedTree(self, val) -> None:
        """
        @copybrief getTruncatePrunedTree @see getTruncatePrunedTree
        """

    def setUse1SERule(self, val) -> None:
        """
        @copybrief getUse1SERule @see getUse1SERule
        """

    def setUseSurrogates(self, val) -> None:
        """
        @copybrief getUseSurrogates @see getUseSurrogates
        """

    def create(self) -> retval:
        """
        @brief Creates the empty model

        The static method creates empty decision tree with the specified parameters. It should be then
        trained using train method (see StatModel::train). Alternatively, you can load the model from
        file using Algorithm::load\<DTrees\>(filename).
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized DTrees from a file
        *
        * Use DTree::save to serialize and store an DTree to disk.
        * Load the DTree from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized DTree
        * @param nodeName name of node containing the classifier
        """


class EM(StatModel):
    def getClustersNumber(self) -> retval:
        """
        @see setClustersNumber
        """

    def getCovarianceMatrixType(self) -> retval:
        """
        @see setCovarianceMatrixType
        """

    def getCovs(self, covs = ...) -> covs:
        """
        @brief Returns covariation matrices

        Returns vector of covariation matrices. Number of matrices is the number of gaussian mixtures,
        each matrix is a square floating-point matrix NxN, where N is the space dimensionality.
        """

    def getMeans(self) -> retval:
        """
        @brief Returns the cluster centers (means of the Gaussian mixture)

        Returns matrix with the number of rows equal to the number of mixtures and number of columns
        equal to the space dimensionality.
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getWeights(self) -> retval:
        """
        @brief Returns weights of the mixtures

        Returns vector with the number of elements equal to the number of mixtures.
        """

    def predict(self, samples, results = ..., flags = ...) -> tuple[retval, results]:
        """
        @brief Returns posterior probabilities for the provided samples

        @param samples The input samples, floating-point matrix
        @param results The optional output \f$ nSamples \times nClusters\f$ matrix of results. It contains posterior probabilities for each sample from the input
        @param flags This parameter will be ignored
        """

    def predict2(self, sample, probs = ...) -> tuple[retval, probs]:
        """
        @brief Returns a likelihood logarithm value and an index of the most probable mixture component
        for the given sample.

        @param sample A sample for classification. It should be a one-channel matrix of \f$1 \times dims\f$ or \f$dims \times 1\f$ size.
        @param probs Optional output matrix that contains posterior probabilities of each component given the sample. It has \f$1 \times nclusters\f$ size and CV_64FC1 type.  The method returns a two-element double vector. Zero element is a likelihood logarithm value for the sample. First element is an index of the most probable mixture component for the given sample.
        """

    def setClustersNumber(self, val) -> None:
        """
        @copybrief getClustersNumber @see getClustersNumber
        """

    def setCovarianceMatrixType(self, val) -> None:
        """
        @copybrief getCovarianceMatrixType @see getCovarianceMatrixType
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def trainE(self, samples, means0, covs0 = ..., weights0 = ..., logLikelihoods = ..., labels = ..., probs = ...) -> tuple[retval, logLikelihoods, labels, probs]:
        """
        @brief Estimate the Gaussian mixture parameters from a samples set.

        This variation starts with Expectation step. You need to provide initial means \f$a_k\f$ of
        mixture components. Optionally you can pass initial weights \f$\pi_k\f$ and covariance matrices
        \f$S_k\f$ of mixture components.

        @param samples Samples from which the Gaussian mixture model will be estimated. It should be a one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it will be converted to the inner matrix of such type for the further computing.
        @param means0 Initial means \f$a_k\f$ of mixture components. It is a one-channel matrix of \f$nclusters \times dims\f$ size. If the matrix does not have CV_64F type it will be converted to the inner matrix of such type for the further computing.
        @param covs0 The vector of initial covariance matrices \f$S_k\f$ of mixture components. Each of covariance matrices is a one-channel matrix of \f$dims \times dims\f$ size. If the matrices do not have CV_64F type they will be converted to the inner matrices of such type for the further computing.
        @param weights0 Initial weights \f$\pi_k\f$ of mixture components. It should be a one-channel floating-point matrix with \f$1 \times nclusters\f$ or \f$nclusters \times 1\f$ size.
        @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
        @param labels The optional output "class label" for each sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
        @param probs The optional output matrix that contains posterior probabilities of each Gaussian mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1 type.
        """

    def trainEM(self, samples, logLikelihoods = ..., labels = ..., probs = ...) -> tuple[retval, logLikelihoods, labels, probs]:
        """
        @brief Estimate the Gaussian mixture parameters from a samples set.

        This variation starts with Expectation step. Initial values of the model parameters will be
        estimated by the k-means algorithm.

        Unlike many of the ML models, %EM is an unsupervised learning algorithm and it does not take
        responses (class labels or function values) as input. Instead, it computes the *Maximum
        Likelihood Estimate* of the Gaussian mixture parameters from an input sample set, stores all the
        parameters inside the structure: \f$p_{i,k}\f$ in probs, \f$a_k\f$ in means , \f$S_k\f$ in
        covs[k], \f$\pi_k\f$ in weights , and optionally computes the output "class label" for each
        sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most
        probable mixture component for each sample).

        The trained model can be used further for prediction, just like any other classifier. The
        trained model is similar to the NormalBayesClassifier.

        @param samples Samples from which the Gaussian mixture model will be estimated. It should be a one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it will be converted to the inner matrix of such type for the further computing.
        @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
        @param labels The optional output "class label" for each sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
        @param probs The optional output matrix that contains posterior probabilities of each Gaussian mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1 type.
        """

    def trainM(self, samples, probs0, logLikelihoods = ..., labels = ..., probs = ...) -> tuple[retval, logLikelihoods, labels, probs]:
        """
        @brief Estimate the Gaussian mixture parameters from a samples set.

        This variation starts with Maximization step. You need to provide initial probabilities
        \f$p_{i,k}\f$ to use this option.

        @param samples Samples from which the Gaussian mixture model will be estimated. It should be a one-channel matrix, each row of which is a sample. If the matrix does not have CV_64F type it will be converted to the inner matrix of such type for the further computing.
        @param probs0 the probabilities
        @param logLikelihoods The optional output matrix that contains a likelihood logarithm value for each sample. It has \f$nsamples \times 1\f$ size and CV_64FC1 type.
        @param labels The optional output "class label" for each sample: \f$\texttt{labels}_i=\texttt{arg max}_k(p_{i,k}), i=1..N\f$ (indices of the most probable mixture component for each sample). It has \f$nsamples \times 1\f$ size and CV_32SC1 type.
        @param probs The optional output matrix that contains posterior probabilities of each Gaussian mixture component given the each sample. It has \f$nsamples \times nclusters\f$ size and CV_64FC1 type.
        """

    def create(self) -> retval:
        """
        Creates empty %EM model.
        The model should be trained then using StatModel::train(traindata, flags) method. Alternatively, you
        can use one of the EM::train\* methods or load it from file using Algorithm::load\<EM\>(filename).
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized EM from a file
        *
        * Use EM::save to serialize and store an EM to disk.
        * Load the EM from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized EM
        * @param nodeName name of node containing the classifier
        """


class KNearest(StatModel):
    def findNearest(self, samples, k, results = ..., neighborResponses = ..., dist = ...) -> tuple[retval, results, neighborResponses, dist]:
        """
        @brief Finds the neighbors and predicts responses for input vectors.

        @param samples Input samples stored by rows. It is a single-precision floating-point matrix of `<number_of_samples> * k` size.
        @param k Number of used nearest neighbors. Should be greater than 1.
        @param results Vector with results of prediction (regression or classification) for each input sample. It is a single-precision floating-point vector with `<number_of_samples>` elements.
        @param neighborResponses Optional output values for corresponding neighbors. It is a single- precision floating-point matrix of `<number_of_samples> * k` size.
        @param dist Optional output distances from the input vectors to the corresponding neighbors. It is a single-precision floating-point matrix of `<number_of_samples> * k` size.  For each input vector (a row of the matrix samples), the method finds the k nearest neighbors. In case of regression, the predicted result is a mean value of the particular vector's neighbor responses. In case of classification, the class is determined by voting.  For each input vector, the neighbors are sorted by their distances to the vector.  In case of C++ interface you can use output pointers to empty matrices and the function will allocate memory itself.  If only a single input vector is passed, all output matrices are optional and the predicted value is returned by the method.  The function is parallelized with the TBB library.
        """

    def getAlgorithmType(self) -> retval:
        """
        @see setAlgorithmType
        """

    def getDefaultK(self) -> retval:
        """
        @see setDefaultK
        """

    def getEmax(self) -> retval:
        """
        @see setEmax
        """

    def getIsClassifier(self) -> retval:
        """
        @see setIsClassifier
        """

    def setAlgorithmType(self, val) -> None:
        """
        @copybrief getAlgorithmType @see getAlgorithmType
        """

    def setDefaultK(self, val) -> None:
        """
        @copybrief getDefaultK @see getDefaultK
        """

    def setEmax(self, val) -> None:
        """
        @copybrief getEmax @see getEmax
        """

    def setIsClassifier(self, val) -> None:
        """
        @copybrief getIsClassifier @see getIsClassifier
        """

    def create(self) -> retval:
        """
        @brief Creates the empty model

        The static method creates empty %KNearest classifier. It should be then trained using StatModel::train method.
        """

    def load(self, filepath) -> retval:
        """
        @brief Loads and creates a serialized knearest from a file
        *
        * Use KNearest::save to serialize and store an KNearest to disk.
        * Load the KNearest from this file again, by calling this function with the path to the file.
        *
        * @param filepath path to serialized KNearest
        """


class LogisticRegression(StatModel):
    def getIterations(self) -> retval:
        """
        @see setIterations
        """

    def getLearningRate(self) -> retval:
        """
        @see setLearningRate
        """

    def getMiniBatchSize(self) -> retval:
        """
        @see setMiniBatchSize
        """

    def getRegularization(self) -> retval:
        """
        @see setRegularization
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getTrainMethod(self) -> retval:
        """
        @see setTrainMethod
        """

    def get_learnt_thetas(self) -> retval:
        """
        @brief This function returns the trained parameters arranged across rows.

        For a two class classification problem, it returns a row matrix. It returns learnt parameters of
        the Logistic Regression as a matrix of type CV_32F.
        """

    def predict(self, samples, results = ..., flags = ...) -> tuple[retval, results]:
        """
        @brief Predicts responses for input samples and returns a float type.

        @param samples The input data for the prediction algorithm. Matrix [m x n], where each row contains variables (features) of one object being classified. Should have data type CV_32F.
        @param results Predicted labels as a column matrix of type CV_32S.
        @param flags Not used.
        """

    def setIterations(self, val) -> None:
        """
        @copybrief getIterations @see getIterations
        """

    def setLearningRate(self, val) -> None:
        """
        @copybrief getLearningRate @see getLearningRate
        """

    def setMiniBatchSize(self, val) -> None:
        """
        @copybrief getMiniBatchSize @see getMiniBatchSize
        """

    def setRegularization(self, val) -> None:
        """
        @copybrief getRegularization @see getRegularization
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def setTrainMethod(self, val) -> None:
        """
        @copybrief getTrainMethod @see getTrainMethod
        """

    def create(self) -> retval:
        """
        @brief Creates empty model.

        Creates Logistic Regression model with parameters given.
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized LogisticRegression from a file
        *
        * Use LogisticRegression::save to serialize and store an LogisticRegression to disk.
        * Load the LogisticRegression from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized LogisticRegression
        * @param nodeName name of node containing the classifier
        """


class NormalBayesClassifier(StatModel):
    def predictProb(self, inputs, outputs = ..., outputProbs = ..., flags = ...) -> tuple[retval, outputs, outputProbs]:
        """
        @brief Predicts the response for sample(s).

        The method estimates the most probable classes for input vectors. Input vectors (one or more)
        are stored as rows of the matrix inputs. In case of multiple input vectors, there should be one
        output vector outputs. The predicted class for a single input vector is returned by the method.
        The vector outputProbs contains the output probabilities corresponding to each element of
        result.
        """

    def create(self) -> retval:
        """
        Creates empty model
        Use StatModel::train to train the model after creation.
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized NormalBayesClassifier from a file
        *
        * Use NormalBayesClassifier::save to serialize and store an NormalBayesClassifier to disk.
        * Load the NormalBayesClassifier from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized NormalBayesClassifier
        * @param nodeName name of node containing the classifier
        """


class ParamGrid(builtins.object):
    def create(self, minVal = ..., maxVal = ..., logstep = ...) -> retval:
        """
        @brief Creates a ParamGrid Ptr that can be given to the %SVM::trainAuto method

        @param minVal minimum value of the parameter grid
        @param maxVal maximum value of the parameter grid
        @param logstep Logarithmic step for iterating the statmodel parameter
        """


class RTrees(DTrees):
    def getActiveVarCount(self) -> retval:
        """
        @see setActiveVarCount
        """

    def getCalculateVarImportance(self) -> retval:
        """
        @see setCalculateVarImportance
        """

    def getOOBError(self) -> retval:
        """
        Returns the OOB error value, computed at the training stage when calcOOBError is set to true.
        * If this flag was set to false, 0 is returned. The OOB error is also scaled by sample weighting.
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getVarImportance(self) -> retval:
        """
        Returns the variable importance array.
        The method returns the variable importance vector, computed at the training stage when
        CalculateVarImportance is set to true. If this flag was set to false, the empty matrix is
        returned.
        """

    def getVotes(self, samples, flags, results = ...) -> results:
        """
        Returns the result of each individual tree in the forest.
        In case the model is a regression problem, the method will return each of the trees'
        results for each of the sample cases. If the model is a classifier, it will return
        a Mat with samples + 1 rows, where the first row gives the class number and the
        following rows return the votes each class had for each sample.
        @param samples Array containing the samples for which votes will be calculated.
        @param results Array where the result of the calculation will be written.
        @param flags Flags for defining the type of RTrees.
        """

    def setActiveVarCount(self, val) -> None:
        """
        @copybrief getActiveVarCount @see getActiveVarCount
        """

    def setCalculateVarImportance(self, val) -> None:
        """
        @copybrief getCalculateVarImportance @see getCalculateVarImportance
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def create(self) -> retval:
        """
        Creates the empty model.
        Use StatModel::train to train the model, StatModel::train to create and train the model,
        Algorithm::load to load the pre-trained model.
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized RTree from a file
        *
        * Use RTree::save to serialize and store an RTree to disk.
        * Load the RTree from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized RTree
        * @param nodeName name of node containing the classifier
        """


class SVM(StatModel):
    def getC(self) -> retval:
        """
        @see setC
        """

    def getClassWeights(self) -> retval:
        """
        @see setClassWeights
        """

    def getCoef0(self) -> retval:
        """
        @see setCoef0
        """

    def getDecisionFunction(self, i, alpha = ..., svidx = ...) -> tuple[retval, alpha, svidx]:
        """
        @brief Retrieves the decision function

        @param i the index of the decision function. If the problem solved is regression, 1-class or 2-class classification, then there will be just one decision function and the index should always be 0. Otherwise, in the case of N-class classification, there will be \f$N(N-1)/2\f$ decision functions.
        @param alpha the optional output vector for weights, corresponding to different support vectors. In the case of linear %SVM all the alpha's will be 1's.
        @param svidx the optional output vector of indices of support vectors within the matrix of support vectors (which can be retrieved by SVM::getSupportVectors). In the case of linear %SVM each decision function consists of a single "compressed" support vector.  The method returns rho parameter of the decision function, a scalar subtracted from the weighted sum of kernel responses.
        """

    def getDegree(self) -> retval:
        """
        @see setDegree
        """

    def getGamma(self) -> retval:
        """
        @see setGamma
        """

    def getKernelType(self) -> retval:
        """
        Type of a %SVM kernel.
        See SVM::KernelTypes. Default value is SVM::RBF.
        """

    def getNu(self) -> retval:
        """
        @see setNu
        """

    def getP(self) -> retval:
        """
        @see setP
        """

    def getSupportVectors(self) -> retval:
        """
        @brief Retrieves all the support vectors

        The method returns all the support vectors as a floating-point matrix, where support vectors are
        stored as matrix rows.
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getType(self) -> retval:
        """
        @see setType
        """

    def getUncompressedSupportVectors(self) -> retval:
        """
        @brief Retrieves all the uncompressed support vectors of a linear %SVM

        The method returns all the uncompressed support vectors of a linear %SVM that the compressed
        support vector, used for prediction, was derived from. They are returned in a floating-point
        matrix, where the support vectors are stored as matrix rows.
        """

    def setC(self, val) -> None:
        """
        @copybrief getC @see getC
        """

    def setClassWeights(self, val) -> None:
        """
        @copybrief getClassWeights @see getClassWeights
        """

    def setCoef0(self, val) -> None:
        """
        @copybrief getCoef0 @see getCoef0
        """

    def setDegree(self, val) -> None:
        """
        @copybrief getDegree @see getDegree
        """

    def setGamma(self, val) -> None:
        """
        @copybrief getGamma @see getGamma
        """

    def setKernel(self, kernelType) -> None:
        """
        Initialize with one of predefined kernels.
        See SVM::KernelTypes.
        """

    def setNu(self, val) -> None:
        """
        @copybrief getNu @see getNu
        """

    def setP(self, val) -> None:
        """
        @copybrief getP @see getP
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def setType(self, val) -> None:
        """
        @copybrief getType @see getType
        """

    def trainAuto(self, samples, layout, responses, kFold = ..., Cgrid = ..., gammaGrid = ..., pGrid = ..., nuGrid = ..., coeffGrid = ..., degreeGrid = ..., balanced = ...) -> retval:
        """
        @brief Trains an %SVM with optimal parameters

        @param samples training samples
        @param layout See ml::SampleTypes.
        @param responses vector of responses associated with the training samples.
        @param kFold Cross-validation parameter. The training set is divided into kFold subsets. One subset is used to test the model, the others form the train set. So, the %SVM algorithm is
        @param Cgrid grid for C
        @param gammaGrid grid for gamma
        @param pGrid grid for p
        @param nuGrid grid for nu
        @param coeffGrid grid for coeff
        @param degreeGrid grid for degree
        @param balanced If true and the problem is 2-class classification then the method creates more balanced cross-validation subsets that is proportions between classes in subsets are close to such proportion in the whole train dataset.  The method trains the %SVM model automatically by choosing the optimal parameters C, gamma, p, nu, coef0, degree. Parameters are considered optimal when the cross-validation estimate of the test set error is minimal.  This function only makes use of SVM::getDefaultGrid for parameter optimization and thus only offers rudimentary parameter options.  This function works for the classification (SVM::C_SVC or SVM::NU_SVC) as well as for the regression (SVM::EPS_SVR or SVM::NU_SVR). If it is SVM::ONE_CLASS, no optimization is made and the usual %SVM with parameters specified in params is executed.
        """

    def create(self) -> retval:
        """
        Creates empty model.
        Use StatModel::train to train the model. Since %SVM has several parameters, you may want to
        find the best parameters for your problem, it can be done with SVM::trainAuto.
        """

    def getDefaultGridPtr(self, param_id) -> retval:
        """
        @brief Generates a grid for %SVM parameters.

        @param param_id %SVM parameters IDs that must be one of the SVM::ParamTypes. The grid is generated for the parameter with this ID.  The function generates a grid pointer for the specified parameter of the %SVM algorithm. The grid may be passed to the function SVM::trainAuto.
        """

    def load(self, filepath) -> retval:
        """
        @brief Loads and creates a serialized svm from a file
        *
        * Use SVM::save to serialize and store an SVM to disk.
        * Load the SVM from this file again, by calling this function with the path to the file.
        *
        * @param filepath path to serialized svm
        """


class SVMSGD(StatModel):
    def getInitialStepSize(self) -> retval:
        """
        @see setInitialStepSize
        """

    def getMarginRegularization(self) -> retval:
        """
        @see setMarginRegularization
        """

    def getMarginType(self) -> retval:
        """
        @see setMarginType
        """

    def getShift(self) -> retval:
        """
        * @return the shift of the trained model (decision function f(x) = weights * x + shift).
        """

    def getStepDecreasingPower(self) -> retval:
        """
        @see setStepDecreasingPower
        """

    def getSvmsgdType(self) -> retval:
        """
        @see setSvmsgdType
        """

    def getTermCriteria(self) -> retval:
        """
        @see setTermCriteria
        """

    def getWeights(self) -> retval:
        """
        * @return the weights of the trained model (decision function f(x) = weights * x + shift).
        """

    def setInitialStepSize(self, InitialStepSize) -> None:
        """
        @copybrief getInitialStepSize @see getInitialStepSize
        """

    def setMarginRegularization(self, marginRegularization) -> None:
        """
        @copybrief getMarginRegularization @see getMarginRegularization
        """

    def setMarginType(self, marginType) -> None:
        """
        @copybrief getMarginType @see getMarginType
        """

    def setOptimalParameters(self, svmsgdType = ..., marginType = ...) -> None:
        """
        @brief Function sets optimal parameters values for chosen SVM SGD model.
        * @param svmsgdType is the type of SVMSGD classifier.
        * @param marginType is the type of margin constraint.
        """

    def setStepDecreasingPower(self, stepDecreasingPower) -> None:
        """
        @copybrief getStepDecreasingPower @see getStepDecreasingPower
        """

    def setSvmsgdType(self, svmsgdType) -> None:
        """
        @copybrief getSvmsgdType @see getSvmsgdType
        """

    def setTermCriteria(self, val) -> None:
        """
        @copybrief getTermCriteria @see getTermCriteria
        """

    def create(self) -> retval:
        """
        @brief Creates empty model.
        * Use StatModel::train to train the model. Since %SVMSGD has several parameters, you may want to
        * find the best parameters for your problem or use setOptimalParameters() to set some default parameters.
        """

    def load(self, filepath, nodeName = ...) -> retval:
        """
        @brief Loads and creates a serialized SVMSGD from a file
        *
        * Use SVMSGD::save to serialize and store an SVMSGD to disk.
        * Load the SVMSGD from this file again, by calling this function with the path to the file.
        * Optionally specify the node for the file containing the classifier
        *
        * @param filepath path to serialized SVMSGD
        * @param nodeName name of node containing the classifier
        """


class StatModel(cv2.Algorithm):
    def calcError(self, data, test, resp = ...) -> tuple[retval, resp]:
        """
        @brief Computes error on the training or test dataset

        @param data the training data
        @param test if true, the error is computed over the test subset of the data, otherwise it's computed over the training subset of the data. Please note that if you loaded a completely different dataset to evaluate already trained classifier, you will probably want not to set the test subset at all with TrainData::setTrainTestSplitRatio and specify test=false, so that the error is computed for the whole new set. Yes, this sounds a bit confusing.
        @param resp the optional output responses.  The method uses StatModel::predict to compute the error. For regression models the error is computed as RMS, for classifiers - as a percent of missclassified samples (0%-100%).
        """

    def empty(self) -> retval:
        """"""

    def getVarCount(self) -> retval:
        """
        @brief Returns the number of variables in training samples
        """

    def isClassifier(self) -> retval:
        """
        @brief Returns true if the model is classifier
        """

    def isTrained(self) -> retval:
        """
        @brief Returns true if the model is trained
        """

    def predict(self, samples, results = ..., flags = ...) -> tuple[retval, results]:
        """
        @brief Predicts response(s) for the provided sample(s)

        @param samples The input samples, floating-point matrix
        @param results The optional output matrix of results.
        @param flags The optional flags, model-dependent. See cv::ml::StatModel::Flags.
        """

    def train(self, trainData, flags = ...) -> retval:
        """
        @brief Trains the statistical model

        @param trainData training data that can be loaded from file using TrainData::loadFromCSV or created with TrainData::create.
        @param flags optional flags, depending on the model. Some of the models can be updated with the new training samples, not completely overwritten (such as NormalBayesClassifier or ANN_MLP).
        """

    def train(self, samples, layout, responses) -> retval:
        """
        @brief Trains the statistical model

        @param samples training samples
        @param layout See ml::SampleTypes.
        @param responses vector of responses associated with the training samples.
        """


class TrainData(builtins.object):
    def getCatCount(self, vi) -> retval:
        """"""

    def getCatMap(self) -> retval:
        """"""

    def getCatOfs(self) -> retval:
        """"""

    def getClassLabels(self) -> retval:
        """
        @brief Returns the vector of class labels

        The function returns vector of unique labels occurred in the responses.
        """

    def getDefaultSubstValues(self) -> retval:
        """"""

    def getLayout(self) -> retval:
        """"""

    def getMissing(self) -> retval:
        """"""

    def getNAllVars(self) -> retval:
        """"""

    def getNSamples(self) -> retval:
        """"""

    def getNTestSamples(self) -> retval:
        """"""

    def getNTrainSamples(self) -> retval:
        """"""

    def getNVars(self) -> retval:
        """"""

    def getNames(self, names) -> None:
        """
        @brief Returns vector of symbolic names captured in loadFromCSV()
        """

    def getNormCatResponses(self) -> retval:
        """"""

    def getResponseType(self) -> retval:
        """"""

    def getResponses(self) -> retval:
        """"""

    def getSample(self, varIdx, sidx, buf) -> None:
        """"""

    def getSampleWeights(self) -> retval:
        """"""

    def getSamples(self) -> retval:
        """"""

    def getTestNormCatResponses(self) -> retval:
        """"""

    def getTestResponses(self) -> retval:
        """"""

    def getTestSampleIdx(self) -> retval:
        """"""

    def getTestSampleWeights(self) -> retval:
        """"""

    def getTestSamples(self) -> retval:
        """
        @brief Returns matrix of test samples
        """

    def getTrainNormCatResponses(self) -> retval:
        """
        @brief Returns the vector of normalized categorical responses

        The function returns vector of responses. Each response is integer from `0` to `<number of
        classes>-1`. The actual label value can be retrieved then from the class label vector, see
        TrainData::getClassLabels.
        """

    def getTrainResponses(self) -> retval:
        """
        @brief Returns the vector of responses

        The function returns ordered or the original categorical responses. Usually it's used in
        regression algorithms.
        """

    def getTrainSampleIdx(self) -> retval:
        """"""

    def getTrainSampleWeights(self) -> retval:
        """"""

    def getTrainSamples(self, layout = ..., compressSamples = ..., compressVars = ...) -> retval:
        """
        @brief Returns matrix of train samples

        @param layout The requested layout. If it's different from the initial one, the matrix is transposed. See ml::SampleTypes.
        @param compressSamples if true, the function returns only the training samples (specified by sampleIdx)
        @param compressVars if true, the function returns the shorter training samples, containing only the active variables.  In current implementation the function tries to avoid physical data copying and returns the matrix stored inside TrainData (unless the transposition or compression is needed).
        """

    def getValues(self, vi, sidx, values) -> None:
        """"""

    def getVarIdx(self) -> retval:
        """"""

    def getVarSymbolFlags(self) -> retval:
        """"""

    def getVarType(self) -> retval:
        """"""

    def setTrainTestSplit(self, count, shuffle = ...) -> None:
        """
        @brief Splits the training data into the training and test parts
        @sa TrainData::setTrainTestSplitRatio
        """

    def setTrainTestSplitRatio(self, ratio, shuffle = ...) -> None:
        """
        @brief Splits the training data into the training and test parts

        The function selects a subset of specified relative size and then returns it as the training
        set. If the function is not called, all the data is used for training. Please, note that for
        each of TrainData::getTrain\* there is corresponding TrainData::getTest\*, so that the test
        subset can be retrieved and processed as well.
        @sa TrainData::setTrainTestSplit
        """

    def shuffleTrainTest(self) -> None:
        """"""

    def create(self, samples, layout, responses, varIdx = ..., sampleIdx = ..., sampleWeights = ..., varType = ...) -> retval:
        """
        @brief Creates training data from in-memory arrays.

        @param samples matrix of samples. It should have CV_32F type.
        @param layout see ml::SampleTypes.
        @param responses matrix of responses. If the responses are scalar, they should be stored as a single row or as a single column. The matrix should have type CV_32F or CV_32S (in the former case the responses are considered as ordered by default; in the latter case - as categorical)
        @param varIdx vector specifying which variables to use for training. It can be an integer vector (CV_32S) containing 0-based variable indices or byte vector (CV_8U) containing a mask of active variables.
        @param sampleIdx vector specifying which samples to use for training. It can be an integer vector (CV_32S) containing 0-based sample indices or byte vector (CV_8U) containing a mask of training samples.
        @param sampleWeights optional vector with weights for each sample. It should have CV_32F type.
        @param varType optional vector of type CV_8U and size `<number_of_variables_in_samples> + <number_of_variables_in_responses>`, containing types of each input and output variable. See ml::VariableTypes.
        """

    def getSubMatrix(self, matrix, idx, layout) -> retval:
        """
        @brief Extract from matrix rows/cols specified by passed indexes.
        @param matrix input matrix (supported types: CV_32S, CV_32F, CV_64F)
        @param idx 1D index vector
        @param layout specifies to extract rows (cv::ml::ROW_SAMPLES) or to extract columns (cv::ml::COL_SAMPLES)
        """

    def getSubVector(self, vec, idx) -> retval:
        """
        @brief Extract from 1D vector elements specified by passed indexes.
        @param vec input vector (supported types: CV_32S, CV_32F, CV_64F)
        @param idx 1D index vector
        """


class Device(builtins.object):
    def OpenCLVersion(self) -> retval:
        """"""

    def OpenCL_C_Version(self) -> retval:
        """

        See help(type(self)) for accurate signature.

        """

    def addressBits(self) -> retval:
        """"""

    def available(self) -> retval:
        """"""

    def compilerAvailable(self) -> retval:
        """"""

    def deviceVersionMajor(self) -> retval:
        """"""

    def deviceVersionMinor(self) -> retval:
        """"""

    def doubleFPConfig(self) -> retval:
        """"""

    def driverVersion(self) -> retval:
        """"""

    def endianLittle(self) -> retval:
        """"""

    def errorCorrectionSupport(self) -> retval:
        """"""

    def executionCapabilities(self) -> retval:
        """"""

    def extensions(self) -> retval:
        """"""

    def globalMemCacheLineSize(self) -> retval:
        """"""

    def globalMemCacheSize(self) -> retval:
        """"""

    def globalMemCacheType(self) -> retval:
        """"""

    def globalMemSize(self) -> retval:
        """"""

    def halfFPConfig(self) -> retval:
        """"""

    def hostUnifiedMemory(self) -> retval:
        """"""

    def image2DMaxHeight(self) -> retval:
        """"""

    def image2DMaxWidth(self) -> retval:
        """"""

    def image3DMaxDepth(self) -> retval:
        """"""

    def image3DMaxHeight(self) -> retval:
        """"""

    def image3DMaxWidth(self) -> retval:
        """"""

    def imageFromBufferSupport(self) -> retval:
        """"""

    def imageMaxArraySize(self) -> retval:
        """"""

    def imageMaxBufferSize(self) -> retval:
        """"""

    def imageSupport(self) -> retval:
        """"""

    def intelSubgroupsSupport(self) -> retval:
        """"""

    def isAMD(self) -> retval:
        """"""

    def isExtensionSupported(self, extensionName) -> retval:
        """"""

    def isIntel(self) -> retval:
        """"""

    def isNVidia(self) -> retval:
        """"""

    def linkerAvailable(self) -> retval:
        """"""

    def localMemSize(self) -> retval:
        """"""

    def localMemType(self) -> retval:
        """"""

    def maxClockFrequency(self) -> retval:
        """"""

    def maxComputeUnits(self) -> retval:
        """"""

    def maxConstantArgs(self) -> retval:
        """"""

    def maxConstantBufferSize(self) -> retval:
        """"""

    def maxMemAllocSize(self) -> retval:
        """"""

    def maxParameterSize(self) -> retval:
        """"""

    def maxReadImageArgs(self) -> retval:
        """"""

    def maxSamplers(self) -> retval:
        """"""

    def maxWorkGroupSize(self) -> retval:
        """"""

    def maxWorkItemDims(self) -> retval:
        """"""

    def maxWriteImageArgs(self) -> retval:
        """"""

    def memBaseAddrAlign(self) -> retval:
        """"""

    def name(self) -> retval:
        """"""

    def nativeVectorWidthChar(self) -> retval:
        """"""

    def nativeVectorWidthDouble(self) -> retval:
        """"""

    def nativeVectorWidthFloat(self) -> retval:
        """"""

    def nativeVectorWidthHalf(self) -> retval:
        """"""

    def nativeVectorWidthInt(self) -> retval:
        """"""

    def nativeVectorWidthLong(self) -> retval:
        """"""

    def nativeVectorWidthShort(self) -> retval:
        """"""

    def preferredVectorWidthChar(self) -> retval:
        """"""

    def preferredVectorWidthDouble(self) -> retval:
        """"""

    def preferredVectorWidthFloat(self) -> retval:
        """"""

    def preferredVectorWidthHalf(self) -> retval:
        """"""

    def preferredVectorWidthInt(self) -> retval:
        """"""

    def preferredVectorWidthLong(self) -> retval:
        """"""

    def preferredVectorWidthShort(self) -> retval:
        """"""

    def printfBufferSize(self) -> retval:
        """"""

    def profilingTimerResolution(self) -> retval:
        """"""

    def singleFPConfig(self) -> retval:
        """"""

    def type(self) -> retval:
        """"""

    def vendorID(self) -> retval:
        """"""

    def vendorName(self) -> retval:
        """"""

    def version(self) -> retval:
        """"""

    def getDefault(self) -> retval:
        """"""


class OpenCLExecutionContext(builtins.object):
    ...


class IntelligentScissorsMB(builtins.object):
    def applyImage(self, image) -> retval:
        """
        @brief Specify input image and extract image features
        *
        * @param image input image. Type is #CV_8UC1 / #CV_8UC3
        """

    def applyImageFeatures(self, non_edge, gradient_direction, gradient_magnitude, image = ...) -> retval:
        """
        @brief Specify custom features of input image
        *
        * Customized advanced variant of applyImage() call.
        *
        * @param non_edge Specify cost of non-edge pixels. Type is CV_8UC1. Expected values are `{0, 1}`.
        * @param gradient_direction Specify gradient direction feature. Type is CV_32FC2. Values are expected to be normalized: `x^2 + y^2 == 1`
        * @param gradient_magnitude Specify cost of gradient magnitude function: Type is CV_32FC1. Values should be in range `[0, 1]`.
        * @param image **Optional parameter**. Must be specified if subset of features is specified (non-specified features are calculated internally)
        """

    def buildMap(self, sourcePt) -> None:
        """
        @brief Prepares a map of optimal paths for the given source point on the image
        *
        * @note applyImage() / applyImageFeatures() must be called before this call
        *
        * @param sourcePt The source point used to find the paths
        """

    def getContour(self, targetPt, contour = ..., backward = ...) -> contour:
        """
        @brief Extracts optimal contour for the given target point on the image
        *
        * @note buildMap() must be called before this call
        *
        * @param targetPt The target point
        * @param[out] contour The list of pixels which contains optimal path between the source and the target points of the image. Type is CV_32SC2 (compatible with `std::vector<Point>`)
        * @param backward Flag to indicate reverse order of retrived pixels (use "true" value to fetch points from the target to the source point)
        """

    def setEdgeFeatureCannyParameters(self, threshold1, threshold2, apertureSize = ..., L2gradient = ...) -> retval:
        """
        @brief Switch edge feature extractor to use Canny edge detector
        *
        * @note "Laplacian Zero-Crossing" feature extractor is used by default (following to original article)
        *
        * @sa Canny
        """

    def setEdgeFeatureZeroCrossingParameters(self, gradient_magnitude_min_value = ...) -> retval:
        """
        @brief Switch to "Laplacian Zero-Crossing" edge feature extractor and specify its parameters
        *
        * This feature extractor is used by default according to article.
        *
        * Implementation has additional filtering for regions with low-amplitude noise.
        * This filtering is enabled through parameter of minimal gradient amplitude (use some small value 4, 8, 16).
        *
        * @note Current implementation of this feature extractor is based on processing of grayscale images (color image is converted to grayscale image first).
        *
        * @note Canny edge detector is a bit slower, but provides better results (especially on color images): use setEdgeFeatureCannyParameters().
        *
        * @param gradient_magnitude_min_value Minimal gradient magnitude value for edge pixels (default: 0, check is disabled)
        """

    def setGradientMagnitudeMaxLimit(self, gradient_magnitude_threshold_max = ...) -> retval:
        """
        @brief Specify gradient magnitude max value threshold
        *
        * Zero limit value is used to disable gradient magnitude thresholding (default behavior, as described in original article).
        * Otherwize pixels with `gradient magnitude >= threshold` have zero cost.
        *
        * @note Thresholding should be used for images with irregular regions (to avoid stuck on parameters from high-contract areas, like embedded logos).
        *
        * @param gradient_magnitude_threshold_max Specify gradient magnitude max value threshold (default: 0, disabled)
        """

    def setWeights(self, weight_non_edge, weight_gradient_direction, weight_gradient_magnitude) -> retval:
        """
        @brief Specify weights of feature functions
        *
        * Consider keeping weights normalized (sum of weights equals to 1.0)
        * Discrete dynamic programming (DP) goal is minimization of costs between pixels.
        *
        * @param weight_non_edge Specify cost of non-edge pixels (default: 0.43f)
        * @param weight_gradient_direction Specify cost of gradient direction function (default: 0.43f)
        * @param weight_gradient_magnitude Specify cost of gradient magnitude function (default: 0.14f)
        """


class ClassWithKeywordProperties(builtins.object):
    ...


class ExportClassName(builtins.object):
    def getFloatParam(self) -> retval:
        """"""

    def getIntParam(self) -> retval:
        """"""

    def create(self, params = ...) -> retval:
        """"""

    def originalName(self) -> retval:
        """"""


class Params(builtins.object):
    ...
