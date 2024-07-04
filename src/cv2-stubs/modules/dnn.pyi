import builtins
from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

timings: TypeAlias = Any
layersTypes: TypeAlias = Any
images_: TypeAlias = Any
inLayersShapes: TypeAlias = Any
outputBlobs: TypeAlias = Any
layersIds: TypeAlias = Any
outputs: TypeAlias = Any
internals: TypeAlias = Any
outs: TypeAlias = Any
detections: TypeAlias = Any
blobs: TypeAlias = Any
weights: TypeAlias = Any
confidences: TypeAlias = Any
zeropoints: TypeAlias = Any
updated_scores: TypeAlias = Any
scales: TypeAlias = Any
classId: TypeAlias = Any
boxes: TypeAlias = Any
conf: TypeAlias = Any
outLayersShapes: TypeAlias = Any
mask: TypeAlias = Any
indices: TypeAlias = Any
classIds: TypeAlias = Any
results: TypeAlias = Any
dst: TypeAlias = Any
retval: TypeAlias = Any

class ClassificationModel(Model):
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
    def detect(self, frame, confThreshold=..., nmsThreshold=...) -> tuple[classIds, confidences, boxes]:
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
    def getIntValue(self, idx=...) -> retval:
        """"""

    def getRealValue(self, idx=...) -> retval:
        """"""

    def getStringValue(self, idx=...) -> retval:
        """"""

    def isInt(self) -> retval:
        """"""

    def isReal(self) -> retval:
        """"""

    def isString(self) -> retval:
        """"""

class KeypointsModel(Model):
    def estimate(self, frame, thresh=...) -> retval:
        """
        @brief Given the @p input frame, create input blob, run net
        *  @param[in]  frame  The input image.
        *  @param thresh minimum confidence threshold to select a keypoint *  @returns a vector holding the x and y coordinates of each detected keypoint *
        """

class Layer(cv2.Algorithm):
    def finalize(self, inputs, outputs=...) -> outputs:
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

    def run(self, inputs, internals, outputs=...) -> tuple[outputs, internals]:
        """
        @brief Allocates layer and computes output.
        *  @deprecated This method will be removed in the future release.
        """

class Model(builtins.object):
    def predict(self, frame, outs=...) -> outs:
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

    def setInputParams(self, scale=..., size=..., mean=..., swapRB=..., crop=...) -> None:
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

    @overload
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

    @overload
    def forward(self, outputName=...) -> retval:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputName name for layer which output is needed to get *  @return blob for first output of specified layer. *  @details By default runs forward pass for the whole network.
        """

    @overload
    def forward(self, outputBlobs=..., outputName=...) -> outputBlobs:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputBlobs contains all output blobs for specified layer.
        *  @param outputName name for layer which output is needed to get *  @details If @p outputName is empty, runs forward pass for the whole network.
        """

    def forward(self, outBlobNames, outputBlobs=...) -> outputBlobs:
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

    def forwardAsync(self, outputName=...) -> retval:
        """
        @brief Runs forward pass to compute output of layer with name @p outputName.
        *  @param outputName name for layer which output is needed to get *  @details By default runs forward pass for the whole network. * *  This is an asynchronous version of forward(const String&). *  dnn::DNN_BACKEND_INFERENCE_ENGINE backend is required.
        """

    @overload
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

    @overload
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

    @overload
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

    @overload
    def getParam(self, layer, numParam=...) -> retval:
        """
        @brief Returns parameter blob of the layer.
        *  @param layer name or id of the layer.
        *  @param numParam index of the layer parameter in the Layer::blobs array. *  @see Layer::blobs
        """

    @overload
    def getParam(self, layerName, numParam=...) -> retval:
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

    def quantize(self, calibData, inputsDtype, outputsDtype, perChannel=...) -> retval:
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

    def setInput(self, blob, name=..., scalefactor=..., mean=...) -> None:
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

    @overload
    def setParam(self, layer, numParam, blob) -> None:
        """
        @brief Sets the new value for the learned param of the layer.
        *  @param layer name or id of the layer.
        *  @param numParam index of the layer parameter in the Layer::blobs array.
        *  @param blob the new value. *  @see Layer::blobs *  @note If shape of the new blob differs from the previous shape, *  then the following forward pass may fail.
        """

    @overload
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

    @overload
    def readFromModelOptimizer(self, xml, bin) -> retval:
        """
        @brief Create a network from Intel's Model Optimizer intermediate representation (IR).
        *  @param[in] xml XML configuration file with network's topology.
        *  @param[in] bin Binary file with trained weights. *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine *  backend.
        """

    @overload
    def readFromModelOptimizer(self, bufferModelConfig, bufferWeights) -> retval:
        """
        @brief Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
        *  @param[in] bufferModelConfig buffer with model's configuration.
        *  @param[in] bufferWeights buffer with model's trained weights. *  @returns Net object.
        """

class SegmentationModel(Model):
    def segment(self, frame, mask=...) -> mask:
        """
        @brief Given the @p input frame, create input blob, run net
        *  @param[in]  frame  The input image.
        *  @param[out] mask Allocated class prediction for each pixel
        """

class TextDetectionModel(Model):
    @overload
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

    @overload
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

    @overload
    def recognize(self, frame) -> retval:
        """
        * @brief Given the @p input frame, create input blob, run net and return recognition result
        * @param[in] frame The input image * @return The text recognition result
        """

    @overload
    def recognize(self, frame, roiRects) -> results:
        """
        * @brief Given the @p input frame, create input blob, run net and return recognition result
        * @param[in] frame The input image
        * @param[in] roiRects List of text detection regions of interest (cv::Rect, CV_32SC4). ROIs is be cropped as the network inputs
        * @param[out] results A set of text recognition results.
        """

    def setDecodeOptsCTCPrefixBeamSearch(self, beamSize, vocPruneSize=...) -> retval:
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

def NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta=..., top_k=...) -> indices:
    """
    @brief Performs non maximum suppression given boxes and corresponding scores.

         * @param bboxes a set of bounding boxes to apply NMS.
         * @param scores a set of corresponding confidences.
         * @param score_threshold a threshold used to filter boxes by score.
         * @param nms_threshold a threshold used in non maximum suppression.
         * @param indices the kept indices of bboxes after NMS.
         * @param eta a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
         * @param top_k if `>0`, keep at most @p top_k picked indices.
    """

def NMSBoxesBatched(bboxes, scores, class_ids, score_threshold, nms_threshold, eta=..., top_k=...) -> indices:
    """
    @brief Performs batched non maximum suppression on given boxes and corresponding scores across different classes.

         * @param bboxes a set of bounding boxes to apply NMS.
         * @param scores a set of corresponding confidences.
         * @param class_ids a set of corresponding class ids. Ids are integer and usually start from 0.
         * @param score_threshold a threshold used to filter boxes by score.
         * @param nms_threshold a threshold used in non maximum suppression.
         * @param indices the kept indices of bboxes after NMS.
         * @param eta a coefficient in adaptive threshold formula: \f$nms\_threshold_{i+1}=eta\cdot nms\_threshold_i\f$.
         * @param top_k if `>0`, keep at most @p top_k picked indices.
    """

def NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold, eta=..., top_k=...) -> indices:
    """
    .
    """

@overload
def Net_readFromModelOptimizer(xml, bin) -> retval:
    """
    @brief Create a network from Intel's Model Optimizer intermediate representation (IR).
             *  @param[in] xml XML configuration file with network's topology.
             *  @param[in] bin Binary file with trained weights.
             *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
             *  backend.
    """

@overload
def Net_readFromModelOptimizer(xml, bin) -> retval:
    """
    @brief Create a network from Intel's Model Optimizer in-memory buffers with intermediate representation (IR).
             *  @param[in] bufferModelConfig buffer with model's configuration.
             *  @param[in] bufferWeights buffer with model's trained weights.
             *  @returns Net object.
    """

def blobFromImage(image, scalefactor=..., size=..., mean=..., swapRB=..., crop=..., ddepth=...) -> retval:
    """
    @brief Creates 4-dimensional blob from image. Optionally resizes and crops @p image from center,
         *  subtract @p mean values, scales values by @p scalefactor, swap Blue and Red channels.
         *  @param image input image (with 1-, 3- or 4-channels).
         *  @param size spatial size for output image
         *  @param mean scalar with mean values which are subtracted from channels. Values are intended
         *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
         *  @param scalefactor multiplier for @p image values.
         *  @param swapRB flag which indicates that swap first and last channels
         *  in 3-channel image is necessary.
         *  @param crop flag which indicates whether image will be cropped after resize or not
         *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
         *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
         *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
         *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
         *  @returns 4-dimensional Mat with NCHW dimensions order.
    """

def blobFromImages(images, scalefactor=..., size=..., mean=..., swapRB=..., crop=..., ddepth=...) -> retval:
    """
    @brief Creates 4-dimensional blob from series of images. Optionally resizes and
         *  crops @p images from center, subtract @p mean values, scales values by @p scalefactor,
         *  swap Blue and Red channels.
         *  @param images input images (all with 1-, 3- or 4-channels).
         *  @param size spatial size for output image
         *  @param mean scalar with mean values which are subtracted from channels. Values are intended
         *  to be in (mean-R, mean-G, mean-B) order if @p image has BGR ordering and @p swapRB is true.
         *  @param scalefactor multiplier for @p images values.
         *  @param swapRB flag which indicates that swap first and last channels
         *  in 3-channel image is necessary.
         *  @param crop flag which indicates whether image will be cropped after resize or not
         *  @param ddepth Depth of output blob. Choose CV_32F or CV_8U.
         *  @details if @p crop is true, input image is resized so one side after resize is equal to corresponding
         *  dimension in @p size and another one is equal or larger. Then, crop from the center is performed.
         *  If @p crop is false, direct resize without cropping and preserving aspect ratio is performed.
         *  @returns 4-dimensional Mat with NCHW dimensions order.
    """

def getAvailableTargets(be) -> retval:
    """
    .
    """

def imagesFromBlob(blob_, images_=...) -> images_:
    """
    @brief Parse a 4D blob and output the images it contains as 2D arrays through a simpler data structure
         *  (std::vector<cv::Mat>).
         *  @param[in] blob_ 4 dimensional array (images, channels, height, width) in floating point precision (CV_32F) from
         *  which you would like to extract the images.
         *  @param[out] images_ array of 2D Mat containing the images extracted from the blob in floating point precision
         *  (CV_32F). They are non normalized neither mean added. The number of returned images equals the first dimension
         *  of the blob (batch size). Every image has a number of channels equals to the second dimension of the blob (depth).
    """

@overload
def readNet(model, config=..., framework=...) -> retval:
    """
    * @brief Read deep learning network represented in one of the supported formats.
          * @param[in] model Binary file contains trained weights. The following file
          *                  extensions are expected for models from different frameworks:
          *                  * `*.caffemodel` (Caffe, http://caffe.berkeleyvision.org/)
          *                  * `*.pb` (TensorFlow, https://www.tensorflow.org/)
          *                  * `*.t7` | `*.net` (Torch, http://torch.ch/)
          *                  * `*.weights` (Darknet, https://pjreddie.com/darknet/)
          *                  * `*.bin` (DLDT, https://software.intel.com/openvino-toolkit)
          *                  * `*.onnx` (ONNX, https://onnx.ai/)
          * @param[in] config Text file contains network configuration. It could be a
          *                   file with the following extensions:
          *                  * `*.prototxt` (Caffe, http://caffe.berkeleyvision.org/)
          *                  * `*.pbtxt` (TensorFlow, https://www.tensorflow.org/)
          *                  * `*.cfg` (Darknet, https://pjreddie.com/darknet/)
          *                  * `*.xml` (DLDT, https://software.intel.com/openvino-toolkit)
          * @param[in] framework Explicit framework name tag to determine a format.
          * @returns Net object.
          *
          * This function automatically detects an origin framework of trained model
          * and calls an appropriate function such @ref readNetFromCaffe, @ref readNetFromTensorflow,
          * @ref readNetFromTorch or @ref readNetFromDarknet. An order of @p model and @p config
          * arguments does not matter.
    """

@overload
def readNet(model, config=..., framework=...) -> retval:
    """
    * @brief Read deep learning network represented in one of the supported formats.
          * @details This is an overloaded member function, provided for convenience.
          *          It differs from the above function only in what argument(s) it accepts.
          * @param[in] framework    Name of origin framework.
          * @param[in] bufferModel  A buffer with a content of binary file with weights
          * @param[in] bufferConfig A buffer with a content of text file contains network configuration.
          * @returns Net object.
    """

@overload
def readNetFromCaffe(prototxt, caffeModel=...) -> retval:
    """
    @brief Reads a network model stored in <a href="http://caffe.berkeleyvision.org">Caffe</a> framework's format.
          * @param prototxt   path to the .prototxt file with text description of the network architecture.
          * @param caffeModel path to the .caffemodel file with learned network.
          * @returns Net object.
    """

@overload
def readNetFromCaffe(prototxt, caffeModel=...) -> retval:
    """
    @brief Reads a network model stored in Caffe model in memory.
          * @param bufferProto buffer containing the content of the .prototxt file
          * @param bufferModel buffer containing the content of the .caffemodel file
          * @returns Net object.
    """

@overload
def readNetFromDarknet(cfgFile, darknetModel=...) -> retval:
    """
    @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
        *  @param cfgFile      path to the .cfg file with text description of the network architecture.
        *  @param darknetModel path to the .weights file with learned network.
        *  @returns Network object that ready to do forward, throw an exception in failure cases.
        *  @returns Net object.
    """

@overload
def readNetFromDarknet(cfgFile, darknetModel=...) -> retval:
    """
    @brief Reads a network model stored in <a href="https://pjreddie.com/darknet/">Darknet</a> model files.
         *  @param bufferCfg   A buffer contains a content of .cfg file with text description of the network architecture.
         *  @param bufferModel A buffer contains a content of .weights file with learned network.
         *  @returns Net object.
    """

@overload
def readNetFromModelOptimizer(xml, bin) -> retval:
    """
    @brief Load a network from Intel's Model Optimizer intermediate representation.
         *  @param[in] xml XML configuration file with network's topology.
         *  @param[in] bin Binary file with trained weights.
         *  @returns Net object.
         *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
         *  backend.
    """

@overload
def readNetFromModelOptimizer(xml, bin) -> retval:
    """
    @brief Load a network from Intel's Model Optimizer intermediate representation.
         *  @param[in] bufferModelConfig Buffer contains XML configuration with network's topology.
         *  @param[in] bufferWeights Buffer contains binary data with trained weights.
         *  @returns Net object.
         *  Networks imported from Intel's Model Optimizer are launched in Intel's Inference Engine
         *  backend.
    """

@overload
def readNetFromONNX(onnxFile) -> retval:
    """
    @brief Reads a network model <a href="https://onnx.ai/">ONNX</a>.
         *  @param onnxFile path to the .onnx file with text description of the network architecture.
         *  @returns Network object that ready to do forward, throw an exception in failure cases.
    """

@overload
def readNetFromONNX(onnxFile) -> retval:
    """
    @brief Reads a network model from <a href="https://onnx.ai/">ONNX</a>
         *         in-memory buffer.
         *  @param buffer in-memory buffer that stores the ONNX model bytes.
         *  @returns Network object that ready to do forward, throw an exception
         *        in failure cases.
    """

@overload
def readNetFromTensorflow(model, config=...) -> retval:
    """
    @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
          * @param model  path to the .pb file with binary protobuf description of the network architecture
          * @param config path to the .pbtxt file that contains text graph definition in protobuf format.
          *               Resulting Net object is built by text graph using weights from a binary one that
          *               let us make it more flexible.
          * @returns Net object.
    """

@overload
def readNetFromTensorflow(model, config=...) -> retval:
    """
    @brief Reads a network model stored in <a href="https://www.tensorflow.org/">TensorFlow</a> framework's format.
          * @param bufferModel buffer containing the content of the pb file
          * @param bufferConfig buffer containing the content of the pbtxt file
          * @returns Net object.
    """

def readNetFromTorch(model, isBinary=..., evaluate=...) -> retval:
    """
    *  @brief Reads a network model stored in <a href="http://torch.ch">Torch7</a> framework's format.
         *  @param model    path to the file, dumped from Torch by using torch.save() function.
         *  @param isBinary specifies whether the network was serialized in ascii mode or binary.
         *  @param evaluate specifies testing phase of network. If true, it's similar to evaluate() method in Torch.
         *  @returns Net object.
         *
         *  @note Ascii mode of Torch serializer is more preferable, because binary mode extensively use `long` type of C language,
         *  which has various bit-length on different systems.
         *
         * The loading file must contain serialized <a href="https://github.com/torch/nn/blob/master/doc/module.md">nn.Module</a> object
         * with importing network. Try to eliminate a custom objects from serialazing data to avoid importing errors.
         *
         * List of supported layers (i.e. object instances derived from Torch nn.Module class):
         * - nn.Sequential
         * - nn.Parallel
         * - nn.Concat
         * - nn.Linear
         * - nn.SpatialConvolution
         * - nn.SpatialMaxPooling, nn.SpatialAveragePooling
         * - nn.ReLU, nn.TanH, nn.Sigmoid
         * - nn.Reshape
         * - nn.SoftMax, nn.LogSoftMax
         *
         * Also some equivalents of these classes from cunn, cudnn, and fbcunn may be successfully imported.
    """

def readTensorFromONNX(path) -> retval:
    """
    @brief Creates blob from .pb file.
         *  @param path to the .pb file with input tensor.
         *  @returns Mat.
    """

def readTorchBlob(filename, isBinary=...) -> retval:
    """
    @brief Loads blob which was serialized as torch.Tensor object of Torch7 framework.
         *  @warning This function has the same limitations as readNetFromTorch().
    """

def shrinkCaffeModel(src, dst, layersTypes=...) -> None:
    """
    @brief Convert all weights of Caffe network to half precision floating point.
         * @param src Path to origin model from Caffe framework contains single
         *            precision floating point weights (usually has `.caffemodel` extension).
         * @param dst Path to destination model with updated weights.
         * @param layersTypes Set of layers types which parameters will be converted.
         *                    By default, converts only Convolutional and Fully-Connected layers'
         *                    weights.
         *
         * @note Shrinked model has no origin float32 weights so it can't be used
         *       in origin Caffe framework anymore. However the structure of data
         *       is taken from NVidia's Caffe fork: https://github.com/NVIDIA/caffe.
         *       So the resulting model may be used there.
    """

def softNMSBoxes(bboxes, scores, score_threshold, nms_threshold, top_k=..., sigma=..., method=...) -> tuple[updated_scores, indices]:
    """
    @brief Performs soft non maximum suppression given boxes and corresponding scores.
         * Reference: https://arxiv.org/abs/1704.04503
         * @param bboxes a set of bounding boxes to apply Soft NMS.
         * @param scores a set of corresponding confidences.
         * @param updated_scores a set of corresponding updated confidences.
         * @param score_threshold a threshold used to filter boxes by score.
         * @param nms_threshold a threshold used in non maximum suppression.
         * @param indices the kept indices of bboxes after NMS.
         * @param top_k keep at most @p top_k picked indices.
         * @param sigma parameter of Gaussian weighting.
         * @param method Gaussian or linear.
         * @see SoftNMSMethod
    """

def writeTextGraph(model, output) -> None:
    """
    @brief Create a text representation for a binary network stored in protocol buffer format.
         *  @param[in] model  A path to binary network.
         *  @param[in] output A path to output text file to be created.
         *
         *  @note To reduce output file size, trained weights are not included.
    """

DNN_BACKEND_CANN: Final[int]
DNN_BACKEND_CUDA: Final[int]
DNN_BACKEND_DEFAULT: Final[int]
DNN_BACKEND_HALIDE: Final[int]
DNN_BACKEND_INFERENCE_ENGINE: Final[int]
DNN_BACKEND_OPENCV: Final[int]
DNN_BACKEND_TIMVX: Final[int]
DNN_BACKEND_VKCOM: Final[int]
DNN_BACKEND_WEBNN: Final[int]
DNN_TARGET_CPU: Final[int]
DNN_TARGET_CUDA: Final[int]
DNN_TARGET_CUDA_FP16: Final[int]
DNN_TARGET_FPGA: Final[int]
DNN_TARGET_HDDL: Final[int]
DNN_TARGET_MYRIAD: Final[int]
DNN_TARGET_NPU: Final[int]
DNN_TARGET_OPENCL: Final[int]
DNN_TARGET_OPENCL_FP16: Final[int]
DNN_TARGET_VULKAN: Final[int]
SOFT_NMSMETHOD_SOFTNMS_GAUSSIAN: Final[int]
SOFT_NMSMETHOD_SOFTNMS_LINEAR: Final[int]
SoftNMSMethod_SOFTNMS_GAUSSIAN: Final[int]
SoftNMSMethod_SOFTNMS_LINEAR: Final[int]
