import builtins
from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

status: TypeAlias = Any
nextPts: TypeAlias = Any
flow: TypeAlias = Any
err: TypeAlias = Any
retval: TypeAlias = Any

class DenseRLOFOpticalFlow(cv2.DenseOpticalFlow):
    def getEPICK(self) -> retval:
        """
        K is a number of nearest-neighbor matches considered, when fitting a locally affine
        *    model. Usually it should be around 128. However, lower values would make the interpolation noticeably faster.
        *    @see ximgproc::EdgeAwareInterpolator,  setEPICK
        """

    def getEPICLambda(self) -> retval:
        """
        Lambda is a parameter defining the weight of the edge-aware term in geodesic distance,
        *    should be in the range of 0 to 1000.
        *    @see ximgproc::EdgeAwareInterpolator, setEPICSigma
        """

    def getEPICSigma(self) -> retval:
        """
        Sigma is a parameter defining how fast the weights decrease in the locally-weighted affine
        *  fitting. Higher values can help preserve fine details, lower values can help to get rid of noise in the
        *  output flow.
        *    @see ximgproc::EdgeAwareInterpolator, setEPICSigma
        """

    def getFgsLambda(self) -> retval:
        """
        Sets the respective fastGlobalSmootherFilter() parameter.
        *    @see ximgproc::EdgeAwareInterpolator, setFgsLambda
        """

    def getFgsSigma(self) -> retval:
        """
        Sets the respective fastGlobalSmootherFilter() parameter.
        *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, setFgsSigma
        """

    def getForwardBackward(self) -> retval:
        """
        @copybrief setForwardBackward
        @see setForwardBackward
        """

    def getGridStep(self) -> retval:
        """
        For each grid point a motion vector is computed. Some motion vectors will be removed due to the forwatd backward
        *  threshold (if set >0). The rest will be the base of the vector field interpolation.
        *    @see getForwardBackward, setGridStep
        """

    def getInterpolation(self) -> retval:
        """
        @copybrief setInterpolation
        *    @see ximgproc::EdgeAwareInterpolator, setInterpolation
        """

    def getRICSLICType(self) -> retval:
        """
        @copybrief setRICSLICType
        *    @see setRICSLICType
        """

    def getRICSPSize(self) -> retval:
        """
        @copybrief setRICSPSize
        *    @see setRICSPSize
        """

    def getRLOFOpticalFlowParameter(self) -> retval:
        """
        @copybrief setRLOFOpticalFlowParameter
        @see optflow::RLOFOpticalFlowParameter, setRLOFOpticalFlowParameter
        """

    def getUsePostProc(self) -> retval:
        """
        @copybrief setUsePostProc
        *    @see ximgproc::fastGlobalSmootherFilter, setUsePostProc
        """

    def getUseVariationalRefinement(self) -> retval:
        """
        @copybrief setUseVariationalRefinement
        *    @see ximgproc::fastGlobalSmootherFilter, setUsePostProc
        """

    def setEPICK(self, val) -> None:
        """
        @copybrief getEPICK
        *    @see ximgproc::EdgeAwareInterpolator, getEPICK
        """

    def setEPICLambda(self, val) -> None:
        """
        @copybrief getEPICLambda
        *    @see ximgproc::EdgeAwareInterpolator, getEPICLambda
        """

    def setEPICSigma(self, val) -> None:
        """
        @copybrief getEPICSigma
        *  @see ximgproc::EdgeAwareInterpolator, getEPICSigma
        """

    def setFgsLambda(self, val) -> None:
        """
        @copybrief getFgsLambda
        *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, getFgsLambda
        """

    def setFgsSigma(self, val) -> None:
        """
        @copybrief getFgsSigma
        *    @see ximgproc::EdgeAwareInterpolator, ximgproc::fastGlobalSmootherFilter, getFgsSigma
        """

    def setForwardBackward(self, val) -> None:
        """
        For each grid point \f$ \mathbf{x} \f$ a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
        *     If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
        *     is larger than threshold given by this function then the motion vector will not be used by the following
        *    vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
        *    will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
        *    @see getForwardBackward, setGridStep
        """

    def setGridStep(self, val) -> None:
        """
        @copybrief getGridStep
        *    @see getGridStep
        """

    def setInterpolation(self, val) -> None:
        """
        Two interpolation algorithms are supported
        * - **INTERP_GEO** applies the fast geodesic interpolation, see @cite Geistert2016.
        * - **INTERP_EPIC_RESIDUAL** applies the edge-preserving interpolation, see @cite Revaud2015,Geistert2016.
        * @see ximgproc::EdgeAwareInterpolator, getInterpolation
        """

    def setRICSLICType(self, val) -> None:
        """
        @brief Parameter to choose superpixel algorithm variant to use:
        * - cv::ximgproc::SLICType SLIC segments image using a desired region_size (value: 100)
        * - cv::ximgproc::SLICType SLICO will optimize using adaptive compactness factor (value: 101)
        * - cv::ximgproc::SLICType MSLIC will optimize using manifold methods resulting in more content-sensitive superpixels (value: 102).
        *  @see cv::ximgproc::createSuperpixelSLIC, cv::ximgproc::RICInterpolator
        """

    def setRICSPSize(self, val) -> None:
        """
        * @see cv::ximgproc::createSuperpixelSLIC, cv::ximgproc::RICInterpolator
        """

    def setRLOFOpticalFlowParameter(self, val) -> None:
        """
        @see optflow::RLOFOpticalFlowParameter, getRLOFOpticalFlowParameter
        """

    def setUsePostProc(self, val) -> None:
        """
        * @see getUsePostProc
        """

    def setUseVariationalRefinement(self, val) -> None:
        """
        * @see getUseVariationalRefinement
        """

    def create(self, rlofParam=..., forwardBackwardThreshold=..., gridStep=..., interp_type=..., epicK=..., epicSigma=..., epicLambda=..., ricSPSize=..., ricSLICType=..., use_post_proc=..., fgsLambda=..., fgsSigma=..., use_variational_refinement=...) -> retval:
        """
        *    @param rlofParam see optflow::RLOFOpticalFlowParameter
        *    @param forwardBackwardThreshold see setForwardBackward
        *    @param gridStep see setGridStep
        *    @param interp_type see setInterpolation
        *    @param epicK see setEPICK
        *    @param epicSigma see setEPICSigma
        *    @param epicLambda see setEPICLambda
        *    @param ricSPSize see setRICSPSize
        *    @param ricSLICType see setRICSLICType
        *    @param use_post_proc see setUsePostProc
        *    @param fgsLambda see setFgsLambda
        *    @param fgsSigma see setFgsSigma
        *    @param use_variational_refinement see setUseVariationalRefinement
        """

class DualTVL1OpticalFlow(cv2.DenseOpticalFlow):
    def getEpsilon(self) -> retval:
        """
        @see setEpsilon
        """

    def getGamma(self) -> retval:
        """
        @see setGamma
        """

    def getInnerIterations(self) -> retval:
        """
        @see setInnerIterations
        """

    def getLambda(self) -> retval:
        """
        @see setLambda
        """

    def getMedianFiltering(self) -> retval:
        """
        @see setMedianFiltering
        """

    def getOuterIterations(self) -> retval:
        """
        @see setOuterIterations
        """

    def getScaleStep(self) -> retval:
        """
        @see setScaleStep
        """

    def getScalesNumber(self) -> retval:
        """
        @see setScalesNumber
        """

    def getTau(self) -> retval:
        """
        @see setTau
        """

    def getTheta(self) -> retval:
        """
        @see setTheta
        """

    def getUseInitialFlow(self) -> retval:
        """
        @see setUseInitialFlow
        """

    def getWarpingsNumber(self) -> retval:
        """
        @see setWarpingsNumber
        """

    def setEpsilon(self, val) -> None:
        """
        @copybrief getEpsilon @see getEpsilon
        """

    def setGamma(self, val) -> None:
        """
        @copybrief getGamma @see getGamma
        """

    def setInnerIterations(self, val) -> None:
        """
        @copybrief getInnerIterations @see getInnerIterations
        """

    def setLambda(self, val) -> None:
        """
        @copybrief getLambda @see getLambda
        """

    def setMedianFiltering(self, val) -> None:
        """
        @copybrief getMedianFiltering @see getMedianFiltering
        """

    def setOuterIterations(self, val) -> None:
        """
        @copybrief getOuterIterations @see getOuterIterations
        """

    def setScaleStep(self, val) -> None:
        """
        @copybrief getScaleStep @see getScaleStep
        """

    def setScalesNumber(self, val) -> None:
        """
        @copybrief getScalesNumber @see getScalesNumber
        """

    def setTau(self, val) -> None:
        """
        @copybrief getTau @see getTau
        """

    def setTheta(self, val) -> None:
        """
        @copybrief getTheta @see getTheta
        """

    def setUseInitialFlow(self, val) -> None:
        """
        @copybrief getUseInitialFlow @see getUseInitialFlow
        """

    def setWarpingsNumber(self, val) -> None:
        """
        @copybrief getWarpingsNumber @see getWarpingsNumber
        """

    def create(self, tau=..., lambda_=..., theta=..., nscales=..., warps=..., epsilon=..., innnerIterations=..., outerIterations=..., scaleStep=..., gamma=..., medianFiltering=..., useInitialFlow=...) -> retval:
        """
        @brief Creates instance of cv::DualTVL1OpticalFlow
        """

class GPCDetails(builtins.object): ...
class GPCPatchDescriptor(builtins.object): ...
class GPCPatchSample(builtins.object): ...
class GPCTrainingSamples(builtins.object): ...
class GPCTree(cv2.Algorithm): ...
class OpticalFlowPCAFlow(cv2.DenseOpticalFlow): ...
class PCAPrior(builtins.object): ...

class RLOFOpticalFlowParameter(builtins.object):
    def getCrossSegmentationThreshold(self) -> retval:
        """"""

    def getGlobalMotionRansacThreshold(self) -> retval:
        """"""

    def getLargeWinSize(self) -> retval:
        """"""

    def getMaxIteration(self) -> retval:
        """"""

    def getMaxLevel(self) -> retval:
        """"""

    def getMinEigenValue(self) -> retval:
        """"""

    def getNormSigma0(self) -> retval:
        """"""

    def getNormSigma1(self) -> retval:
        """"""

    def getSmallWinSize(self) -> retval:
        """"""

    def getSolverType(self) -> retval:
        """"""

    def getSupportRegionType(self) -> retval:
        """"""

    def getUseGlobalMotionPrior(self) -> retval:
        """"""

    def getUseIlluminationModel(self) -> retval:
        """"""

    def getUseInitialFlow(self) -> retval:
        """"""

    def setCrossSegmentationThreshold(self, val) -> None:
        """"""

    def setGlobalMotionRansacThreshold(self, val) -> None:
        """"""

    def setLargeWinSize(self, val) -> None:
        """"""

    def setMaxIteration(self, val) -> None:
        """"""

    def setMaxLevel(self, val) -> None:
        """"""

    def setMinEigenValue(self, val) -> None:
        """"""

    def setNormSigma0(self, val) -> None:
        """"""

    def setNormSigma1(self, val) -> None:
        """"""

    def setSmallWinSize(self, val) -> None:
        """"""

    def setSolverType(self, val) -> None:
        """"""

    def setSupportRegionType(self, val) -> None:
        """"""

    def setUseGlobalMotionPrior(self, val) -> None:
        """"""

    def setUseIlluminationModel(self, val) -> None:
        """"""

    def setUseInitialFlow(self, val) -> None:
        """"""

    def setUseMEstimator(self, val) -> None:
        """
        Enables M-estimator by setting sigma parameters to (3.2, 7.0). Disabling M-estimator can reduce
        *  runtime, while enabling can improve the accuracy.
        *  @param val If true M-estimator is used. If false least-square estimator is used. *    @see setNormSigma0, setNormSigma1
        """

    def create(self) -> retval:
        """"""

class SparseRLOFOpticalFlow(cv2.SparseOpticalFlow):
    def getForwardBackward(self) -> retval:
        """
        @copybrief setForwardBackward
        *    @see setForwardBackward
        """

    def getRLOFOpticalFlowParameter(self) -> retval:
        """
        @copybrief setRLOFOpticalFlowParameter
        *    @see setRLOFOpticalFlowParameter
        """

    def setForwardBackward(self, val) -> None:
        """
        For each feature point a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
        *     If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
        *     is larger than threshold given by this function then the status  will not be used by the following
        *    vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
        *    will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
        *    @see setForwardBackward
        """

    def setRLOFOpticalFlowParameter(self, val) -> None:
        """
        @copydoc DenseRLOFOpticalFlow::setRLOFOpticalFlowParameter
        """

    def create(self, rlofParam=..., forwardBackwardThreshold=...) -> retval:
        """
        *    @param rlofParam see setRLOFOpticalFlowParameter
        *    @param forwardBackwardThreshold see setForwardBackward
        """

def DenseRLOFOpticalFlow_create(rlofParam=..., forwardBackwardThreshold=..., gridStep=..., interp_type=..., epicK=..., epicSigma=..., epicLambda=..., ricSPSize=..., ricSLICType=..., use_post_proc=..., fgsLambda=..., fgsSigma=..., use_variational_refinement=...) -> retval:
    """
    *    @param rlofParam see optflow::RLOFOpticalFlowParameter
         *    @param forwardBackwardThreshold see setForwardBackward
         *    @param gridStep see setGridStep
         *    @param interp_type see setInterpolation
         *    @param epicK see setEPICK
         *    @param epicSigma see setEPICSigma
         *    @param epicLambda see setEPICLambda
         *    @param ricSPSize see setRICSPSize
         *    @param ricSLICType see setRICSLICType
         *    @param use_post_proc see setUsePostProc
         *    @param fgsLambda see setFgsLambda
         *    @param fgsSigma see setFgsSigma
         *    @param use_variational_refinement see setUseVariationalRefinement
    """

def DualTVL1OpticalFlow_create(tau=..., lambda_=..., theta=..., nscales=..., warps=..., epsilon=..., innnerIterations=..., outerIterations=..., scaleStep=..., gamma=..., medianFiltering=..., useInitialFlow=...) -> retval:
    """
    @brief Creates instance of cv::DualTVL1OpticalFlow
    """

def RLOFOpticalFlowParameter_create() -> retval:
    """
    .
    """

def SparseRLOFOpticalFlow_create(rlofParam=..., forwardBackwardThreshold=...) -> retval:
    """
    *    @param rlofParam see setRLOFOpticalFlowParameter
         *    @param forwardBackwardThreshold see setForwardBackward
    """

def calcOpticalFlowDenseRLOF(I0, I1, flow, rlofParam=..., forwardBackwardThreshold=..., gridStep=..., interp_type=..., epicK=..., epicSigma=..., epicLambda=..., ricSPSize=..., ricSLICType=..., use_post_proc=..., fgsLambda=..., fgsSigma=..., use_variational_refinement=...) -> flow:
    """
    @brief Fast dense optical flow computation based on robust local optical flow (RLOF) algorithms and sparse-to-dense interpolation scheme.

    The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
    and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
    proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
    The implementation is derived from optflow::calcOpticalFlowPyrLK().

    The sparse-to-dense interpolation scheme allows for fast computation of dense optical flow using RLOF (see @cite Geistert2016).
    For this scheme the following steps are applied:
    -# motion vector seeded at a regular sampled grid are computed. The sparsity of this grid can be configured with setGridStep
    -# (optinally) errornous motion vectors are filter based on the forward backward confidence. The threshold can be configured
    with setForwardBackward. The filter is only applied if the threshold >0 but than the runtime is doubled due to the estimation
    of the backward flow.
    -# Vector field interpolation is applied to the motion vector set to obtain a dense vector field.

    @param I0 first 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
    = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
    @param I1 second 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
    = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
    @param flow computed flow image that has the same size as I0 and type CV_32FC2.
    @param rlofParam see optflow::RLOFOpticalFlowParameter
    @param forwardBackwardThreshold Threshold for the forward backward confidence check.
    For each grid point \f$ \mathbf{x} \f$ a motion vector \f$ d_{I0,I1}(\mathbf{x}) \f$ is computed.
    If the forward backward error \f[ EP_{FB} = || d_{I0,I1} + d_{I1,I0} || \f]
    is larger than threshold given by this function then the motion vector will not be used by the following
    vector field interpolation. \f$ d_{I1,I0} \f$ denotes the backward flow. Note, the forward backward test
       will only be applied if the threshold > 0. This may results into a doubled runtime for the motion estimation.
    @param gridStep Size of the grid to spawn the motion vectors. For each grid point a motion vector is computed.
    Some motion vectors will be removed due to the forwatd backward threshold (if set >0). The rest will be the
    base of the vector field interpolation.
    @param interp_type interpolation method used to compute the dense optical flow. Two interpolation algorithms are
    supported:
    - **INTERP_GEO** applies the fast geodesic interpolation, see @cite Geistert2016.
    - **INTERP_EPIC_RESIDUAL** applies the edge-preserving interpolation, see @cite Revaud2015,Geistert2016.
    @param epicK see ximgproc::EdgeAwareInterpolator sets the respective parameter.
    @param epicSigma see ximgproc::EdgeAwareInterpolator sets the respective parameter.
    @param epicLambda see ximgproc::EdgeAwareInterpolator sets the respective parameter.
    @param ricSPSize  see ximgproc::RICInterpolator sets the respective parameter.
    @param ricSLICType see ximgproc::RICInterpolator sets the respective parameter.
    @param use_post_proc enables ximgproc::fastGlobalSmootherFilter() parameter.
    @param fgsLambda sets the respective ximgproc::fastGlobalSmootherFilter() parameter.
    @param fgsSigma sets the respective ximgproc::fastGlobalSmootherFilter() parameter.
    @param use_variational_refinement enables VariationalRefinement

    Parameters have been described in @cite Senst2012, @cite Senst2013, @cite Senst2014, @cite Senst2016.
    For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
    @note If the grid size is set to (1,1) and the forward backward threshold <= 0 that the dense optical flow field is purely
    computed with the RLOF.

    @note SIMD parallelization is only available when compiling with SSE4.1.
    @note Note that in output, if no correspondences are found between \a I0 and \a I1, the \a flow is set to 0.

    @sa optflow::DenseRLOFOpticalFlow, optflow::RLOFOpticalFlowParameter
    """

@overload
def calcOpticalFlowSF(from_, to, layers, averaging_block_size, max_flow, flow=...) -> flow:
    """
    @overload
    """

@overload
def calcOpticalFlowSF(from_, to, layers, averaging_block_size, max_flow, flow=...) -> flow:
    """
    @brief Calculate an optical flow using "SimpleFlow" algorithm.

    @param from First 8-bit 3-channel image.
    @param to Second 8-bit 3-channel image of the same size as prev
    @param flow computed flow image that has the same size as prev and type CV_32FC2
    @param layers Number of layers
    @param averaging_block_size Size of block through which we sum up when calculate cost function
    for pixel
    @param max_flow maximal flow that we search at each level
    @param sigma_dist vector smooth spatial sigma parameter
    @param sigma_color vector smooth color sigma parameter
    @param postprocess_window window size for postprocess cross bilateral filter
    @param sigma_dist_fix spatial sigma for postprocess cross bilateralf filter
    @param sigma_color_fix color sigma for postprocess cross bilateral filter
    @param occ_thr threshold for detecting occlusions
    @param upscale_averaging_radius window size for bilateral upscale operation
    @param upscale_sigma_dist spatial sigma for bilateral upscale operation
    @param upscale_sigma_color color sigma for bilateral upscale operation
    @param speed_up_thr threshold to detect point with irregular flow - where flow should be
    recalculated after upscale

    See @cite Tao2012 . And site of project - <http://graphics.berkeley.edu/papers/Tao-SAN-2012-05/>.

    @note
       -   An example using the simpleFlow algorithm can be found at samples/simpleflow_demo.cpp
    """

def calcOpticalFlowSparseRLOF(prevImg, nextImg, prevPts, nextPts, status=..., err=..., rlofParam=..., forwardBackwardThreshold=...) -> tuple[nextPts, status, err]:
    """
    @brief Calculates fast optical flow for a sparse feature set using the robust local optical flow (RLOF) similar
    * to optflow::calcOpticalFlowPyrLK().
    *
    * The RLOF is a fast local optical flow approach described in @cite Senst2012 @cite Senst2013 @cite Senst2014
    and @cite Senst2016 similar to the pyramidal iterative Lucas-Kanade method as
    * proposed by @cite Bouguet00. More details and experiments can be found in the following thesis @cite Senst2019.
    * The implementation is derived from optflow::calcOpticalFlowPyrLK().
    *
    * @param prevImg first 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
    * = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
    * @param nextImg second 8-bit input image. If The cross-based RLOF is used (by selecting optflow::RLOFOpticalFlowParameter::supportRegionType
    * = SupportRegionType::SR_CROSS) image has to be a 8-bit 3 channel image.
    * @param prevPts vector of 2D points for which the flow needs to be found; point coordinates must be single-precision
    * floating-point numbers.
    * @param nextPts output vector of 2D points (with single-precision floating-point coordinates) containing the calculated
    * new positions of input features in the second image; when optflow::RLOFOpticalFlowParameter::useInitialFlow variable is true  the vector must
    * have the same size as in the input and contain the initialization point correspondences.
    * @param status output status vector (of unsigned chars); each element of the vector is set to 1 if the flow for the
    * corresponding features has passed the forward backward check.
    * @param err output vector of errors; each element of the vector is set to the forward backward error for the corresponding feature.
    * @param rlofParam see optflow::RLOFOpticalFlowParameter
    * @param forwardBackwardThreshold Threshold for the forward backward confidence check. If forewardBackwardThreshold <=0 the forward
    *
    * @note SIMD parallelization is only available when compiling with SSE4.1.
    *
    * Parameters have been described in @cite Senst2012, @cite Senst2013, @cite Senst2014 and @cite Senst2016.
    * For the RLOF configuration see optflow::RLOFOpticalFlowParameter for further details.
    """

def calcOpticalFlowSparseToDense(from_, to, flow=..., grid_step=..., k=..., sigma=..., use_post_proc=..., fgs_lambda=..., fgs_sigma=...) -> flow:
    """
    @brief Fast dense optical flow based on PyrLK sparse matches interpolation.

    @param from first 8-bit 3-channel or 1-channel image.
    @param to  second 8-bit 3-channel or 1-channel image of the same size as from
    @param flow computed flow image that has the same size as from and CV_32FC2 type
    @param grid_step stride used in sparse match computation. Lower values usually
           result in higher quality but slow down the algorithm.
    @param k number of nearest-neighbor matches considered, when fitting a locally affine
           model. Lower values can make the algorithm noticeably faster at the cost of
           some quality degradation.
    @param sigma parameter defining how fast the weights decrease in the locally-weighted affine
           fitting. Higher values can help preserve fine details, lower values can help to get rid
           of the noise in the output flow.
    @param use_post_proc defines whether the ximgproc::fastGlobalSmootherFilter() is used
           for post-processing after interpolation
    @param fgs_lambda see the respective parameter of the ximgproc::fastGlobalSmootherFilter()
    @param fgs_sigma  see the respective parameter of the ximgproc::fastGlobalSmootherFilter()
    """

def createOptFlow_DeepFlow() -> retval:
    """
    @brief DeepFlow optical flow algorithm implementation.

    The class implements the DeepFlow optical flow algorithm described in @cite Weinzaepfel2013 . See
    also <http://lear.inrialpes.fr/src/deepmatching/> .
    Parameters - class fields - that may be modified after creating a class instance:
    -   member float alpha
    Smoothness assumption weight
    -   member float delta
    Color constancy assumption weight
    -   member float gamma
    Gradient constancy weight
    -   member float sigma
    Gaussian smoothing parameter
    -   member int minSize
    Minimal dimension of an image in the pyramid (next, smaller images in the pyramid are generated
    until one of the dimensions reaches this size)
    -   member float downscaleFactor
    Scaling factor in the image pyramid (must be \< 1)
    -   member int fixedPointIterations
    How many iterations on each level of the pyramid
    -   member int sorIterations
    Iterations of Succesive Over-Relaxation (solver)
    -   member float omega
    Relaxation factor in SOR
    """

def createOptFlow_DenseRLOF() -> retval:
    """
    .
    """

def createOptFlow_DualTVL1() -> retval:
    """
    @brief Creates instance of cv::DenseOpticalFlow
    """

def createOptFlow_Farneback() -> retval:
    """
    .
    """

def createOptFlow_PCAFlow() -> retval:
    """
    @brief Creates an instance of PCAFlow
    """

def createOptFlow_SimpleFlow() -> retval:
    """
    .
    """

def createOptFlow_SparseRLOF() -> retval:
    """
    .
    """

def createOptFlow_SparseToDense() -> retval:
    """
    .
    """

GPC_DESCRIPTOR_DCT: Final[int]
GPC_DESCRIPTOR_WHT: Final[int]
INTERP_EPIC: Final[int]
INTERP_GEO: Final[int]
INTERP_RIC: Final[int]
SR_CROSS: Final[int]
SR_FIXED: Final[int]
ST_BILINEAR: Final[int]
ST_STANDART: Final[int]
