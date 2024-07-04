import builtins
from typing import Any, overload, TypeAlias

residual: TypeAlias = Any
pose: TypeAlias = Any
poses: TypeAlias = Any
results: TypeAlias = Any
PCNormals: TypeAlias = Any

dst: TypeAlias = Any
retval: TypeAlias = Any

class ICP(builtins.object):
    @overload
    def registerModelToScene(self, srcPC, dstPC) -> tuple[retval, residual, pose]:
        """
        *  \brief Perform registration
        *
        *  @param [in] srcPC The input point cloud for the model. Expected to have the normals (Nx6). Currently, *  CV_32F is the only supported data type.
        *  @param [in] dstPC The input point cloud for the scene. It is assumed that the model is registered on the scene. Scene remains static. Expected to have the normals (Nx6). Currently, CV_32F is the only supported data type.
        *  @param [out] residual The output registration error.
        *  @param [out] pose Transformation between srcPC and dstPC. *  \return On successful termination, the function returns 0. * *  \details It is assumed that the model is registered on the scene. Scene remains static, while the model transforms. The output poses transform the models onto the scene. Because of the point to plane minimization, the scene is expected to have the normals available. Expected to have the normals (Nx6).
        """

    @overload
    def registerModelToScene(self, srcPC, dstPC, poses) -> tuple[retval, poses]:
        """
        *  \brief Perform registration with multiple initial poses
        *
        *  @param [in] srcPC The input point cloud for the model. Expected to have the normals (Nx6). Currently, *  CV_32F is the only supported data type.
        *  @param [in] dstPC The input point cloud for the scene. Currently, CV_32F is the only supported data type.
        *  @param [in,out] poses Input poses to start with but also list output of poses. *  \return On successful termination, the function returns 0. * *  \details It is assumed that the model is registered on the scene. Scene remains static, while the model transforms. The output poses transform the models onto the scene. Because of the point to plane minimization, the scene is expected to have the normals available. Expected to have the normals (Nx6).
        """

class PPF3DDetector(builtins.object):
    def match(self, scene, relativeSceneSampleStep=..., relativeSceneDistance=...) -> results:
        """
        *  \brief Matches a trained model across a provided scene.
        *
        *  @param [in] scene Point cloud for the scene
        *  @param [out] results List of output poses
        *  @param [in] relativeSceneSampleStep The ratio of scene points to be used for the matching after sampling with relativeSceneDistance. For example, if this value is set to 1.0/5.0, every 5th point from the scene is used for pose estimation. This parameter allows an easy trade-off between speed and accuracy of the matching. Increasing the value leads to less points being used and in turn to a faster but less accurate pose computation. Decreasing the value has the inverse effect.
        *  @param [in] relativeSceneDistance Set the distance threshold relative to the diameter of the model. This parameter is equivalent to relativeSamplingStep in the training stage. This parameter acts like a prior sampling with the relativeSceneSampleStep parameter.
        """

    def trainModel(self, Model) -> None:
        """
        *  \brief Trains a new model.
        *
        *  @param [in] Model The input point cloud with normals (Nx6) * *  \details Uses the parameters set in the constructor to downsample and learn a new model. When the model is learnt, the instance gets ready for calling "match".
        """

class Pose3D(builtins.object):
    def appendPose(self, IncrementalPose) -> None:
        """
        *  \brief Left multiplies the existing pose in order to update the transformation
        *  \param [in] IncrementalPose New pose to apply
        """

    def printPose(self) -> None:
        """"""

    @overload
    def updatePose(self, NewPose) -> None:
        """
        *  \brief Updates the pose with the new one
        *  \param [in] NewPose New pose to overwrite
        """

    @overload
    def updatePose(self, NewR, NewT) -> None:
        """
        *  \brief Updates the pose with the new one
        """

    def updatePoseQuat(self, Q, NewT) -> None:
        """
        *  \brief Updates the pose with the new one, but this time using quaternions to represent rotation
        """

class PoseCluster3D(builtins.object): ...

def addNoisePC(pc, scale) -> retval:
    """
    *  Adds a uniform noise in the given scale to the input point cloud
     @param [in] pc Input point cloud (CV_32F family).
     @param [in] scale Input scale of the noise. The larger the scale, the more noisy the output
    """

def computeNormalsPC3d(PC, NumNeighbors, FlipViewpoint, viewpoint, PCNormals=...) -> tuple[retval, PCNormals]:
    """
    *  @brief Compute the normals of an arbitrary point cloud
     computeNormalsPC3d uses a plane fitting approach to smoothly compute
     local normals. Normals are obtained through the eigenvector of the covariance
     matrix, corresponding to the smallest eigen value.
     If PCNormals is provided to be an Nx6 matrix, then no new allocation
     is made, instead the existing memory is overwritten.
     @param [in] PC Input point cloud to compute the normals for.
     @param [out] PCNormals Output point cloud
     @param [in] NumNeighbors Number of neighbors to take into account in a local region
     @param [in] FlipViewpoint Should normals be flipped to a viewing direction?
     @param [in] viewpoint
     @return Returns 0 on success
    """

def getRandomPose(Pose) -> None:
    """
    *  Generate a random 4x4 pose matrix
     @param [out] Pose The random pose
    """

def loadPLYSimple(fileName, withNormals=...) -> retval:
    """
    *  @brief Load a PLY file
     @param [in] fileName The PLY model to read
     @param [in] withNormals Flag wheather the input PLY contains normal information,
     and whether it should be loaded or not
     @return Returns the matrix on successful load
    """

def samplePCByQuantization(pc, xrange, yrange, zrange, sample_step_relative, weightByCenter=...) -> retval:
    """
    *  Sample a point cloud using uniform steps
     @param [in] pc Input point cloud
     @param [in] xrange X components (min and max) of the bounding box of the model
     @param [in] yrange Y components (min and max) of the bounding box of the model
     @param [in] zrange Z components (min and max) of the bounding box of the model
     @param [in] sample_step_relative The point cloud is sampled such that all points
     have a certain minimum distance. This minimum distance is determined relatively using
     the parameter sample_step_relative.
     @param [in] weightByCenter The contribution of the quantized data points can be weighted
     by the distance to the origin. This parameter enables/disables the use of weighting.
     @return Sampled point cloud
    """

def transformPCPose(pc, Pose) -> retval:
    """
    *  Transforms the point cloud with a given a homogeneous 4x4 pose matrix (in double precision)
     @param [in] pc Input point cloud (CV_32F family). Point clouds with 3 or 6 elements per
     row are expected. In the case where the normals are provided, they are also rotated to be
     compatible with the entire transformation
     @param [in] Pose 4x4 pose matrix, but linearized in row-major form.
     @return Transformed point cloud
    """

def writePLY(PC, fileName) -> None:
    """
    *  @brief Write a point cloud to PLY file
     @param [in] PC Input point cloud
     @param [in] fileName The PLY model file to write
    """

def writePLYVisibleNormals(PC, fileName) -> None:
    """
    *  @brief Used for debbuging pruposes, writes a point cloud to a PLY file with the tip
    *  of the normal vectors as visible red points
    *  @param [in] PC Input point cloud
    *  @param [in] fileName The PLY model file to write
    """
