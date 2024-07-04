import builtins
from typing import Any, TypeAlias

result: TypeAlias = Any

retval: TypeAlias = Any

class DnnSuperResImpl(builtins.object):
    def getAlgorithm(self) -> retval:
        """
        @brief Returns the scale factor of the model:
        @return Current algorithm.
        """

    def getScale(self) -> retval:
        """
        @brief Returns the scale factor of the model:
        @return Current scale factor.
        """

    def readModel(self, path) -> None:
        """
        @brief Read the model from the given path
        @param path Path to the model file.
        """

    def setModel(self, algo, scale) -> None:
        """
        @brief Set desired model
        @param algo String containing one of the desired models: - __edsr__ - __espcn__ - __fsrcnn__ - __lapsrn__
        @param scale Integer specifying the upscale factor
        """

    def setPreferableBackend(self, backendId) -> None:
        """
        @brief Set computation backend
        """

    def setPreferableTarget(self, targetId) -> None:
        """
        @brief Set computation target
        """

    def upsample(self, img, result=...) -> result:
        """
        @brief Upsample via neural network
        @param img Image to upscale
        @param result Destination upscaled image
        """

    def upsampleMultioutput(self, img, imgs_new, scale_factors, node_names) -> None:
        """
        @brief Upsample via neural network of multiple outputs
        @param img Image to upscale
        @param imgs_new Destination upscaled images
        @param scale_factors Scaling factors of the output nodes
        @param node_names Names of the output nodes in the neural network
        """

    def create(self) -> retval:
        """
        @brief Empty constructor for python
        """

def DnnSuperResImpl_create() -> retval:
    """
    @brief Empty constructor for python
    """
