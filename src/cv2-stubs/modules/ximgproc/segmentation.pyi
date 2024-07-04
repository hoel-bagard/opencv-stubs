from typing import Any, overload, TypeAlias

from ... import functions as cv2

rects: TypeAlias = Any

dst: TypeAlias = Any
retval: TypeAlias = Any

class GraphSegmentation(cv2.Algorithm):
    def getK(self) -> retval:
        """"""

    def getMinSize(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def processImage(self, src, dst=...) -> dst:
        """
        @brief Segment an image and store output in dst
        @param src The input image. Any number of channel (1 (Eg: Gray), 3 (Eg: RGB), 4 (Eg: RGB-D)) can be provided
        @param dst The output segmentation. It's a CV_32SC1 Mat with the same number of cols and rows as input image, with an unique, sequential, id for each pixel.
        """

    def setK(self, k) -> None:
        """"""

    def setMinSize(self, min_size) -> None:
        """"""

    def setSigma(self, sigma) -> None:
        """"""

class SelectiveSearchSegmentation(cv2.Algorithm):
    def addGraphSegmentation(self, g) -> None:
        """
        @brief Add a new graph segmentation in the list of graph segementations to process.
        @param g The graph segmentation
        """

    def addImage(self, img) -> None:
        """
        @brief Add a new image in the list of images to process.
        @param img The image
        """

    def addStrategy(self, s) -> None:
        """
        @brief Add a new strategy in the list of strategy to process.
        @param s The strategy
        """

    def clearGraphSegmentations(self) -> None:
        """
        @brief Clear the list of graph segmentations to process;
        """

    def clearImages(self) -> None:
        """
        @brief Clear the list of images to process
        """

    def clearStrategies(self) -> None:
        """
        @brief Clear the list of strategy to process;
        """

    def process(self) -> rects:
        """
        @brief Based on all images, graph segmentations and stragies, computes all possible rects and return them
        @param rects The list of rects. The first ones are more relevents than the lasts ones.
        """

    def setBaseImage(self, img) -> None:
        """
        @brief Set a image used by switch* functions to initialize the class
        @param img The image
        """

    def switchToSelectiveSearchFast(self, base_k=..., inc_k=..., sigma=...) -> None:
        """
        @brief Initialize the class with the 'Selective search fast' parameters describled in @cite uijlings2013selective.
        @param base_k The k parameter for the first graph segmentation
        @param inc_k The increment of the k parameter for all graph segmentations
        @param sigma The sigma parameter for the graph segmentation
        """

    def switchToSelectiveSearchQuality(self, base_k=..., inc_k=..., sigma=...) -> None:
        """
        @brief Initialize the class with the 'Selective search fast' parameters describled in @cite uijlings2013selective.
        @param base_k The k parameter for the first graph segmentation
        @param inc_k The increment of the k parameter for all graph segmentations
        @param sigma The sigma parameter for the graph segmentation
        """

    def switchToSingleStrategy(self, k=..., sigma=...) -> None:
        """
        @brief Initialize the class with the 'Single stragegy' parameters describled in @cite uijlings2013selective.
        @param k The k parameter for the graph segmentation
        @param sigma The sigma parameter for the graph segmentation
        """

class SelectiveSearchSegmentationStrategy(cv2.Algorithm):
    def get(self, r1, r2) -> retval:
        """
        @brief Return the score between two regions (between 0 and 1)
        @param r1 The first region
        @param r2 The second region
        """

    def merge(self, r1, r2) -> None:
        """
        @brief Inform the strategy that two regions will be merged
        @param r1 The first region
        @param r2 The second region
        """

    def setImage(self, img, regions, sizes, image_id=...) -> None:
        """
        @brief Set a initial image, with a segmentation.
        @param img The input image. Any number of channel can be provided
        @param regions A segmentation of the image. The parameter must be the same size of img.
        @param sizes The sizes of different regions
        @param image_id If not set to -1, try to cache pre-computations. If the same set og (img, regions, size) is used, the image_id need to be the same.
        """

class SelectiveSearchSegmentationStrategyColor(SelectiveSearchSegmentationStrategy): ...
class SelectiveSearchSegmentationStrategyFill(SelectiveSearchSegmentationStrategy): ...

class SelectiveSearchSegmentationStrategyMultiple(SelectiveSearchSegmentationStrategy):
    def addStrategy(self, g, weight) -> None:
        """
        @brief Add a new sub-strategy
        @param g The strategy
        @param weight The weight of the strategy
        """

    def clearStrategies(self) -> None:
        """
        @brief Remove all sub-strategies
        """

class SelectiveSearchSegmentationStrategySize(SelectiveSearchSegmentationStrategy): ...
class SelectiveSearchSegmentationStrategyTexture(SelectiveSearchSegmentationStrategy): ...

def createGraphSegmentation(sigma=..., k=..., min_size=...) -> retval:
    """
    @brief Creates a graph based segmentor
                            @param sigma The sigma parameter, used to smooth image
                            @param k The k parameter of the algorythm
                            @param min_size The minimum size of segments
    """

def createSelectiveSearchSegmentation() -> retval:
    """
    @brief Create a new SelectiveSearchSegmentation class.
    """

def createSelectiveSearchSegmentationStrategyColor() -> retval:
    """
    @brief Create a new color-based strategy
    """

def createSelectiveSearchSegmentationStrategyFill() -> retval:
    """
    @brief Create a new fill-based strategy
    """

@overload
def createSelectiveSearchSegmentationStrategyMultiple() -> retval:
    """
    @brief Create a new multiple strategy
    """

@overload
def createSelectiveSearchSegmentationStrategyMultiple() -> retval:
    """
    @brief Create a new multiple strategy and set one subtrategy
                            @param s1 The first strategy
    """

@overload
def createSelectiveSearchSegmentationStrategyMultiple() -> retval:
    """
    @brief Create a new multiple strategy and set two subtrategies, with equal weights
                            @param s1 The first strategy
                            @param s2 The second strategy
    """

@overload
def createSelectiveSearchSegmentationStrategyMultiple() -> retval:
    """
    @brief Create a new multiple strategy and set three subtrategies, with equal weights
                            @param s1 The first strategy
                            @param s2 The second strategy
                            @param s3 The third strategy
    """

@overload
def createSelectiveSearchSegmentationStrategyMultiple() -> retval:
    """
    @brief Create a new multiple strategy and set four subtrategies, with equal weights
                            @param s1 The first strategy
                            @param s2 The second strategy
                            @param s3 The third strategy
                            @param s4 The forth strategy
    """

def createSelectiveSearchSegmentationStrategySize() -> retval:
    """
    @brief Create a new size-based strategy
    """

def createSelectiveSearchSegmentationStrategyTexture() -> retval:
    """
    @brief Create a new size-based strategy
    """
