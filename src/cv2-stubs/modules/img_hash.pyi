from typing import Any, Final, overload, TypeAlias

from .. import functions as cv2

outputArr: TypeAlias = Any
retval: TypeAlias = Any

class AverageHash(ImgHashBase):
    def create(self) -> retval:
        """"""

class BlockMeanHash(ImgHashBase):
    def getMean(self) -> retval:
        """"""

    def setMode(self, mode) -> None:
        """
        @brief Create BlockMeanHash object
        @param mode the mode
        """

    def create(self, mode=...) -> retval:
        """"""

class ColorMomentHash(ImgHashBase):
    def create(self) -> retval:
        """"""

class ImgHashBase(cv2.Algorithm):
    def compare(self, hashOne, hashTwo) -> retval:
        """
        @brief Compare the hash value between inOne and inTwo
        @param hashOne Hash value one
        @param hashTwo Hash value two @return value indicate similarity between inOne and inTwo, the meaning of the value vary from algorithms to algorithms
        """

    def compute(self, inputArr, outputArr=...) -> outputArr:
        """
        @brief Computes hash of the input image
        @param inputArr input image want to compute hash value
        @param outputArr hash of the image
        """

class MarrHildrethHash(ImgHashBase):
    def getAlpha(self) -> retval:
        """
        * @brief self explain
        """

    def getScale(self) -> retval:
        """
        * @brief self explain
        """

    def setKernelParam(self, alpha, scale) -> None:
        """
        @brief Set Mh kernel parameters
        @param alpha int scale factor for marr wavelet (default=2).
        @param scale int level of scale factor (default = 1)
        """

    def create(self, alpha=..., scale=...) -> retval:
        """
        @param alpha int scale factor for marr wavelet (default=2).
        @param scale int level of scale factor (default = 1)
        """

class PHash(ImgHashBase):
    def create(self) -> retval:
        """"""

class RadialVarianceHash(ImgHashBase):
    def getNumOfAngleLine(self) -> retval:
        """"""

    def getSigma(self) -> retval:
        """"""

    def setNumOfAngleLine(self, value) -> None:
        """"""

    def setSigma(self, value) -> None:
        """"""

    def create(self, sigma=..., numOfAngleLine=...) -> retval:
        """"""

def AverageHash_create() -> retval:
    """
    .
    """

def BlockMeanHash_create(mode=...) -> retval:
    """
    .
    """

def ColorMomentHash_create() -> retval:
    """
    .
    """

def MarrHildrethHash_create(alpha=..., scale=...) -> retval:
    """
    @param alpha int scale factor for marr wavelet (default=2).
            @param scale int level of scale factor (default = 1)
    """

def PHash_create() -> retval:
    """
    .
    """

def RadialVarianceHash_create(sigma=..., numOfAngleLine=...) -> retval:
    """
    .
    """

def averageHash(inputArr, outputArr=...) -> outputArr:
    """
    @brief Calculates img_hash::AverageHash in one call
    @param inputArr input image want to compute hash value, type should be CV_8UC4, CV_8UC3 or CV_8UC1.
    @param outputArr Hash value of input, it will contain 16 hex decimal number, return type is CV_8U
    """

def blockMeanHash(inputArr, outputArr=..., mode=...) -> outputArr:
    """
    @brief Computes block mean hash of the input image
        @param inputArr input image want to compute hash value, type should be CV_8UC4, CV_8UC3 or CV_8UC1.
        @param outputArr Hash value of input, it will contain 16 hex decimal number, return type is CV_8U
        @param mode the mode
    """

@overload
def colorMomentHash(inputArr, outputArr=...) -> outputArr:
    """
    @brief Computes color moment hash of the input, the algorithm
    """

@overload
def colorMomentHash(inputArr, outputArr=...) -> outputArr:
    """ """

@overload
def colorMomentHash(inputArr, outputArr=...) -> outputArr:
    """
    @param inputArr input image want to compute hash value,
    """

@overload
def colorMomentHash(inputArr, outputArr=...) -> outputArr:
    """
    @param outputArr 42 hash values with type CV_64F(double)
    """

@overload
def marrHildrethHash(inputArr, outputArr=..., alpha=..., scale=...) -> outputArr:
    """
    @brief Computes average hash value of the input image
        @param inputArr input image want to compute hash value,
    """

@overload
def marrHildrethHash(inputArr, outputArr=..., alpha=..., scale=...) -> outputArr:
    """
    @param outputArr Hash value of input, it will contain 16 hex
    """

@overload
def marrHildrethHash(inputArr, outputArr=..., alpha=..., scale=...) -> outputArr:
    """
    @param alpha int scale factor for marr wavelet (default=2).
    @param scale int level of scale factor (default = 1)
    """

def pHash(inputArr, outputArr=...) -> outputArr:
    """
    @brief Computes pHash value of the input image
        @param inputArr input image want to compute hash value,
         type should be CV_8UC4, CV_8UC3, CV_8UC1.
        @param outputArr Hash value of input, it will contain 8 uchar value
    """

@overload
def radialVarianceHash(inputArr, outputArr=..., sigma=..., numOfAngleLine=...) -> outputArr:
    """
    @brief Computes radial variance hash of the input image
        @param inputArr input image want to compute hash value,
    """

@overload
def radialVarianceHash(inputArr, outputArr=..., sigma=..., numOfAngleLine=...) -> outputArr:
    """
    @param outputArr Hash value of input
    @param sigma Gaussian kernel standard deviation
    @param numOfAngleLine The number of angles to consider
    """

BLOCK_MEAN_HASH_MODE_0: int
BLOCK_MEAN_HASH_MODE_1: Final[int]
