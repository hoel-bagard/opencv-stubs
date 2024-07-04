from typing import Any, overload, TypeAlias

from .. import functions as cv2

qualityMap: TypeAlias = Any
features: TypeAlias = Any

dst: TypeAlias = Any
retval: TypeAlias = Any

class QualityBRISQUE(QualityBase):
    @overload
    def compute(self, img) -> retval:
        """
        @brief Computes BRISQUE quality score for input image
        @param img Image for which to compute quality @returns cv::Scalar with the score in the first element.  The score ranges from 0 (best quality) to 100 (worst quality)
        """

    @overload
    def compute(self, img, model_file_path, range_file_path) -> retval:
        """
        @brief static method for computing quality
        @param img image for which to compute quality
        @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
        @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml @returns cv::Scalar with the score in the first element.  The score ranges from 0 (best quality) to 100 (worst quality)
        """

    def computeFeatures(self, img, features=...) -> features:
        """
        @brief static method for computing image features used by the BRISQUE algorithm
        @param img image (BGR(A) or grayscale) for which to compute features
        @param features output row vector of features to cv::Mat or cv::UMat
        """

    @overload
    def create(self, model_file_path, range_file_path) -> retval:
        """
        @brief Create an object which calculates quality
        @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
        @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml
        """

    @overload
    def create(self, model, range) -> retval:
        """
        @brief Create an object which calculates quality
        @param model cv::Ptr<cv::ml::SVM> which contains a loaded BRISQUE model
        @param range cv::Mat which contains BRISQUE range data
        """

class QualityBase(cv2.Algorithm):
    def clear(self) -> None:
        """
        @brief Implements Algorithm::clear()
        """

    def compute(self, img) -> retval:
        """
        @brief Compute quality score per channel with the per-channel score in each element of the resulting cv::Scalar.  See specific algorithm for interpreting result scores
        @param img comparison image, or image to evalute for no-reference quality algorithms
        """

    def empty(self) -> retval:
        """
        @brief Implements Algorithm::empty()
        """

    def getQualityMap(self, dst=...) -> dst:
        """
        @brief Returns output quality map that was generated during computation, if supported by the algorithm
        """

class QualityGMSD(QualityBase):
    def clear(self) -> None:
        """
        @brief Implements Algorithm::clear()
        """

    @overload
    def compute(self, cmp) -> retval:
        """
        @brief Compute GMSD
        @param cmp comparison image @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
        """

    @overload
    def compute(self, ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
        """
        @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray() @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
        """

    def empty(self) -> retval:
        """
        @brief Implements Algorithm::empty()
        """

    def create(self, ref) -> retval:
        """
        @brief Create an object which calculates image quality
        @param ref reference image
        """

class QualityMSE(QualityBase):
    def clear(self) -> None:
        """
        @brief Implements Algorithm::clear()
        """

    @overload
    def compute(self, cmpImgs) -> retval:
        """
        @brief Computes MSE for reference images supplied in class constructor and provided comparison images
        @param cmpImgs Comparison image(s) @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to potentially max float (worst)
        """

    @overload
    def compute(self, ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
        """
        @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image=
        @param qualityMap output quality map, or cv::noArray() @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
        """

    def empty(self) -> retval:
        """
        @brief Implements Algorithm::empty()
        """

    def create(self, ref) -> retval:
        """
        @brief Create an object which calculates quality
        @param ref input image to use as the reference for comparison
        """

class QualityPSNR(QualityBase):
    def clear(self) -> None:
        """
        @brief Implements Algorithm::clear()
        """

    @overload
    def compute(self, cmp) -> retval:
        """
        @brief Compute the PSNR
        @param cmp Comparison image @returns Per-channel PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
        """

    @overload
    def compute(self, ref, cmp, qualityMap=..., maxPixelValue=...) -> tuple[retval, qualityMap]:
        """
        @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray()
        @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image @returns PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
        """

    def empty(self) -> retval:
        """
        @brief Implements Algorithm::empty()
        """

    def getMaxPixelValue(self) -> retval:
        """
        @brief return the maximum pixel value used for PSNR computation
        """

    def setMaxPixelValue(self, val) -> None:
        """
        @brief sets the maximum pixel value used for PSNR computation
        @param val Maximum pixel value
        """

    def create(self, ref, maxPixelValue=...) -> retval:
        """
        @brief Create an object which calculates quality
        @param ref input image to use as the source for comparison
        @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image
        """

class QualitySSIM(QualityBase):
    def clear(self) -> None:
        """
        @brief Implements Algorithm::clear()
        """

    @overload
    def compute(self, cmp) -> retval:
        """
        @brief Computes SSIM
        @param cmp Comparison image @returns cv::Scalar with per-channel quality values.  Values range from 0 (worst) to 1 (best)
        """

    @overload
    def compute(self, ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
        """
        @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray() @returns cv::Scalar with per-channel quality values.  Values range from 0 (worst) to 1 (best)
        """

    def empty(self) -> retval:
        """
        @brief Implements Algorithm::empty()
        """

    def create(self, ref) -> retval:
        """
        @brief Create an object which calculates quality
        @param ref input image to use as the reference image for comparison
        """

def QualityBRISQUE_compute(img, model_file_path, range_file_path) -> retval:
    """
    @brief static method for computing quality
        @param img image for which to compute quality
        @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
        @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml
        @returns cv::Scalar with the score in the first element.  The score ranges from 0 (best quality) to 100 (worst quality)
    """

def QualityBRISQUE_computeFeatures(img, features=...) -> features:
    """
    @brief static method for computing image features used by the BRISQUE algorithm
        @param img image (BGR(A) or grayscale) for which to compute features
        @param features output row vector of features to cv::Mat or cv::UMat
    """

@overload
def QualityBRISQUE_create(model_file_path, range_file_path) -> retval:
    """
    @brief Create an object which calculates quality
        @param model_file_path cv::String which contains a path to the BRISQUE model data, eg. /path/to/brisque_model_live.yml
        @param range_file_path cv::String which contains a path to the BRISQUE range data, eg. /path/to/brisque_range_live.yml
    """

@overload
def QualityBRISQUE_create(model_file_path, range_file_path) -> retval:
    """
    @brief Create an object which calculates quality
        @param model cv::Ptr<cv::ml::SVM> which contains a loaded BRISQUE model
        @param range cv::Mat which contains BRISQUE range data
    """

def QualityGMSD_compute(ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
    """
    @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray()
        @returns cv::Scalar with per-channel quality value.  Values range from 0 (worst) to 1 (best)
    """

def QualityGMSD_create(ref) -> retval:
    """
    @brief Create an object which calculates image quality
        @param ref reference image
    """

def QualityMSE_compute(ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
    """
    @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image=
        @param qualityMap output quality map, or cv::noArray()
        @returns cv::Scalar with per-channel quality values.  Values range from 0 (best) to max float (worst)
    """

def QualityMSE_create(ref) -> retval:
    """
    @brief Create an object which calculates quality
        @param ref input image to use as the reference for comparison
    """

def QualityPSNR_compute(ref, cmp, qualityMap=..., maxPixelValue=...) -> tuple[retval, qualityMap]:
    """
    @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray()
        @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image
        @returns PSNR value, or std::numeric_limits<double>::infinity() if the MSE between the two images == 0
    """

def QualityPSNR_create(ref, maxPixelValue=...) -> retval:
    """
    @brief Create an object which calculates quality
        @param ref input image to use as the source for comparison
        @param maxPixelValue maximum per-channel value for any individual pixel; eg 255 for uint8 image
    """

def QualitySSIM_compute(ref, cmp, qualityMap=...) -> tuple[retval, qualityMap]:
    """
    @brief static method for computing quality
        @param ref reference image
        @param cmp comparison image
        @param qualityMap output quality map, or cv::noArray()
        @returns cv::Scalar with per-channel quality values.  Values range from 0 (worst) to 1 (best)
    """

def QualitySSIM_create(ref) -> retval:
    """
    @brief Create an object which calculates quality
        @param ref input image to use as the reference image for comparison
    """
