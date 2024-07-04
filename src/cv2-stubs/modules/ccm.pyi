import builtins
from typing import Any, Final, TypeAlias

dst: TypeAlias = Any
retval: TypeAlias = Any

class ColorCorrectionModel(builtins.object):
    def getCCM(self) -> retval:
        """"""

    def getLoss(self) -> retval:
        """"""

    def getMask(self) -> retval:
        """"""

    def getWeights(self) -> retval:
        """"""

    def get_dst_rgbl(self) -> retval:
        """"""

    def get_src_rgbl(self) -> retval:
        """"""

    def infer(self, img, islinear=...) -> retval:
        """
        @brief Infer using fitting ccm.
        @param img the input image.
        @param islinear default false. @return the output array.
        """

    def run(self) -> None:
        """
        @brief make color correction
        """

    def setCCM_TYPE(self, ccm_type) -> None:
        """
        @brief set ccm_type
        @param ccm_type the shape of color correction matrix(CCM);\n default: @ref CCM_3x3
        """

    def setColorSpace(self, cs) -> None:
        """
        @brief set ColorSpace
        @note It should be some RGB color space;
        Supported list of color cards:
        - @ref COLOR_SPACE_sRGB
        - @ref COLOR_SPACE_AdobeRGB
        - @ref COLOR_SPACE_WideGamutRGB
        - @ref COLOR_SPACE_ProPhotoRGB
        - @ref COLOR_SPACE_DCI_P3_RGB
        - @ref COLOR_SPACE_AppleRGB
        - @ref COLOR_SPACE_REC_709_RGB
        - @ref COLOR_SPACE_REC_2020_RGB
        @param cs the absolute color space that detected colors convert to;\n default: @ref COLOR_SPACE_sRGB
        """

    def setDistance(self, distance) -> None:
        """
        @brief set Distance
        @param distance the type of color distance;\n default: @ref DISTANCE_CIE2000
        """

    def setEpsilon(self, epsilon) -> None:
        """
        @brief set Epsilon
        @param epsilon used in MinProblemSolver-DownhillSolver;\n Terminal criteria to the algorithm;\n default: 1e-4;
        """

    def setInitialMethod(self, initial_method_type) -> None:
        """
        @brief set InitialMethod
        @param initial_method_type the method of calculating CCM initial value;\n default: INITIAL_METHOD_LEAST_SQUARE
        """

    def setLinear(self, linear_type) -> None:
        """
        @brief set Linear
        @param linear_type the method of linearization;\n default: @ref LINEARIZATION_GAMMA
        """

    def setLinearDegree(self, deg) -> None:
        """
        @brief set degree
        @note only valid when linear is set to
        - @ref LINEARIZATION_COLORPOLYFIT
        - @ref LINEARIZATION_GRAYPOLYFIT
        - @ref LINEARIZATION_COLORLOGPOLYFIT
        - @ref LINEARIZATION_GRAYLOGPOLYFIT

        @param deg the degree of linearization polynomial;\n default: 3
        """

    def setLinearGamma(self, gamma) -> None:
        """
        @brief set Gamma

        @note only valid when linear is set to "gamma";\n

        @param gamma the gamma value of gamma correction;\n default: 2.2;
        """

    def setMaxCount(self, max_count) -> None:
        """
        @brief set MaxCount
        @param max_count used in MinProblemSolver-DownhillSolver;\n Terminal criteria to the algorithm;\n default: 5000;
        """

    def setSaturatedThreshold(self, lower, upper) -> None:
        """
        @brief set SaturatedThreshold.
        The colors in the closed interval [lower, upper] are reserved to participate
        in the calculation of the loss function and initialization parameters
        @param lower the lower threshold to determine saturation;\n default: 0;
        @param upper the upper threshold to determine saturation;\n default: 0
        """

    def setWeightCoeff(self, weights_coeff) -> None:
        """
        @brief set WeightCoeff
        @param weights_coeff the exponent number of L* component of the reference color in CIE Lab color space;\n default: 0
        """

    def setWeightsList(self, weights_list) -> None:
        """
        @brief set WeightsList
        @param weights_list the list of weight of each color;\n default: empty array
        """

CCM_3X3: Final[int]
CCM_3x3: Final[int]
CCM_4X3: Final[int]
CCM_4x3: Final[int]
COLORCHECKER_DIGITAL_SG: Final[int]
COLORCHECKER_DigitalSG: Final[int]
COLORCHECKER_MACBETH: Final[int]
COLORCHECKER_Macbeth: Final[int]
COLORCHECKER_VINYL: Final[int]
COLORCHECKER_Vinyl: Final[int]
COLOR_SPACE_ADOBE_RGB: Final[int]
COLOR_SPACE_ADOBE_RGBL: Final[int]
COLOR_SPACE_APPLE_RGB: Final[int]
COLOR_SPACE_APPLE_RGBL: Final[int]
COLOR_SPACE_AdobeRGB: Final[int]
COLOR_SPACE_AdobeRGBL: Final[int]
COLOR_SPACE_AppleRGB: Final[int]
COLOR_SPACE_AppleRGBL: Final[int]
COLOR_SPACE_DCI_P3_RGB: Final[int]
COLOR_SPACE_DCI_P3_RGBL: Final[int]
COLOR_SPACE_LAB_A_10: int
COLOR_SPACE_LAB_A_2: Final[int]
COLOR_SPACE_LAB_D50_10: int
COLOR_SPACE_LAB_D50_2: int
COLOR_SPACE_LAB_D55_10: int
COLOR_SPACE_LAB_D55_2: Final[int]
COLOR_SPACE_LAB_D65_10: int
COLOR_SPACE_LAB_D65_2: Final[int]
COLOR_SPACE_LAB_D75_10: int
COLOR_SPACE_LAB_D75_2: Final[int]
COLOR_SPACE_LAB_E_10: int
COLOR_SPACE_LAB_E_2: Final[int]
COLOR_SPACE_Lab_A_10: int
COLOR_SPACE_Lab_A_2: Final[int]
COLOR_SPACE_Lab_D50_10: int
COLOR_SPACE_Lab_D50_2: int
COLOR_SPACE_Lab_D55_10: int
COLOR_SPACE_Lab_D55_2: Final[int]
COLOR_SPACE_Lab_D65_10: int
COLOR_SPACE_Lab_D65_2: Final[int]
COLOR_SPACE_Lab_D75_10: int
COLOR_SPACE_Lab_D75_2: Final[int]
COLOR_SPACE_Lab_E_10: int
COLOR_SPACE_Lab_E_2: Final[int]
COLOR_SPACE_PRO_PHOTO_RGB: Final[int]
COLOR_SPACE_PRO_PHOTO_RGBL: Final[int]
COLOR_SPACE_ProPhotoRGB: Final[int]
COLOR_SPACE_ProPhotoRGBL: Final[int]
COLOR_SPACE_REC_2020_RGB: int
COLOR_SPACE_REC_2020_RGBL: int
COLOR_SPACE_REC_709_RGB: int
COLOR_SPACE_REC_709_RGBL: int
COLOR_SPACE_S_RGB: Final[int]
COLOR_SPACE_S_RGBL: Final[int]
COLOR_SPACE_WIDE_GAMUT_RGB: Final[int]
COLOR_SPACE_WIDE_GAMUT_RGBL: Final[int]
COLOR_SPACE_WideGamutRGB: Final[int]
COLOR_SPACE_WideGamutRGBL: Final[int]
COLOR_SPACE_XYZ_A_10: int
COLOR_SPACE_XYZ_A_2: Final[int]
COLOR_SPACE_XYZ_D50_10: int
COLOR_SPACE_XYZ_D50_2: int
COLOR_SPACE_XYZ_D55_10: int
COLOR_SPACE_XYZ_D55_2: Final[int]
COLOR_SPACE_XYZ_D65_10: int
COLOR_SPACE_XYZ_D65_2: Final[int]
COLOR_SPACE_XYZ_D75_10: int
COLOR_SPACE_XYZ_D75_2: Final[int]
COLOR_SPACE_XYZ_E_10: int
COLOR_SPACE_XYZ_E_2: Final[int]
COLOR_SPACE_sRGB: Final[int]
COLOR_SPACE_sRGBL: Final[int]
DISTANCE_CIE2000: int
DISTANCE_CIE76: Final[int]
DISTANCE_CIE94_GRAPHIC_ARTS: Final[int]
DISTANCE_CIE94_TEXTILES: Final[int]
DISTANCE_CMC_1TO1: Final[int]
DISTANCE_CMC_2TO1: Final[int]
DISTANCE_RGB: Final[int]
DISTANCE_RGBL: Final[int]
INITIAL_METHOD_LEAST_SQUARE: Final[int]
INITIAL_METHOD_WHITE_BALANCE: Final[int]
LINEARIZATION_COLORLOGPOLYFIT: Final[int]
LINEARIZATION_COLORPOLYFIT: Final[int]
LINEARIZATION_GAMMA: Final[int]
LINEARIZATION_GRAYLOGPOLYFIT: Final[int]
LINEARIZATION_GRAYPOLYFIT: Final[int]
LINEARIZATION_IDENTITY: Final[int]
