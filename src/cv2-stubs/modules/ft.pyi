from typing import Any, Final, overload, TypeAlias

c01: TypeAlias = Any
components: TypeAlias = Any
output: TypeAlias = Any
maskOutput: TypeAlias = Any
c00: TypeAlias = Any
kernel: TypeAlias = Any
matrix: TypeAlias = Any
c10: TypeAlias = Any
retval: TypeAlias = Any

@overload
def FT02D_FL_process(matrix, radius, output=...) -> output:
    """
    @brief Sligtly less accurate version of \f$F^0\f$-transfrom computation optimized for higher speed. The methods counts with linear basic function.
        @param matrix Input 3 channels matrix.
        @param radius Radius of the `ft::LINEAR` basic function.
        @param output Output array.
    """

@overload
def FT02D_FL_process(matrix, radius, output=...) -> output:
    """ """

@overload
def FT02D_FL_process_float(matrix, radius, output=...) -> output:
    """
    @brief Sligtly less accurate version of \f$F^0\f$-transfrom computation optimized for higher speed. The methods counts with linear basic function.
        @param matrix Input 3 channels matrix.
        @param radius Radius of the `ft::LINEAR` basic function.
        @param output Output array.
    """

@overload
def FT02D_FL_process_float(matrix, radius, output=...) -> output:
    """ """

@overload
def FT02D_components(matrix, kernel, components=..., mask=...) -> components:
    """
    @brief Computes components of the array using direct \f$F^0\f$-transform.
        @param matrix Input array.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param components Output 32-bit float array for the components.
        @param mask Mask can be used for unwanted area marking.
    """

@overload
def FT02D_components(matrix, kernel, components=..., mask=...) -> components:
    """ """

@overload
def FT02D_inverseFT(components, kernel, width, height, output=...) -> output:
    """
    @brief Computes inverse \f$F^0\f$-transfrom.
        @param components Input 32-bit float single channel array for the components.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param output Output 32-bit float array.
        @param width Width of the output array.
        @param height Height of the output array.
    """

@overload
def FT02D_inverseFT(components, kernel, width, height, output=...) -> output:
    """ """

@overload
def FT02D_iteration(matrix, kernel, mask, firstStop, output=..., maskOutput=...) -> tuple[retval, output, maskOutput]:
    """
    @brief Computes \f$F^0\f$-transfrom and inverse \f$F^0\f$-transfrom at once and return state.
        @param matrix Input matrix.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param output Output 32-bit float array.
        @param mask Mask used for unwanted area marking.
        @param maskOutput Mask after one iteration.
        @param firstStop If **true** function returns -1 when first problem appears. In case of `false` the process is completed and summation of all problems returned.
    """

@overload
def FT02D_iteration(matrix, kernel, mask, firstStop, output=..., maskOutput=...) -> tuple[retval, output, maskOutput]:
    """ """

@overload
def FT02D_process(matrix, kernel, output=..., mask=...) -> output:
    """
    @brief Computes \f$F^0\f$-transfrom and inverse \f$F^0\f$-transfrom at once.
        @param matrix Input matrix.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param output Output 32-bit float array.
        @param mask Mask used for unwanted area marking.
    """

@overload
def FT02D_process(matrix, kernel, output=..., mask=...) -> output:
    """ """

@overload
def FT12D_components(matrix, kernel, components=...) -> components:
    """
    @brief Computes components of the array using direct \f$F^1\f$-transform.
        @param matrix Input array.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param components Output 32-bit float array for the components.
    """

@overload
def FT12D_components(matrix, kernel, components=...) -> components:
    """ """

@overload
def FT12D_createPolynomMatrixHorizontal(radius, chn, matrix=...) -> matrix:
    """
    @brief Creates horizontal matrix for \f$F^1\f$-transform computation.
        @param radius Radius of the basic function.
        @param matrix The horizontal matrix.
        @param chn Number of channels.
    """

@overload
def FT12D_createPolynomMatrixHorizontal(radius, chn, matrix=...) -> matrix:
    """ """

@overload
def FT12D_createPolynomMatrixVertical(radius, chn, matrix=...) -> matrix:
    """
    @brief Creates vertical matrix for \f$F^1\f$-transform computation.
        @param radius Radius of the basic function.
        @param matrix The vertical matrix.
        @param chn Number of channels.
    """

@overload
def FT12D_createPolynomMatrixVertical(radius, chn, matrix=...) -> matrix:
    """ """

@overload
def FT12D_inverseFT(components, kernel, width, height, output=...) -> output:
    """
    @brief Computes inverse \f$F^1\f$-transfrom.
        @param components Input 32-bit float single channel array for the components.
        @param kernel Kernel used for processing. The same kernel as for components computation must be used.
        @param output Output 32-bit float array.
        @param width Width of the output array.
        @param height Height of the output array.
    """

@overload
def FT12D_inverseFT(components, kernel, width, height, output=...) -> output:
    """ """

@overload
def FT12D_polynomial(matrix, kernel, c00=..., c10=..., c01=..., components=..., mask=...) -> tuple[c00, c10, c01, components]:
    """
    @brief Computes elements of \f$F^1\f$-transform components.
        @param matrix Input array.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param c00 Elements represent average color.
        @param c10 Elements represent average vertical gradient.
        @param c01 Elements represent average horizontal gradient.
        @param components Output 32-bit float array for the components.
        @param mask Mask can be used for unwanted area marking.
    """

@overload
def FT12D_polynomial(matrix, kernel, c00=..., c10=..., c01=..., components=..., mask=...) -> tuple[c00, c10, c01, components]:
    """ """

@overload
def FT12D_process(matrix, kernel, output=..., mask=...) -> output:
    """
    @brief Computes \f$F^1\f$-transfrom and inverse \f$F^1\f$-transfrom at once.
        @param matrix Input matrix.
        @param kernel Kernel used for processing. Function `ft::createKernel` can be used.
        @param output Output 32-bit float array.
        @param mask Mask used for unwanted area marking.
    """

@overload
def FT12D_process(matrix, kernel, output=..., mask=...) -> output:
    """

    @note
        F-transform technique of first degreee is described in paper @cite Vlas:FT.
    """

@overload
def createKernel(function, radius, chn, kernel=...) -> kernel:
    """
    @brief Creates kernel from general functions.
        @param function Function type could be one of the following:
            -   **LINEAR** Linear basic function.
        @param radius Radius of the basic function.
        @param kernel Final 32-bit kernel.
        @param chn Number of kernel channels.
    """

@overload
def createKernel(function, radius, chn, kernel=...) -> kernel:
    """ """

@overload
def createKernel1(A, B, chn, kernel=...) -> kernel:
    """
    @brief Creates kernel from basic functions.
        @param A Basic function used in axis **x**.
        @param B Basic function used in axis **y**.
        @param kernel Final 32-bit kernel derived from **A** and **B**.
        @param chn Number of kernel channels.
    """

@overload
def createKernel1(A, B, chn, kernel=...) -> kernel:
    """ """

@overload
def filter(image, kernel, output=...) -> output:
    """
    @brief Image filtering
        @param image Input image.
        @param kernel Final 32-bit kernel.
        @param output Output 32-bit image.
    """

@overload
def filter(image, kernel, output=...) -> output:
    """ """

@overload
def inpaint(image, mask, radius, function, algorithm, output=...) -> output:
    """
    @brief Image inpainting
        @param image Input image.
        @param mask Mask used for unwanted area marking.
        @param output Output 32-bit image.
        @param radius Radius of the basic function.
        @param function Function type could be one of the following:
            -   `ft::LINEAR` Linear basic function.
        @param algorithm Algorithm could be one of the following:
            -   `ft::ONE_STEP` One step algorithm.
            -   `ft::MULTI_STEP` This algorithm automaticaly increases radius of the basic function.
            -   `ft::ITERATIVE` Iterative algorithm running in more steps using partial computations.
    """

@overload
def inpaint(image, mask, radius, function, algorithm, output=...) -> output:
    """

    @note
        The algorithms are described in paper @cite Perf:rec.
    """

ITERATIVE: Final[int]
LINEAR: Final[int]
MULTI_STEP: Final[int]
ONE_STEP: Final[int]
SINUS: Final[int]
