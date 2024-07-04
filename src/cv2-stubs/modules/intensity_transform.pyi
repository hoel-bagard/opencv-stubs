from typing import Any, overload, TypeAlias

output: TypeAlias = Any

def BIMEF(input, output=..., mu=..., a=..., b=...) -> output:
    """
    * @brief Given an input color image, enhance low-light images using the BIMEF method (@cite ying2017bio @cite ying2017new).

    @param input input color image.
    @param output resulting image.
    @param mu enhancement ratio.
    @param a a-parameter in the Camera Response Function (CRF).
    @param b b-parameter in the Camera Response Function (CRF).

    @warning This is a C++ implementation of the [original MATLAB algorithm](https://github.com/baidut/BIMEF).
    Compared to the original code, this implementation is a little bit slower and does not provide the same results.
    In particular, quality of the image enhancement is degraded for the bright areas in certain conditions.
    """

def BIMEF2(input, k, mu, a, b, output=...) -> output:
    """
    * @brief Given an input color image, enhance low-light images using the BIMEF method (@cite ying2017bio @cite ying2017new).

    This is an overloaded function with the exposure ratio given as parameter.

    @param input input color image.
    @param output resulting image.
    @param k exposure ratio.
    @param mu enhancement ratio.
    @param a a-parameter in the Camera Response Function (CRF).
    @param b b-parameter in the Camera Response Function (CRF).

    @warning This is a C++ implementation of the [original MATLAB algorithm](https://github.com/baidut/BIMEF).
    Compared to the original code, this implementation is a little bit slower and does not provide the same results.
    In particular, quality of the image enhancement is degraded for the bright areas in certain conditions.
    """

def autoscaling(input, output) -> None:
    """
    * @brief Given an input bgr or grayscale image, apply autoscaling on domain [0, 255] to increase
    the contrast of the input image and return the resulting image.

    @param input input bgr or grayscale image.
    @param output resulting image of autoscaling.
    """

def contrastStretching(input, output, r1, s1, r2, s2) -> None:
    """
    * @brief Given an input bgr or grayscale image, apply linear contrast stretching on domain [0, 255]
    and return the resulting image.

    @param input input bgr or grayscale image.
    @param output resulting image of contrast stretching.
    @param r1 x coordinate of first point (r1, s1) in the transformation function.
    @param s1 y coordinate of first point (r1, s1) in the transformation function.
    @param r2 x coordinate of second point (r2, s2) in the transformation function.
    @param s2 y coordinate of second point (r2, s2) in the transformation function.
    """

def gammaCorrection(input, output, gamma) -> None:
    """
    * @brief Given an input bgr or grayscale image and constant gamma, apply power-law transformation,
    a.k.a. gamma correction to the image on domain [0, 255] and return the resulting image.

    @param input input bgr or grayscale image.
    @param output resulting image of gamma corrections.
    @param gamma constant in c*r^gamma where r is pixel value.
    """

def logTransform(input, output) -> None:
    """
    * @brief Given an input bgr or grayscale image and constant c, apply log transformation to the image
    on domain [0, 255] and return the resulting image.

    @param input input bgr or grayscale image.
    @param output resulting image of log transformations.
    """
