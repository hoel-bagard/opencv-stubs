import builtins
from typing import Any, TypeAlias

points: TypeAlias = Any

retval: TypeAlias = Any

class WeChatQRCode(builtins.object):
    def detectAndDecode(self, img, points=...) -> tuple[retval, points]:
        """
        * @brief  Both detects and decodes QR code.
        * To simplify the usage, there is a only API: detectAndDecode
        *
        * @param img supports grayscale or color (BGR) image.
        * @param points optional output array of vertices of the found QR code quadrangle. Will be * empty if not found. * @return list of decoded string.
        """

    def getScaleFactor(self) -> retval:
        """"""

    def setScaleFactor(self, _scalingFactor) -> None:
        """
        * @brief set scale factor
        * QR code detector use neural network to detect QR.
        * Before running the neural network, the input image is pre-processed by scaling.
        * By default, the input image is scaled to an image with an area of 160000 pixels.
        * The scale factor allows to use custom scale the input image:
        * width = scaleFactor*width
        * height = scaleFactor*width
        *
        * scaleFactor valuse must be > 0 and <= 1, otherwise the scaleFactor value is set to -1
        * and use default scaled to an image with an area of 160000 pixels.
        """
