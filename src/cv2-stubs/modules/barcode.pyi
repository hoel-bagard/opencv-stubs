import builtins
from typing import Any, Final, TypeAlias

decoded_info: TypeAlias = Any
points: TypeAlias = Any
decoded_type: TypeAlias = Any
retval: TypeAlias = Any

class BarcodeDetector(builtins.object):
    def decode(self, img, points) -> tuple[retval, decoded_info, decoded_type]:
        """
        @brief Decodes barcode in image once it's found by the detect() method.
        *
        * @param img grayscale or color (BGR) image containing bar code.
        * @param points vector of rotated rectangle vertices found by detect() method (or some other algorithm). * For N detected barcodes, the dimensions of this array should be [N][4]. * Order of four points in vector<Point2f> is bottomLeft, topLeft, topRight, bottomRight.
        * @param decoded_info UTF8-encoded output vector of string or empty vector of string if the codes cannot be decoded.
        * @param decoded_type vector of BarcodeType, specifies the type of these barcodes
        """

    def detect(self, img, points=...) -> tuple[retval, points]:
        """
        @brief Detects Barcode in image and returns the rectangle(s) containing the code.
        *
        * @param img grayscale or color (BGR) image containing (or not) Barcode.
        * @param points Output vector of vector of vertices of the minimum-area rotated rectangle containing the codes. * For N detected barcodes, the dimensions of this array should be [N][4]. * Order of four points in vector< Point2f> is bottomLeft, topLeft, topRight, bottomRight.
        """

    def detectAndDecode(self, img, points=...) -> tuple[retval, decoded_info, decoded_type, points]:
        """
        @brief Both detects and decodes barcode

        * @param img grayscale or color (BGR) image containing barcode.
        * @param decoded_info UTF8-encoded output vector of string(s) or empty vector of string if the codes cannot be decoded.
        * @param decoded_type vector of BarcodeType, specifies the type of these barcodes
        * @param points optional output vector of vertices of the found  barcode rectangle. Will be empty if not found.
        """

EAN_13: Final[int]
EAN_8: Final[int]
NONE: Final[int]
UPC_A: Final[int]
UPC_E: Final[int]
UPC_EAN_EXTENSION: Final[int]
