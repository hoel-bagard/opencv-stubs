__version__ = "0.0.1"

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import numpy.typing as npt

from .constants import *


TImg = TypeVar("TImg", np.uint8, np.float64)
TColor = TypeVar("TColor", tuple[int, int, int], int, tuple[float, float, float], float)
TPoint = tuple[int, int]


def addWeighted(src1: npt.NDArray[TImg], alpha: float, src2: npt.NDArray[TImg], beta: float, gamma: float, dst: npt.NDArray[TImg] = ..., dtype: int = ...) -> npt.NDArray[TImg]:
    'addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]]) -> dst\n.   @brief Calculates the weighted sum of two arrays.\n.   \n.   The function addWeighted calculates the weighted sum of two arrays as follows:\n.   \\f[\\texttt{dst} (I)= \\texttt{saturate} ( \\texttt{src1} (I)* \\texttt{alpha} +  \\texttt{src2} (I)* \\texttt{beta} +  \\texttt{gamma} )\\f]\n.   where I is a multi-dimensional index of array elements. In case of multi-channel arrays, each\n.   channel is processed independently.\n.   The function can be replaced with a matrix expression:\n.   @code{.cpp}\n.       dst = src1*alpha + src2*beta + gamma;\n.   @endcode\n.   @note Saturation is not applied when the output array has the depth CV_32S. You may even get\n.   result of an incorrect sign in the case of overflow.\n.   @param src1 first input array.\n.   @param alpha weight of the first array elements.\n.   @param src2 second input array of the same size and channel number as src1.\n.   @param beta weight of the second array elements.\n.   @param gamma scalar added to each sum.\n.   @param dst output array that has the same size and number of channels as the input arrays.\n.   @param dtype optional depth of the output array; when both input arrays have the same depth, dtype\n.   can be set to -1, which will be equivalent to src1.depth().\n.   @sa  add, subtract, scaleAdd, Mat::convertTo'
    ...

def connectedComponentsWithStats(image: npt.NDArray[TImg], connectivity: int = ..., ltype: int = ...) -> tuple[int, npt.NDArray[np.int32], npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    'connectedComponentsWithStats(image[, labels[, stats[, centroids[, connectivity[, ltype]]]]]) -> retval, labels, stats, centroids\n.   @overload\n.   @param image the 8-bit single-channel image to be labeled\n.   @param labels destination labeled image\n.   @param stats statistics output for each label, including the background label.\n.   Statistics are accessed via stats(label, COLUMN) where COLUMN is one of\n.   #ConnectedComponentsTypes, selecting the statistic. The data type is CV_32S.\n.   @param centroids centroid output for each label, including the background label. Centroids are\n.   accessed via centroids(label, 0) for x and centroids(label, 1) for y. The data type CV_64F.\n.   @param connectivity 8 or 4 for 8-way or 4-way connectivity respectively\n.   @param ltype output image label type. Currently CV_32S and CV_16U are supported.'
    ...

def copyMakeBorder(src: npt.NDArray[TImg], top: int, bottom: int, left: int, right: int, borderType: int, dst:  npt.NDArray[TImg] = ..., value: TColor = ...) -> npt.NDArray[TImg]:
    'copyMakeBorder(src, top, bottom, left, right, borderType[, dst[, value]]) -> dst\n.   @brief Forms a border around an image.\n.   \n.   The function copies the source image into the middle of the destination image. The areas to the\n.   left, to the right, above and below the copied source image will be filled with extrapolated\n.   pixels. This is not what filtering functions based on it do (they extrapolate pixels on-fly), but\n.   what other more complex functions, including your own, may do to simplify image boundary handling.\n.   \n.   The function supports the mode when src is already in the middle of dst . In this case, the\n.   function does not copy src itself but simply constructs the border, for example:\n.   \n.   @code{.cpp}\n.       // let border be the same in all directions\n.       int border=2;\n.       // constructs a larger image to fit both the image and the border\n.       Mat gray_buf(rgb.rows + border*2, rgb.cols + border*2, rgb.depth());\n.       // select the middle part of it w/o copying data\n.       Mat gray(gray_canvas, Rect(border, border, rgb.cols, rgb.rows));\n.       // convert image from RGB to grayscale\n.       cvtColor(rgb, gray, COLOR_RGB2GRAY);\n.       // form a border in-place\n.       copyMakeBorder(gray, gray_buf, border, border,\n.                      border, border, BORDER_REPLICATE);\n.       // now do some custom filtering ...\n.       ...\n.   @endcode\n.   @note When the source image is a part (ROI) of a bigger image, the function will try to use the\n.   pixels outside of the ROI to form a border. To disable this feature and always do extrapolation, as\n.   if src was not a ROI, use borderType | #BORDER_ISOLATED.\n.   \n.   @param src Source image.\n.   @param dst Destination image of the same type as src and the size Size(src.cols+left+right,\n.   src.rows+top+bottom) .\n.   @param top the top pixels\n.   @param bottom the bottom pixels\n.   @param left the left pixels\n.   @param right Parameter specifying how many pixels in each direction from the source image rectangle\n.   to extrapolate. For example, top=1, bottom=1, left=1, right=1 mean that 1 pixel-wide border needs\n.   to be built.\n.   @param borderType Border type. See borderInterpolate for details.\n.   @param value Border value if borderType==BORDER_CONSTANT .\n.   \n.   @sa  borderInterpolate'
    ...

def circle(img: npt.NDArray[TImg], center: tuple[int, int], radius: int, color: int | tuple[int, int, int], thickness: int = ..., lineType: int = ..., shift: int = ...) -> npt.NDArray[TImg]:
    'circle(img, center, radius, color[, thickness[, lineType[, shift]]]) -> img\n.   @brief Draws a circle.\n.   \n.   The function cv::circle draws a simple or filled circle with a given center and radius.\n.   @param img Image where the circle is drawn.\n.   @param center Center of the circle.\n.   @param radius Radius of the circle.\n.   @param color Circle color.\n.   @param thickness Thickness of the circle outline, if positive. Negative values, like #FILLED,\n.   mean that a filled circle is to be drawn.\n.   @param lineType Type of the circle boundary. See #LineTypes\n.   @param shift Number of fractional bits in the coordinates of the center and in the radius value.'
    ...

def cvtColor(src: npt.NDArray[TImg], code: int, dst: npt.NDArray[TImg] = ..., dstCn: int = ...) -> npt.NDArray[TImg]:
    'cvtColor(src, code[, dst[, dstCn]]) -> dst\n.   @brief Converts an image from one color space to another.\n.   \n.   The function converts an input image from one color space to another. In case of a transformation\n.   to-from RGB color space, the order of the channels should be specified explicitly (RGB or BGR). Note\n.   that the default color format in OpenCV is often referred to as RGB but it is actually BGR (the\n.   bytes are reversed). So the first byte in a standard (24-bit) color image will be an 8-bit Blue\n.   component, the second byte will be Green, and the third byte will be Red. The fourth, fifth, and\n.   sixth bytes would then be the second pixel (Blue, then Green, then Red), and so on.\n.   \n.   The conventional ranges for R, G, and B channel values are:\n.   -   0 to 255 for CV_8U images\n.   -   0 to 65535 for CV_16U images\n.   -   0 to 1 for CV_32F images\n.   \n.   In case of linear transformations, the range does not matter. But in case of a non-linear\n.   transformation, an input RGB image should be normalized to the proper value range to get the correct\n.   results, for example, for RGB \\f$\\rightarrow\\f$ L\\*u\\*v\\* transformation. For example, if you have a\n.   32-bit floating-point image directly converted from an 8-bit image without any scaling, then it will\n.   have the 0..255 value range instead of 0..1 assumed by the function. So, before calling #cvtColor ,\n.   you need first to scale the image down:\n.   @code\n.       img *= 1./255;\n.       cvtColor(img, img, COLOR_BGR2Luv);\n.   @endcode\n.   If you use #cvtColor with 8-bit images, the conversion will have some information lost. For many\n.   applications, this will not be noticeable but it is recommended to use 32-bit images in applications\n.   that need the full range of colors or that convert an image before an operation and then convert\n.   back.\n.   \n.   If conversion adds the alpha channel, its value will set to the maximum of corresponding channel\n.   range: 255 for CV_8U, 65535 for CV_16U, 1 for CV_32F.\n.   \n.   @param src input image: 8-bit unsigned, 16-bit unsigned ( CV_16UC... ), or single-precision\n.   floating-point.\n.   @param dst output image of the same size and depth as src.\n.   @param code color space conversion code (see #ColorConversionCodes).\n.   @param dstCn number of channels in the destination image; if the parameter is 0, the number of the\n.   channels is derived automatically from src and code.\n.   \n.   @see @ref imgproc_color_conversions'
    ...

def destroyAllWindows() -> None:
    'destroyAllWindows() -> None\n.   @brief Destroys all of the HighGUI windows.\n.   \n.   The function destroyAllWindows destroys all of the opened HighGUI windows.'
    ...

def dilate(src: npt.NDArray[TImg], dts: npt.NDArray[TImg] = ..., anchor: TPoint = ..., iterations: int = ..., borderType: int = ..., borderValue: TColor = ...) -> npt.NDArray[TImg]:
    "dilate(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst\n.   @brief Dilates an image by using a specific structuring element.\n.   \n.   The function dilates the source image using the specified structuring element that determines the\n.   shape of a pixel neighborhood over which the maximum is taken:\n.   \\f[\\texttt{dst} (x,y) =  \\max _{(x',y'):  \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\f]\n.   \n.   The function supports the in-place mode. Dilation can be applied several ( iterations ) times. In\n.   case of multi-channel images, each channel is processed independently.\n.   \n.   @param src input image; the number of channels can be arbitrary, but the depth should be one of\n.   CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n.   @param dst output image of the same size and type as src.\n.   @param kernel structuring element used for dilation; if elemenat=Mat(), a 3 x 3 rectangular\n.   structuring element is used. Kernel can be created using #getStructuringElement\n.   @param anchor position of the anchor within the element; default value (-1, -1) means that the\n.   anchor is at the element center.\n.   @param iterations number of times dilation is applied.\n.   @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not suported.\n.   @param borderValue border value in case of a constant border\n.   @sa  erode, morphologyEx, getStructuringElement"
    ...

def erode(src: npt.NDArray[TImg], kernel: npt.NDArray[np.uint8], dts: npt.NDArray[TImg] = ..., anchor: TPoint = ..., iterations: int = ..., borderType: int = ..., borderValue: TColor = ...) -> npt.NDArray[TImg]:
    "erode(src, kernel[, dst[, anchor[, iterations[, borderType[, borderValue]]]]]) -> dst\n.   @brief Erodes an image by using a specific structuring element.\n.   \n.   The function erodes the source image using the specified structuring element that determines the\n.   shape of a pixel neighborhood over which the minimum is taken:\n.   \n.   \\f[\\texttt{dst} (x,y) =  \\min _{(x',y'):  \\, \\texttt{element} (x',y') \\ne0 } \\texttt{src} (x+x',y+y')\\f]\n.   \n.   The function supports the in-place mode. Erosion can be applied several ( iterations ) times. In\n.   case of multi-channel images, each channel is processed independently.\n.   \n.   @param src input image; the number of channels can be arbitrary, but the depth should be one of\n.   CV_8U, CV_16U, CV_16S, CV_32F or CV_64F.\n.   @param dst output image of the same size and type as src.\n.   @param kernel structuring element used for erosion; if `element=Mat()`, a `3 x 3` rectangular\n.   structuring element is used. Kernel can be created using #getStructuringElement.\n.   @param anchor position of the anchor within the element; default value (-1, -1) means that the\n.   anchor is at the element center.\n.   @param iterations number of times erosion is applied.\n.   @param borderType pixel extrapolation method, see #BorderTypes. #BORDER_WRAP is not supported.\n.   @param borderValue border value in case of a constant border\n.   @sa  dilate, morphologyEx, getStructuringElement"
    ...

def getTextSize(text: str, fontFace: int, fontScale: float, thickness: int) -> tuple[tuple[int, int], int]:
    'getTextSize(text, fontFace, fontScale, thickness) -> retval, baseLine\n.   @brief Calculates the width and height of a text string.\n.   \n.   The function cv::getTextSize calculates and returns the size of a box that contains the specified text.\n.   That is, the following code renders some text, the tight box surrounding it, and the baseline: :\n.   @code\n.       String text = "Funny text inside the box";\n.       int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;\n.       double fontScale = 2;\n.       int thickness = 3;\n.   \n.       Mat img(600, 800, CV_8UC3, Scalar::all(0));\n.   \n.       int baseline=0;\n.       Size textSize = getTextSize(text, fontFace,\n.                                   fontScale, thickness, &baseline);\n.       baseline += thickness;\n.   \n.       // center the text\n.       Point textOrg((img.cols - textSize.width)/2,\n.                     (img.rows + textSize.height)/2);\n.   \n.       // draw the box\n.       rectangle(img, textOrg + Point(0, baseline),\n.                 textOrg + Point(textSize.width, -textSize.height),\n.                 Scalar(0,0,255));\n.       // ... and the baseline first\n.       line(img, textOrg + Point(0, thickness),\n.            textOrg + Point(textSize.width, thickness),\n.            Scalar(0, 0, 255));\n.   \n.       // then put the text itself\n.       putText(img, text, textOrg, fontFace, fontScale,\n.               Scalar::all(255), thickness, 8);\n.   @endcode\n.   \n.   @param text Input text string.\n.   @param fontFace Font to use, see #HersheyFonts.\n.   @param fontScale Font scale factor that is multiplied by the font-specific base size.\n.   @param thickness Thickness of lines used to render the text. See #putText for details.\n.   @param[out] baseLine y-coordinate of the baseline relative to the bottom-most text\n.   point.\n.   @return The size of a box that contains the specified text.\n.   \n.   @see putText'
    ...

def hconcat(src: tuple[npt.NDArray[TImg], ...], dts: npt.NDArray[TImg] = ...) -> npt.NDArray[TImg]:
    'hconcat(src[, dst]) -> dst\n.   @overload\n.    @code{.cpp}\n.       std::vector<cv::Mat> matrices = { cv::Mat(4, 1, CV_8UC1, cv::Scalar(1)),\n.                                         cv::Mat(4, 1, CV_8UC1, cv::Scalar(2)),\n.                                         cv::Mat(4, 1, CV_8UC1, cv::Scalar(3)),};\n.   \n.       cv::Mat out;\n.       cv::hconcat( matrices, out );\n.       //out:\n.       //[1, 2, 3;\n.       // 1, 2, 3;\n.       // 1, 2, 3;\n.       // 1, 2, 3]\n.    @endcode\n.    @param src input array or vector of matrices. all of the matrices must have the same number of rows and the same depth.\n.    @param dst output array. It has the same number of rows and depth as the src, and the sum of cols of the src.\n.   same depth.'
    ...

def imdecode(buf: npt.NDArray[np.uint8], flags: int) -> npt.NDArray[np.uint8]:
    'imdecode(buf, flags) -> retval\n.   @brief Reads an image from a buffer in memory.\n.   \n.   The function imdecode reads an image from the specified buffer in the memory. If the buffer is too short or\n.   contains invalid data, the function returns an empty matrix ( Mat::data==NULL ).\n.   \n.   See cv::imread for the list of supported formats and flags description.\n.   \n.   @note In the case of color images, the decoded images will have the channels stored in **B G R** order.\n.   @param buf Input array or vector of bytes.\n.   @param flags The same flags as in cv::imread, see cv::ImreadModes.'
    ...

def imencode(ext: str, img: npt.NDArray[TImg], params: tuple[int, ...] = ...) -> tuple[bool, npt.NDArray[np.uint8]]:
    'imencode(ext, img[, params]) -> retval, buf\n.   @brief Encodes an image into a memory buffer.\n.   \n.   The function imencode compresses the image and stores it in the memory buffer that is resized to fit the\n.   result. See cv::imwrite for the list of supported formats and flags description.\n.   \n.   @param ext File extension that defines the output format.\n.   @param img Image to be written.\n.   @param buf Output buffer resized to fit the compressed image.\n.   @param params Format-specific parameters. See cv::imwrite and cv::ImwriteFlags.'
    ...

def imread(filename: str, flags: int = ...) -> npt.NDArray[np.uint8]:
    'imread(filename[, flags]) -> retval\n.   @brief Loads an image from a file.\n.   \n.   @anchor imread\n.   \n.   The function imread loads an image from the specified file and returns it. If the image cannot be\n.   read (because of missing file, improper permissions, unsupported or invalid format), the function\n.   returns an empty matrix ( Mat::data==NULL ).\n.   \n.   Currently, the following file formats are supported:\n.   \n.   -   Windows bitmaps - \\*.bmp, \\*.dib (always supported)\n.   -   JPEG files - \\*.jpeg, \\*.jpg, \\*.jpe (see the *Note* section)\n.   -   JPEG 2000 files - \\*.jp2 (see the *Note* section)\n.   -   Portable Network Graphics - \\*.png (see the *Note* section)\n.   -   WebP - \\*.webp (see the *Note* section)\n.   -   Portable image format - \\*.pbm, \\*.pgm, \\*.ppm \\*.pxm, \\*.pnm (always supported)\n.   -   PFM files - \\*.pfm (see the *Note* section)\n.   -   Sun rasters - \\*.sr, \\*.ras (always supported)\n.   -   TIFF files - \\*.tiff, \\*.tif (see the *Note* section)\n.   -   OpenEXR Image files - \\*.exr (see the *Note* section)\n.   -   Radiance HDR - \\*.hdr, \\*.pic (always supported)\n.   -   Raster and Vector geospatial data supported by GDAL (see the *Note* section)\n.   \n.   @note\n.   -   The function determines the type of an image by the content, not by the file extension.\n.   -   In the case of color images, the decoded images will have the channels stored in **B G R** order.\n.   -   When using IMREAD_GRAYSCALE, the codec\'s internal grayscale conversion will be used, if available.\n.       Results may differ to the output of cvtColor()\n.   -   On Microsoft Windows\\* OS and MacOSX\\*, the codecs shipped with an OpenCV image (libjpeg,\n.       libpng, libtiff, and libjasper) are used by default. So, OpenCV can always read JPEGs, PNGs,\n.       and TIFFs. On MacOSX, there is also an option to use native MacOSX image readers. But beware\n.       that currently these native image loaders give images with different pixel values because of\n.       the color management embedded into MacOSX.\n.   -   On Linux\\*, BSD flavors and other Unix-like open-source operating systems, OpenCV looks for\n.       codecs supplied with an OS image. Install the relevant packages (do not forget the development\n.       files, for example, "libjpeg-dev", in Debian\\* and Ubuntu\\*) to get the codec support or turn\n.       on the OPENCV_BUILD_3RDPARTY_LIBS flag in CMake.\n.   -   In the case you set *WITH_GDAL* flag to true in CMake and @ref IMREAD_LOAD_GDAL to load the image,\n.       then the [GDAL](http://www.gdal.org) driver will be used in order to decode the image, supporting\n.       the following formats: [Raster](http://www.gdal.org/formats_list.html),\n.       [Vector](http://www.gdal.org/ogr_formats.html).\n.   -   If EXIF information is embedded in the image file, the EXIF orientation will be taken into account\n.       and thus the image will be rotated accordingly except if the flags @ref IMREAD_IGNORE_ORIENTATION\n.       or @ref IMREAD_UNCHANGED are passed.\n.   -   Use the IMREAD_UNCHANGED flag to keep the floating point values from PFM image.\n.   -   By default number of pixels must be less than 2^30. Limit can be set using system\n.       variable OPENCV_IO_MAX_IMAGE_PIXELS\n.   \n.   @param filename Name of file to be loaded.\n.   @param flags Flag that can take values of cv::ImreadModes'
    ...


def imshow(winname: str, mat: npt.NDArray[TImg]) -> None:
    'imshow(winname, mat) -> None\n.   @brief Displays an image in the specified window.\n.   \n.   The function imshow displays an image in the specified window. If the window was created with the\n.   cv::WINDOW_AUTOSIZE flag, the image is shown with its original size, however it is still limited by the screen resolution.\n.   Otherwise, the image is scaled to fit the window. The function may scale the image, depending on its depth:\n.   \n.   -   If the image is 8-bit unsigned, it is displayed as is.\n.   -   If the image is 16-bit unsigned or 32-bit integer, the pixels are divided by 256. That is, the\n.       value range [0,255\\*256] is mapped to [0,255].\n.   -   If the image is 32-bit or 64-bit floating-point, the pixel values are multiplied by 255. That is, the\n.       value range [0,1] is mapped to [0,255].\n.   \n.   If window was created with OpenGL support, cv::imshow also support ogl::Buffer , ogl::Texture2D and\n.   cuda::GpuMat as input.\n.   \n.   If the window was not created before this function, it is assumed creating a window with cv::WINDOW_AUTOSIZE.\n.   \n.   If you need to show an image that is bigger than the screen resolution, you will need to call namedWindow("", WINDOW_NORMAL) before the imshow.\n.   \n.   @note This function should be followed by cv::waitKey function which displays the image for specified\n.   milliseconds. Otherwise, it won\'t display the image. For example, **waitKey(0)** will display the window\n.   infinitely until any keypress (it is suitable for image display). **waitKey(25)** will display a frame\n.   for 25 ms, after which display will be automatically closed. (If you put it in a loop to read\n.   videos, it will display the video frame-by-frame)\n.   \n.   @note\n.   \n.   [__Windows Backend Only__] Pressing Ctrl+C will copy the image to the clipboard.\n.   \n.   [__Windows Backend Only__] Pressing Ctrl+S will show a dialog to save the image.\n.   \n.   @param winname Name of the window.\n.   @param mat Image to be shown.'
    ...

def imwrite(filename: str, img: npt.NDArray[TImg], params: Sequence[int] = ...) -> bool:
    "imwrite(filename, img[, params]) -> retval\n.   @brief Saves an image to a specified file.\n.   \n.   The function imwrite saves the image to the specified file. The image format is chosen based on the\n.   filename extension (see cv::imread for the list of extensions). In general, only 8-bit\n.   single-channel or 3-channel (with 'BGR' channel order) images\n.   can be saved using this function, with these exceptions:\n.   \n.   - 16-bit unsigned (CV_16U) images can be saved in the case of PNG, JPEG 2000, and TIFF formats\n.   - 32-bit float (CV_32F) images can be saved in PFM, TIFF, OpenEXR, and Radiance HDR formats;\n.     3-channel (CV_32FC3) TIFF images will be saved using the LogLuv high dynamic range encoding\n.     (4 bytes per pixel)\n.   - PNG images with an alpha channel can be saved using this function. To do this, create\n.   8-bit (or 16-bit) 4-channel image BGRA, where the alpha channel goes last. Fully transparent pixels\n.   should have alpha set to 0, fully opaque pixels should have alpha set to 255/65535 (see the code sample below).\n.   - Multiple images (vector of Mat) can be saved in TIFF format (see the code sample below).\n.   \n.   If the format, depth or channel order is different, use\n.   Mat::convertTo and cv::cvtColor to convert it before saving. Or, use the universal FileStorage I/O\n.   functions to save the image to XML or YAML format.\n.   \n.   The sample below shows how to create a BGRA image, how to set custom compression parameters and save it to a PNG file.\n.   It also demonstrates how to save multiple images in a TIFF file:\n.   @include snippets/imgcodecs_imwrite.cpp\n.   @param filename Name of the file.\n.   @param img (Mat or vector of Mat) Image or Images to be saved.\n.   @param params Format-specific parameters encoded as pairs (paramId_1, paramValue_1, paramId_2, paramValue_2, ... .) see cv::ImwriteFlags"
    ...

def line(img: npt.NDArray[TImg], pt1: tuple[int, int], pt2: tuple[int, int], color: tuple[int, int, int] | int, thickness: int = ..., lineType: int = ..., shift: int = ...) -> npt.NDArray[TImg]:
    'line(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n.   @brief Draws a line segment connecting two points.\n.   \n.   The function line draws the line segment between pt1 and pt2 points in the image. The line is\n.   clipped by the image boundaries. For non-antialiased lines with integer coordinates, the 8-connected\n.   or 4-connected Bresenham algorithm is used. Thick lines are drawn with rounding endings. Antialiased\n.   lines are drawn using Gaussian filtering.\n.   \n.   @param img Image.\n.   @param pt1 First point of the line segment.\n.   @param pt2 Second point of the line segment.\n.   @param color Line color.\n.   @param thickness Line thickness.\n.   @param lineType Type of the line. See #LineTypes.\n.   @param shift Number of fractional bits in the point coordinates.'
    ...

def namedWindow(winname: str, flags: int = ...) -> None:
    'namedWindow(winname[, flags]) -> None\n.   @brief Creates a window.\n.   \n.   The function namedWindow creates a window that can be used as a placeholder for images and\n.   trackbars. Created windows are referred to by their names.\n.   \n.   If a window with the same name already exists, the function does nothing.\n.   \n.   You can call cv::destroyWindow or cv::destroyAllWindows to close the window and de-allocate any associated\n.   memory usage. For a simple program, you do not really have to call these functions because all the\n.   resources and windows of the application are closed automatically by the operating system upon exit.\n.   \n.   @note\n.   \n.   Qt backend supports additional flags:\n.    -   **WINDOW_NORMAL or WINDOW_AUTOSIZE:** WINDOW_NORMAL enables you to resize the\n.        window, whereas WINDOW_AUTOSIZE adjusts automatically the window size to fit the\n.        displayed image (see imshow ), and you cannot change the window size manually.\n.    -   **WINDOW_FREERATIO or WINDOW_KEEPRATIO:** WINDOW_FREERATIO adjusts the image\n.        with no respect to its ratio, whereas WINDOW_KEEPRATIO keeps the image ratio.\n.    -   **WINDOW_GUI_NORMAL or WINDOW_GUI_EXPANDED:** WINDOW_GUI_NORMAL is the old way to draw the window\n.        without statusbar and toolbar, whereas WINDOW_GUI_EXPANDED is a new enhanced GUI.\n.   By default, flags == WINDOW_AUTOSIZE | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED\n.   \n.   @param winname Name of the window in the window caption that may be used as a window identifier.\n.   @param flags Flags of the window. The supported flags are: (cv::WindowFlags)'
    ...

def putText(img: npt.NDArray[TImg], text: str, org: TPoint, fontFace: int, fontScale: float, color: TColor, thickness: int = ..., lineType: int = ..., bottomLeftOrigin: bool = ...) -> npt.NDArray[TImg]:
    'putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) -> img\n.   @brief Draws a text string.\n.   \n.   The function cv::putText renders the specified text string in the image. Symbols that cannot be rendered\n.   using the specified font are replaced by question marks. See #getTextSize for a text rendering code\n.   example.\n.   \n.   @param img Image.\n.   @param text Text string to be drawn.\n.   @param org Bottom-left corner of the text string in the image.\n.   @param fontFace Font type, see #HersheyFonts.\n.   @param fontScale Font scale factor that is multiplied by the font-specific base size.\n.   @param color Text color.\n.   @param thickness Thickness of the lines used to draw a text.\n.   @param lineType Line type. See #LineTypes\n.   @param bottomLeftOrigin When true, the image data origin is at the bottom-left corner. Otherwise,\n.   it is at the top-left corner.'
    ...

def rectangle(img: npt.NDArray[TImg], pt1: TPoint, pt2: TPoint, color: TColor, thickness: int = 1, lineType: int = LINE_8, shift: int = 0) -> npt.NDArray[TImg]:
    """rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]]) -> img\n.   @brief Draws a simple, thick, or filled up-right rectangle.\n.   \n.   The function cv::rectangle draws a rectangle outline or a filled rectangle whose two opposite corners\n.   are pt1 and pt2.\n.   \n.   @param img Image.\n.   @param pt1 Vertex of the rectangle.\n.   @param pt2 Vertex of the rectangle opposite to pt1 .\n.   @param color Rectangle color or brightness (grayscale image).\n.   @param thickness Thickness of lines that make up the rectangle. Negative values, like #FILLED,\n.   mean that the function has to draw a filled rectangle.\n.   @param lineType Type of the line. See #LineTypes\n.   @param shift Number of fractional bits in the point coordinates.\n\n\n\nrectangle(img, rec, color[, thickness[, lineType[, shift]]]) -> img\n.   @overload\n.   \n.   use `rec` parameter as alternative specification of the drawn rectangle: `r.tl() and\n.   r.br()-Point(1,1)` are opposite corners"""
    ...

def resize(src: npt.NDArray[TImg], dsize: tuple[int, int], dst: npt.NDArray[TImg] = ..., fx: int = ..., fy: int = ..., interpolation: int = ...) -> npt.NDArray[TImg]:
    'resize(src, dsize[, dst[, fx[, fy[, interpolation]]]]) -> dst\n.   @brief Resizes an image.\n.   \n.   The function resize resizes the image src down to or up to the specified size. Note that the\n.   initial dst type or size are not taken into account. Instead, the size and type are derived from\n.   the `src`,`dsize`,`fx`, and `fy`. If you want to resize src so that it fits the pre-created dst,\n.   you may call the function as follows:\n.   @code\n.       // explicitly specify dsize=dst.size(); fx and fy will be computed from that.\n.       resize(src, dst, dst.size(), 0, 0, interpolation);\n.   @endcode\n.   If you want to decimate the image by factor of 2 in each direction, you can call the function this\n.   way:\n.   @code\n.       // specify fx and fy and let the function compute the destination image size.\n.       resize(src, dst, Size(), 0.5, 0.5, interpolation);\n.   @endcode\n.   To shrink an image, it will generally look best with #INTER_AREA interpolation, whereas to\n.   enlarge an image, it will generally look best with c#INTER_CUBIC (slow) or #INTER_LINEAR\n.   (faster but still looks OK).\n.   \n.   @param src input image.\n.   @param dst output image; it has the size dsize (when it is non-zero) or the size computed from\n.   src.size(), fx, and fy; the type of dst is the same as of src.\n.   @param dsize output image size; if it equals zero, it is computed as:\n.    \\f[\\texttt{dsize = Size(round(fx*src.cols), round(fy*src.rows))}\\f]\n.    Either dsize or both fx and fy must be non-zero.\n.   @param fx scale factor along the horizontal axis; when it equals 0, it is computed as\n.   \\f[\\texttt{(double)dsize.width/src.cols}\\f]\n.   @param fy scale factor along the vertical axis; when it equals 0, it is computed as\n.   \\f[\\texttt{(double)dsize.height/src.rows}\\f]\n.   @param interpolation interpolation method, see #InterpolationFlags\n.   \n.   @sa  warpAffine, warpPerspective, remap'
    ...

def setWindowProperty(winname: str, prop_id:int , prop_value: int) -> None:
    'setWindowProperty(winname, prop_id, prop_value) -> None\n.   @brief Changes parameters of a window dynamically.\n.   \n.   The function setWindowProperty enables changing properties of a window.\n.   \n.   @param winname Name of the window.\n.   @param prop_id Window property to edit. The supported operation flags are: (cv::WindowPropertyFlags)\n.   @param prop_value New value of the window property. The supported flags are: (cv::WindowFlags)'
    ...

def threshold(src: npt.NDArray[TImg], thresh: float, maxval: float, type: int, dst: npt.NDArray[TImg] = ...) -> tuple[int, npt.NDArray[TImg]]:
    "threshold(src, thresh, maxval, type[, dst]) -> retval, dst\n.   @brief Applies a fixed-level threshold to each array element.\n.   \n.   The function applies fixed-level thresholding to a multiple-channel array. The function is typically\n.   used to get a bi-level (binary) image out of a grayscale image ( #compare could be also used for\n.   this purpose) or for removing a noise, that is, filtering out pixels with too small or too large\n.   values. There are several types of thresholding supported by the function. They are determined by\n.   type parameter.\n.   \n.   Also, the special values #THRESH_OTSU or #THRESH_TRIANGLE may be combined with one of the\n.   above values. In these cases, the function determines the optimal threshold value using the Otsu's\n.   or Triangle algorithm and uses it instead of the specified thresh.\n.   \n.   @note Currently, the Otsu's and Triangle methods are implemented only for 8-bit single-channel images.\n.   \n.   @param src input array (multiple-channel, 8-bit or 32-bit floating point).\n.   @param dst output array of the same size  and type and the same number of channels as src.\n.   @param thresh threshold value.\n.   @param maxval maximum value to use with the #THRESH_BINARY and #THRESH_BINARY_INV thresholding\n.   types.\n.   @param type thresholding type (see #ThresholdTypes).\n.   @return the computed threshold value if Otsu's or Triangle methods used.\n.   \n.   @sa  adaptiveThreshold, findContours, compare, min, max"
    ...

def UMat(img: npt.NDArray[TImg]) ->  npt.NDArray[TImg]:
    ...

def vconcat(src: tuple[npt.NDArray[TImg], ...], dts: npt.NDArray[TImg] = ...) -> npt.NDArray[TImg]:
    'vconcat(src[, dst]) -> dst\n.   @overload\n.    @code{.cpp}\n.       std::vector<cv::Mat> matrices = { cv::Mat(1, 4, CV_8UC1, cv::Scalar(1)),\n.                                         cv::Mat(1, 4, CV_8UC1, cv::Scalar(2)),\n.                                         cv::Mat(1, 4, CV_8UC1, cv::Scalar(3)),};\n.   \n.       cv::Mat out;\n.       cv::vconcat( matrices, out );\n.       //out:\n.       //[1,   1,   1,   1;\n.       // 2,   2,   2,   2;\n.       // 3,   3,   3,   3]\n.    @endcode\n.    @param src input array or vector of matrices. all of the matrices must have the same number of cols and the same depth\n.    @param dst output array. It has the same number of cols and depth as the src, and the sum of rows of the src.\n.   same depth.'
    ...

def waitKey(delay: int =...) -> int:
    'waitKey([, delay]) -> retval\n.   @brief Waits for a pressed key.\n.   \n.   The function waitKey waits for a key event infinitely (when \\f$\\texttt{delay}\\leq 0\\f$ ) or for delay\n.   milliseconds, when it is positive. Since the OS has a minimum time between switching threads, the\n.   function will not wait exactly delay ms, it will wait at least delay ms, depending on what else is\n.   running on your computer at that time. It returns the code of the pressed key or -1 if no key was\n.   pressed before the specified time had elapsed.\n.   \n.   @note\n.   \n.   This function is the only method in HighGUI that can fetch and handle events, so it needs to be\n.   called periodically for normal event processing unless HighGUI is used within an environment that\n.   takes care of event processing.\n.   \n.   @note\n.   \n.   The function only works if there is at least one HighGUI window created and the window is active.\n.   If there are several HighGUI windows, any of them can be active.\n.   \n.   @param delay Delay in milliseconds. 0 is the special value that means "forever".'
    ...
