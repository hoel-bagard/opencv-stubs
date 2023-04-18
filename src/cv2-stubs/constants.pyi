# BorderTypes
BORDER_CONSTANT: int
BORDER_REPLICATE: int
BORDER_REFLECT: int
BORDER_WRAP: int
BORDER_REFLECT_101: int
BORDER_TRANSPARENT: int
BORDER_REFLECT101: int
BORDER_DEFAULT: int
BORDER_ISOLATED: int

# ContourApproximationModes
CHAIN_APPROX_NONE: int
CHAIN_APPROX_SIMPLE: int
CHAIN_APPROX_TC89_L1: int
CHAIN_APPROX_TC89_KCOS: int

# Color conversion codes
COLOR_BGR2BGRA: int  # Add alpha channel to RGB or BGR image
COLOR_RGB2RGBA: int
COLOR_BGRA2BGR: int  # Remove alpha channel from RGB or BGR image
COLOR_RGBA2RGB: int
COLOR_BGR2RGBA: int  # Convert between RGB and BGR color spaces (with or without alpha channel)
COLOR_RGB2BGRA: int
COLOR_RGBA2BGR: int
COLOR_BGRA2RGB: int
COLOR_BGR2RGB: int
COLOR_RGB2BGR: int
COLOR_BGRA2RGBA: int
COLOR_RGBA2BGRA: int
COLOR_BGR2GRAY: int  # Convert between RGB/BGR and grayscale, color conversions
COLOR_RGB2GRAY: int
COLOR_GRAY2BGR: int
COLOR_GRAY2RGB: int
COLOR_GRAY2BGRA: int
COLOR_GRAY2RGBA: int
COLOR_BGRA2GRAY: int
COLOR_RGBA2GRAY: int
COLOR_BGR2BGR565: int  # Convert between RGB/BGR and BGR565 (16-bit images)
COLOR_RGB2BGR565: int
COLOR_BGR5652BGR: int
COLOR_BGR5652RGB: int
COLOR_BGRA2BGR565: int
COLOR_RGBA2BGR565: int
COLOR_BGR5652BGRA: int
COLOR_BGR5652RGBA: int
COLOR_GRAY2BGR565: int  # Convert between grayscale to BGR565 (16-bit images)
COLOR_BGR5652GRAY: int
COLOR_BGR2BGR555: int  # Convert between RGB/BGR and BGR555 (16-bit images)
COLOR_RGB2BGR555: int
COLOR_BGR5552BGR: int
COLOR_BGR5552RGB: int
COLOR_BGRA2BGR555: int
COLOR_RGBA2BGR555: int
COLOR_BGR5552BGRA: int
COLOR_BGR5552RGBA: int
COLOR_GRAY2BGR555: int  # Convert between grayscale and BGR555 (16-bit images)
COLOR_BGR5552GRAY: int
COLOR_BGR2XYZ: int  # Convert RGB/BGR to CIE XYZ, color conversions
COLOR_RGB2XYZ: int
COLOR_XYZ2BGR: int
COLOR_XYZ2RGB: int
COLOR_BGR2YCrCb: int  # Convert RGB/BGR to luma-chroma (aka YCC), color conversions
COLOR_RGB2YCrCb: int
COLOR_YCrCb2BGR: int
COLOR_YCrCb2RGB: int
COLOR_BGR2HSV: int  # Convert RGB/BGR to HSV (hue saturation value) with H range 0..180 if 8 bit image, color conversions
COLOR_RGB2HSV: int
COLOR_BGR2Lab: int  # Convert RGB/BGR to CIE Lab, color conversions
COLOR_RGB2Lab: int
COLOR_BGR2Luv: int  # Convert RGB/BGR to CIE Luv, color conversions
COLOR_RGB2Luv: int
COLOR_BGR2HLS: int  # Convert RGB/BGR to HLS (hue lightness saturation) with H range 0..180 if 8 bit image, color conversions
COLOR_RGB2HLS: int
COLOR_HSV2BGR: int  # backward conversions HSV to RGB/BGR with H range 0..180 if 8 bit image
COLOR_HSV2RGB: int
COLOR_Lab2BGR: int
COLOR_Lab2RGB: int
COLOR_Luv2BGR: int
COLOR_Luv2RGB: int
COLOR_HLS2BGR: int  # backward conversions HLS to RGB/BGR with H range 0..180 if 8 bit image
COLOR_HLS2RGB: int
COLOR_BGR2HSV_FULL: int  # Convert RGB/BGR to HSV (hue saturation value) with H range 0..255 if 8 bit image, color conversions
COLOR_RGB2HSV_FULL: int
COLOR_BGR2HLS_FULL: int  # Convert RGB/BGR to HLS (hue lightness saturation) with H range 0..255 if 8 bit image, color conversions
COLOR_RGB2HLS_FULL: int
COLOR_HSV2BGR_FULL: int  # Backward conversions HSV to RGB/BGR with H range 0..255 if 8 bit image
COLOR_HSV2RGB_FULL: int
COLOR_HLS2BGR_FULL: int  # Backward conversions HLS to RGB/BGR with H range 0..255 if 8 bit image
COLOR_HLS2RGB_FULL: int
COLOR_LBGR2Lab: int
COLOR_LRGB2Lab: int
COLOR_LBGR2Luv: int
COLOR_LRGB2Luv: int
COLOR_Lab2LBGR: int
COLOR_Lab2LRGB: int
COLOR_Luv2LBGR: int
COLOR_Luv2LRGB: int
COLOR_BGR2YUV: int  # Convert between RGB/BGR and YUV
COLOR_RGB2YUV: int
COLOR_YUV2BGR: int
COLOR_YUV2RGB: int
COLOR_YUV2RGB_NV12: int  # YUV 4:2:0 family to RGB.
COLOR_YUV2BGR_NV12: int
COLOR_YUV2RGB_NV21: int
COLOR_YUV2BGR_NV21: int
COLOR_YUV420sp2RGB: int
COLOR_YUV420sp2BGR: int
COLOR_YUV2RGBA_NV12: int
COLOR_YUV2BGRA_NV12: int
COLOR_YUV2RGBA_NV21: int
COLOR_YUV2BGRA_NV21: int
COLOR_YUV420sp2RGBA: int
COLOR_YUV420sp2BGRA: int
COLOR_YUV2RGB_YV12: int
COLOR_YUV2BGR_YV12: int
COLOR_YUV2RGB_IYUV: int
COLOR_YUV2BGR_IYUV: int
COLOR_YUV2RGB_I420: int
COLOR_YUV2BGR_I420: int
COLOR_YUV420p2RGB: int
COLOR_YUV420p2BGR: int
COLOR_YUV2RGBA_YV12: int
COLOR_YUV2BGRA_YV12: int
COLOR_YUV2RGBA_IYUV: int
COLOR_YUV2BGRA_IYUV: int
COLOR_YUV2RGBA_I420: int
COLOR_YUV2BGRA_I420: int
COLOR_YUV420p2RGBA: int
COLOR_YUV420p2BGRA: int
COLOR_YUV2GRAY_420: int
COLOR_YUV2GRAY_NV21: int
COLOR_YUV2GRAY_NV12: int
COLOR_YUV2GRAY_YV12: int
COLOR_YUV2GRAY_IYUV: int
COLOR_YUV2GRAY_I420: int
COLOR_YUV420sp2GRAY: int
COLOR_YUV420p2GRAY: int
COLOR_YUV2RGB_UYVY: int  #  YUV 4:2:2 family to RGB.
COLOR_YUV2BGR_UYVY: int
COLOR_YUV2RGB_Y422: int
COLOR_YUV2BGR_Y422: int
COLOR_YUV2RGB_UYNV: int
COLOR_YUV2BGR_UYNV: int
COLOR_YUV2RGBA_UYVY: int
COLOR_YUV2BGRA_UYVY: int
COLOR_YUV2RGBA_Y422: int
COLOR_YUV2BGRA_Y422: int
COLOR_YUV2RGBA_UYNV: int
COLOR_YUV2BGRA_UYNV: int
COLOR_YUV2RGB_YUY2: int
COLOR_YUV2BGR_YUY2: int
COLOR_YUV2RGB_YVYU: int
COLOR_YUV2BGR_YVYU: int
COLOR_YUV2RGB_YUYV: int
COLOR_YUV2BGR_YUYV: int
COLOR_YUV2RGB_YUNV: int
COLOR_YUV2BGR_YUNV: int
COLOR_YUV2RGBA_YUY2: int
COLOR_YUV2BGRA_YUY2: int
COLOR_YUV2RGBA_YVYU: int
COLOR_YUV2BGRA_YVYU: int
COLOR_YUV2RGBA_YUYV: int
COLOR_YUV2BGRA_YUYV: int
COLOR_YUV2RGBA_YUNV: int
COLOR_YUV2BGRA_YUNV: int
COLOR_YUV2GRAY_UYVY: int
COLOR_YUV2GRAY_YUY2: int
COLOR_YUV2GRAY_Y422: int
COLOR_YUV2GRAY_UYNV: int
COLOR_YUV2GRAY_YVYU: int
COLOR_YUV2GRAY_YUYV: int
COLOR_YUV2GRAY_YUNV: int
COLOR_RGBA2mRGBA: int  # Ulpha premultiplication
COLOR_mRGBA2RGBA: int
COLOR_RGB2YUV_I420: int  # RGB to YUV 4:2:0 family.
COLOR_BGR2YUV_I420: int
COLOR_RGB2YUV_IYUV: int
COLOR_BGR2YUV_IYUV: int
COLOR_RGBA2YUV_I420: int
COLOR_BGRA2YUV_I420: int
COLOR_RGBA2YUV_IYUV: int
COLOR_BGRA2YUV_IYUV: int
COLOR_RGB2YUV_YV12: int
COLOR_BGR2YUV_YV12: int
COLOR_RGBA2YUV_YV12: int
COLOR_BGRA2YUV_YV12: int
COLOR_BayerBG2BGR: int  # Demosaicing, see color conversions (https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_bayer) for additional information. Equivalent to RGGB Bayer pattern.
COLOR_BayerGB2BGR: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2BGR: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2BGR: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerRGGB2BGR: int
COLOR_BayerGRBG2BGR: int
COLOR_BayerBGGR2BGR: int
COLOR_BayerGBRG2BGR: int
COLOR_BayerRGGB2RGB: int
COLOR_BayerGRBG2RGB: int
COLOR_BayerBGGR2RGB: int
COLOR_BayerGBRG2RGB: int
COLOR_BayerBG2RGB: int  # Equivalent to RGGB Bayer pattern
COLOR_BayerGB2RGB: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2RGB: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2RGB: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerBG2GRAY: int  # Equivalent to RGGB Bayer pattern
COLOR_BayerGB2GRAY: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2GRAY: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2GRAY: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerRGGB2GRAY: int
COLOR_BayerGRBG2GRAY: int
COLOR_BayerBGGR2GRAY: int
COLOR_BayerGBRG2GRAY: int
COLOR_BayerBG2BGR_VNG: int  # Demosaicing using Variable Number of Gradients. equivalent to RGGB Bayer pattern
COLOR_BayerGB2BGR_VNG: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2BGR_VNG: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2BGR_VNG: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerRGGB2BGR_VNG: int
COLOR_BayerGRBG2BGR_VNG: int
COLOR_BayerBGGR2BGR_VNG: int
COLOR_BayerGBRG2BGR_VNG: int
COLOR_BayerRGGB2RGB_VNG: int
COLOR_BayerGRBG2RGB_VNG: int
COLOR_BayerBGGR2RGB_VNG: int
COLOR_BayerGBRG2RGB_VNG: int
COLOR_BayerBG2RGB_VNG: int  # Equivalent to RGGB Bayer pattern
COLOR_BayerGB2RGB_VNG: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2RGB_VNG: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2RGB_VNG: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerBG2BGR_EA: int  # Edge-Aware Demosaicing. equivalent to RGGB Bayer pattern
COLOR_BayerGB2BGR_EA: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2BGR_EA: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2BGR_EA: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerRGGB2BGR_EA: int
COLOR_BayerGRBG2BGR_EA: int
COLOR_BayerBGGR2BGR_EA: int
COLOR_BayerGBRG2BGR_EA: int
COLOR_BayerRGGB2RGB_EA: int
COLOR_BayerGRBG2RGB_EA: int
COLOR_BayerBGGR2RGB_EA: int
COLOR_BayerGBRG2RGB_EA: int
COLOR_BayerBG2RGB_EA: int  # Equivalent to RGGB Bayer pattern
COLOR_BayerGB2RGB_EA: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2RGB_EA: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2RGB_EA: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerBG2BGRA: int  # Demosaicing with alpha channel. equivalent to RGGB Bayer pattern
COLOR_BayerGB2BGRA: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2BGRA: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2BGRA: int  # Equivalent to GBRG Bayer pattern
COLOR_BayerRGGB2BGRA: int
COLOR_BayerGRBG2BGRA: int
COLOR_BayerBGGR2BGRA: int
COLOR_BayerGBRG2BGRA: int
COLOR_BayerRGGB2RGBA: int
COLOR_BayerGRBG2RGBA: int
COLOR_BayerBGGR2RGBA: int
COLOR_BayerGBRG2RGBA: int
COLOR_BayerBG2RGBA: int  # Equivalent to RGGB Bayer pattern
COLOR_BayerGB2RGBA: int  # Equivalent to GRBG Bayer pattern
COLOR_BayerRG2RGBA: int  # Equivalent to BGGR Bayer pattern
COLOR_BayerGR2RGBA: int  # Equivalent to GBRG Bayer pattern
COLOR_COLORCVT_MAX: int

# ConnectedComponentsTypes
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5
CC_STAT_LEFT: int  # The leftmost (x) coordinate which is the inclusive start of the bounding box in the horizontal direction.
CC_STAT_TOP: int  # The topmost (y) coordinate which is the inclusive start of the bounding box in the vertical direction.
CC_STAT_WIDTH: int  # The horizontal size of the bounding box.
CC_STAT_HEIGHT: int  # The vertical size of the bounding box.
CC_STAT_AREA: int  # The total area (in pixels) of the connected component.

# Fonts
FONT_HERSHEY_COMPLEX: int
FONT_HERSHEY_COMPLEX_SMALL: int
FONT_HERSHEY_DUPLEX: int
FONT_HERSHEY_PLAIN: int
FONT_HERSHEY_SCRIPT_COMPLEX: int
FONT_HERSHEY_SCRIPT_SIMPLEX: int
FONT_HERSHEY_SIMPLEX: int
FONT_HERSHEY_TRIPLEX: int
FONT_ITALIC: int

# MorphShapes
# https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#gac2db39b56866583a95a5680313c314ad
MORPH_RECT: int
MORPH_CROSS: int
MORPH_ELLIPSE: int

# MorphTypes
# https://docs.opencv.org/3.4/d4/d86/group__imgproc__filter.html#ga7be549266bad7b2e6a04db49827f9f32
MORPH_ERODE: int
MORPH_DILATE: int
MORPH_OPEN: int
MORPH_CLOSE: int
MORPH_GRADIENT: int
MORPH_TOPHAT: int
MORPH_BLACKHAT: int
MORPH_HITMISS: int

# Number Types
CV_16S: int
CV_16SC1: int
CV_16SC2: int
CV_16SC3: int
CV_16SC4: int
CV_16U: int
CV_16UC1: int
CV_16UC2: int
CV_16UC3: int
CV_16UC4: int
CV_32F: int
CV_32FC1: int
CV_32FC2: int
CV_32FC3: int
CV_32FC4: int
CV_32S: int
CV_32SC1: int
CV_32SC2: int
CV_32SC3: int
CV_32SC4: int
CV_64F: int
CV_64FC1: int
CV_64FC2: int
CV_64FC3: int
CV_64FC4: int
CV_8S: int
CV_8SC1: int
CV_8SC2: int
CV_8SC3: int
CV_8SC4: int
CV_8U: int
CV_8UC1: int
CV_8UC2: int
CV_8UC3: int
CV_8UC4: int

# ImreadModes
IMREAD_UNCHANGED: int  # If set, return the loaded image as is (with alpha channel, otherwise it gets cropped). Ignore EXIF orientation.
IMREAD_GRAYSCALE: int  # If set, always convert image to the single channel grayscale image (codec internal conversion).
IMREAD_COLOR: int  # If set, always convert image to the 3 channel BGR color image.
IMREAD_ANYDEPTH: int  # If set, return 16-bit/32-bit image when the input has the corresponding depth, otherwise convert it to 8-bit.
IMREAD_ANYCOLOR: int  # If set, the image is read in any possible color format.
IMREAD_LOAD_GDAL: int  # If set, use the gdal driver for loading the image.
IMREAD_REDUCED_GRAYSCALE_2: int  # If set, always convert image to the single channel grayscale image and the image size reduced 1/2.
IMREAD_REDUCED_COLOR_2: int  # If set, always convert image to the 3 channel BGR color image and the image size reduced 1/2.
IMREAD_REDUCED_GRAYSCALE_4: int  # If set, always convert image to the single channel grayscale image and the image size reduced 1/4.
IMREAD_REDUCED_COLOR_4: int  # If set, always convert image to the 3 channel BGR color image and the image size reduced 1/4.
IMREAD_REDUCED_GRAYSCALE_8: int  # If set, always convert image to the single channel grayscale image and the image size reduced 1/8.
IMREAD_REDUCED_COLOR_8: int  # If set, always convert image to the 3 channel BGR color image and the image size reduced 1/8.
IMREAD_IGNORE_ORIENTATION: int  # If set, do not rotate the image according to EXIF's orientation flag.

# Imwrite flags.
IMWRITE_JPEG_QUALITY: int  # For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
IMWRITE_JPEG_PROGRESSIVE: int  # Enable JPEG features, 0 or 1, default is False.
IMWRITE_JPEG_OPTIMIZE: int  # Enable JPEG features, 0 or 1, default is False.
IMWRITE_JPEG_RST_INTERVAL: int  # JPEG restart interval, 0 - 65535, default is 0 - no restart.
IMWRITE_JPEG_LUMA_QUALITY: int  # Separate luma quality level, 0 - 100, default is -1 - don't use.
IMWRITE_JPEG_CHROMA_QUALITY: int  # Separate chroma quality level, 0 - 100, default is -1 - don't use.
IMWRITE_JPEG_SAMPLING_FACTOR: int  # For JPEG, set sampling factor. See cv::ImwriteJPEGSamplingFactorParams.
IMWRITE_PNG_COMPRESSION: int  # For PNG, it can be the compression level from 0 to 9. A higher value means a smaller size and longer compression time. If specified, strategy is changed to IMWRITE_PNG_STRATEGY_DEFAULT (Z_DEFAULT_STRATEGY). Default value is 1 (best speed setting).
IMWRITE_PNG_STRATEGY: int  # One of cv::ImwritePNGFlags, default is IMWRITE_PNG_STRATEGY_RLE.
IMWRITE_PNG_BILEVEL: int  # Binary level PNG, 0 or 1, default is 0.
IMWRITE_PXM_BINARY: int  # For PPM, PGM, or PBM, it can be a binary format flag, 0 or 1. Default value is 1.
IMWRITE_EXR_TYPE: int
IMWRITE_WEBP_QUALITY: int  # override EXR storage type (FLOAT (FP32) is default). For WEBP, it can be a quality from 1 to 100 (the higher is the better). By default (without any parameter) and for quality above 100 the lossless compression is used.
IMWRITE_HDR_COMPRESSION: int
IMWRITE_PAM_TUPLETYPE: int  # specify HDR compression. For PAM, sets the TUPLETYPE field to the corresponding string value that is defined for the format
IMWRITE_TIFF_RESUNIT: int  # For TIFF, use to specify which DPI resolution unit to set; see libtiff documentation for valid values.
IMWRITE_TIFF_XDPI: int  # For TIFF, use to specify the X direction DPI.
IMWRITE_TIFF_YDPI: int  # For TIFF, use to specify the Y direction DPI.
IMWRITE_TIFF_COMPRESSION: int  # For TIFF, use to specify the image compression scheme. See libtiff for integer constants corresponding to compression formats. Note, for images whose depth is CV_32F, only libtiff's SGILOG compression scheme is used. For other supported depths, the compression scheme can be specified by this flag; LZW compression is the default.

# ImwritePAMFlags
IMWRITE_PAM_FORMAT_NULL: int
IMWRITE_PAM_FORMAT_BLACKANDWHITE: int
IMWRITE_PAM_FORMAT_GRAYSCALE: int
IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA: int
IMWRITE_PAM_FORMAT_RGB: int
IMWRITE_PAM_FORMAT_RGB_ALPHA: int

# InterpolationFlags
INTER_NEAREST: int  # Nearest neighbor interpolation.
INTER_LINEAR: int  # Bilinear interpolation.
INTER_CUBIC: int  # Bicubic interpolation.
INTER_AREA: int  # Resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire'-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
INTER_LANCZOS4: int  # Lanczos interpolation over 8x8 neighborhood.
INTER_LINEAR_EXACT: int  # Bit exact bilinear interpolation.
INTER_NEAREST_EXACT: int  # Bit exact nearest neighbor interpolation. This will produce same results as the nearest neighbor method in PIL, scikit-image or Matlab.
INTER_MAX: int  # Mask for interpolation codes.
WARP_FILL_OUTLIERS: int  # Flag, fills all of the destination image pixels. If some of them correspond to outliers in the source image, they are set to zero.
WARP_INVERSE_MAP: int  # Flag, inverse transformation. For example, linearPolar or logPolar transforms: flag is not set: dst(ρ,ϕ)=src(x,y) and flag is set: dst(x,y)=src(ρ,ϕ).  # noqa: RUF003

# Line types
FILLED: int
LINE_4: int  # 4-connected line
LINE_8: int  # 8-connected line
LINE_AA: int  # Antialiased line

# RetrievalModes
# https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#ga819779b9857cc2f8601e6526a3a5bc71
RETR_EXTERNAL: int
RETR_LIST: int
RETR_CCOMP: int
RETR_TREE: int
RETR_FLOODFILL: int

# seamlessClone algorithm flags
# https://docs.opencv.org/3.4/df/da0/group__photo__clone.html#ga19386064a1bd4e1153262844e6875bcc
NORMAL_CLONE: int  # The power of the method is fully expressed when inserting objects with complex outlines into a new background
MIXED_CLONE:int   # The classic method, color-based selection and alpha masking might be time consuming and often leaves an undesirable halo. Seamless cloning, even averaged with the original image, is not effective. Mixed seamless cloning based on a loose selection proves effective.
MONOCHROME_TRANSFER: int  # Monochrome transfer allows the user to easily replace certain features of one object by alternative features.

# ThresholdTypes
THRESH_BINARY: int
THRESH_BINARY_INV: int
THRESH_TRUNC: int
THRESH_TOZERO: int
THRESH_TOZERO_INV: int
THRESH_MASK: int
THRESH_OTSU: int
THRESH_TRIANGLE: int

# WindowFlags
WINDOW_NORMAL: int # The user can resize the window (no constraint) / also use to switch a fullscreen window to a normal size.
WINDOW_AUTOSIZE: int # The user cannot resize the window, the size is constrainted by the image displayed.
WINDOW_OPENGL: int  # Window with opengl support.
WINDOW_FULLSCREEN: int  # Change the window to fullscreen.
WINDOW_FREERATIO: int  # The image expends as much as it can (no ratio constraint).
WINDOW_KEEPRATIO: int  # The ratio of the image is respected.
WINDOW_GUI_EXPANDED: int  # Status bar and tool bar
WINDOW_GUI_NORMAL: int  # Old fashious way

# WindowPropertyFlags
WND_PROP_FULLSCREEN: int  # Fullscreen property (can be WINDOW_NORMAL or WINDOW_FULLSCREEN).
WND_PROP_AUTOSIZE: int  # Autosize property (can be WINDOW_NORMAL or WINDOW_AUTOSIZE).
WND_PROP_ASPECT_RATIO: int  # Window's aspect ration (can be set to WINDOW_FREERATIO or WINDOW_KEEPRATIO).
WND_PROP_OPENGL: int  # Opengl support.
WND_PROP_VISIBLE: int  # Checks whether the window exists and is visible
WND_PROP_TOPMOST: int  # Property to toggle normal window being topmost or not
WND_PROP_VSYNC: int
