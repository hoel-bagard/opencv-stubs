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

# Color conversion codes
COLOR_BGR2GRAY: int
COLOR_BGR2RGB: int
COLOR_GRAY2BGR: int
COLOR_GRAY2RGB: int
COLOR_RGB2BGR: int
COLOR_RGB2GRAY: int

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
