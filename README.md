# OpenCV stubs

[![PyPI](https://img.shields.io/pypi/v/opencv-stubs?color=green&style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/opencv-stubs?style=flat-square)](https://pypistats.org/packages/opencv-stubs)
[![License](https://img.shields.io/pypi/l/opencv-stubs?style=flat)](https://opensource.org/licenses/MIT)
![Linting](https://github.com/hoel-bagard/opencv-stubs/actions/workflows/pre-commit.yaml/badge.svg)

Unofficial python stubs for the opencv-python package.

Official OpenCV stubs now exist (in OpenCV>=4.8, cf [this issue](https://github.com/opencv/opencv/issues/14590) and [this PR](https://github.com/opencv/opencv/pull/20370)), so this repo was archived from 2023/03/27 to 2024/04/13.\
However I found the official stubs unhelpfull when it came to type hinting, so I'm un-archiving the repo.

This package includes all the functions, classes and constants (please open an issue if you find a missing one).\
For some functions, OpenCV may handle more types than defined in this package. If you would like a type/function to be added or modified, please open an issue or a PR. There may also be a few errors as some types have been added programmatically. Please open an issue/PR if you see one.

The stubs include the docstrings as they are otherwise not available in IDEs (as far as I know).

## Installation

The package is available on pypi [here](https://pypi.org/project/opencv-stubs/), you can install it with:

```
pip install opencv-stubs
```

The dependency on opencv is optional, and can be accessed with:
- `pip install opencv-stubs[opencv]`
- `pip install opencv-stubs[opencv-contrib]`
- `pip install opencv-stubs[opencv-headless]`


## Acknowledgements

A stub file with opencv functions can be found on the [Microsoft stubs repo](https://github.com/microsoft/python-type-stubs/tree/main/cv2). This package reused those functions (with some added typing).

## TODO:
- [ ] Do something about `cv2.gapi.cv`, `cv2.utils.cv2` and `cv2.mat_wrapper.cv` (do not duplicate everything if possible).
- [ ] Handle cases like `cv2.misc.version.cv2.misc.version.cv2.misc.get_ocv_version()`.
- [ ] Only include the `opencv-contrib` specific stubs when using `opencv-stubs[opencv-contrib]`.
