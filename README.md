# OpenCV stubs

[![PyPI](https://img.shields.io/pypi/v/opencv-stubs?color=green&style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/opencv-stubs?style=flat-square)](https://pypistats.org/packages/opencv-stubs)
[![License](https://img.shields.io/pypi/l/opencv-stubs?style=flat)](https://opensource.org/licenses/MIT)
![Linting](https://github.com/hoel-bagard/opencv-stubs/actions/workflows/pre-commit.yaml/badge.svg)


Unofficial python stubs for the opencv-python package.

A stub file with all the cv2 function can be found on the [Microsoft stubs repo](https://github.com/microsoft/python-type-stubs/tree/main/cv2).\
The stubs from this package are different in the sense that they include better (although wrong) typing. OpenCV handles more types than defined in this package (and has much more functions than defined in this package). If you would like a function / type to be added, feel free to open a PR.

The stubs include the docstrings as they are otherwise not available in the IDE (as far as I know).

These stubs are a temporary help until official ones are made (see [this issue](https://github.com/opencv/opencv/issues/14590#issuecomment-1493255962)).


## Installation

The package is available on pypi [here](https://pypi.org/project/opencv-stubs/), you can install it with:

```
pip install opencv-stubs
```

The dependency on opencv is optional, and be accessed with `pip install opencv-stubs[opencv]` or `pip install opencv-stubs[opencv-headless]`.
