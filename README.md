# OpenCV stubs

[![PyPI](https://img.shields.io/pypi/v/opencv-stubs?color=green&style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Implementation](https://img.shields.io/pypi/implementation/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/opencv-stubs?style=flat)](https://pypi.org/project/opencv-stubs)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/opencv-stubs?style=flat-square)](https://pypistats.org/packages/opencv-stubs)
[![License](https://img.shields.io/pypi/l/opencv-stubs?style=flat)](https://opensource.org/licenses/MIT)
![Linting](https://github.com/hoel-bagard/opencv-stubs/actions/workflows/pre-commit.yaml/badge.svg)

Unofficial python stubs for the opencv-python package.

This package includes all the functions, classes and constants (please open an issue if you find a missing one).\
For some functions, OpenCV may handle more types than defined in this package. If you would like a type/function to be added or modified, please open an issue or a PR. There may also be a few errors as some types have been added progrmmatically. Please open an issue if you see one.
The typing is still a work in progress, if you want a function/method to be added first you can open an issue.

The stubs include the docstrings as they are otherwise not available in the IDE (as far as I know).

These stubs are a temporary help until official ones are made (see [this issue](https://github.com/opencv/opencv/issues/14590)).


## Installation

The package is available on pypi [here](https://pypi.org/project/opencv-stubs/), you can install it with:

```
pip install opencv-stubs
```

The dependency on opencv is optional, and can be accessed with:
- `pip install opencv-stubs[opencv]`
- `pip install opencv-stubs[opencv-contrib]`
- `pip install opencv-stubs[opencv-headless]`


## Notes:

A stub file with all the cv2 functions can be found on the [Microsoft stubs repo](https://github.com/microsoft/python-type-stubs/tree/main/cv2). This package reused those functions and added typing.
