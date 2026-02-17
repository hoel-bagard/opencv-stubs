# Changelog

## [0.1.3] - 2026-02-17

- Add missing `__init__` to `BFMatcher`
- Add missing `__init__` overloads to `CascadeClassifier` and `DMatch`
- Add missing `apply` overload to `BackgroundSubtractorMOG2`
- Add missing methods to `CLAHE` (`getBitShift`, `setBitShift`)
- Add missing methods to `DISOpticalFlow` (`getCoarsestScale`, `setCoarsestScale`, `getVariationalRefinementEpsilon`, `setVariationalRefinementEpsilon`)
- Fix `CascadeClassifier.convert` to be a staticmethod
- Fix `FaceDetectorYN.create` to be a staticmethod and add buffer overload
- Fix `FaceRecognizerSF.create` to be a staticmethod and add buffer overload
- Change `cv2.BFMatcher.create` to a staticmethod

## [0.1.2] - 2026-02-09

- Add missing return types for \_create factory functions

## [0.1.1] - 2025-11-04

- Complete `cv2.applyColorMap` signature (add `COLORMAP_TYPES`)

## [0.1.0] - 2025-08-03

- Add `cv2.error`
- Other fixes.

## [0.0.12] - 2025-06-08

- Add `cv2.pollKey`

## [0.0.11] - 2025-03-07

-Add missing inits for `VideoCapture` and `VideoWriter`.
