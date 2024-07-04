import builtins
from typing import Any, overload, TypeAlias

img: TypeAlias = Any
bounding_box: TypeAlias = Any
quantized_images: TypeAlias = Any
templ: TypeAlias = Any
matches: TypeAlias = Any

dst: TypeAlias = Any
retval: TypeAlias = Any

class ColorGradient(Modality):
    def create(self, weak_threshold, num_features, strong_threshold) -> retval:
        """
        * \brief Constructor.
        *
        * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
        * \param num_features     How many features a template must contain.
        * \param strong_threshold Consider as candidate features only gradients whose norms are
        *                         larger than this.
        """

class DepthNormal(Modality):
    def create(self, distance_threshold, difference_threshold, num_features, extract_threshold) -> retval:
        """
        * \brief Constructor.
        *
        * \param distance_threshold   Ignore pixels beyond this distance.
        * \param difference_threshold When computing normals, ignore contributions of pixels whose
        *                             depth difference with the central pixel is above this threshold.
        * \param num_features         How many features a template must contain.
        * \param extract_threshold    Consider as candidate feature only if there are no differing
        *                             orientations within a distance of extract_threshold.
        """

class Detector(builtins.object):
    def addSyntheticTemplate(self, templates, class_id) -> retval:
        """
        * \brief Add a new object template computed by external means.
        """

    def addTemplate(self, sources, class_id, object_mask) -> tuple[retval, bounding_box]:
        """
        * \brief Add new object template.
        *
        * \param      sources      Source images, one for each modality.
        * \param      class_id     Object class ID.
        * \param      object_mask  Mask separating object from background.
        * \param[out] bounding_box Optionally return bounding box of the extracted features.
        *
        * \return Template ID, or -1 if failed to extract a valid template.
        """

    def classIds(self) -> retval:
        """"""

    def getModalities(self) -> retval:
        """
        * \brief Get the modalities used by this detector.
        *
        * You are not permitted to add/remove modalities, but you may dynamic_cast them to
        * tweak parameters.
        """

    def getT(self, pyramid_level) -> retval:
        """
        * \brief Get sampling step T at pyramid_level.
        """

    def getTemplates(self, class_id, template_id) -> retval:
        """
        * \brief Get the template pyramid identified by template_id.
        *
        * For example, with 2 modalities (Gradient, Normal) and two pyramid levels
        * (L0, L1), the order is (GradientL0, NormalL0, GradientL1, NormalL1).
        """

    def match(self, sources, threshold, class_ids=..., quantized_images=..., masks=...) -> tuple[matches, quantized_images]:
        """
        * \brief Detect objects by template matching.
        *
        * Matches globally at the lowest pyramid level, then refines locally stepping up the pyramid.
        *
        * \param      sources   Source images, one for each modality.
        * \param      threshold Similarity threshold, a percentage between 0 and 100.
        * \param[out] matches   Template matches, sorted by similarity score.
        * \param      class_ids If non-empty, only search for the desired object classes.
        * \param[out] quantized_images Optionally return vector<Mat> of quantized images.
        * \param      masks     The masks for consideration during matching. The masks should be CV_8UC1
        *                       where 255 represents a valid pixel.  If non-empty, the vector must be
        *                       the same size as sources.  Each element must be
        *                       empty or the same size as its corresponding source.
        """

    def numClasses(self) -> retval:
        """"""

    @overload
    def numTemplates(self) -> retval:
        """"""

    @overload
    def numTemplates(self, class_id) -> retval:
        """"""

    def pyramidLevels(self) -> retval:
        """
        * \brief Get number of pyramid levels used by this detector.
        """

    def read(self, fn) -> None:
        """"""

    def readClasses(self, class_ids, format=...) -> None:
        """"""

    def writeClasses(self, format=...) -> None:
        """"""

class Feature(builtins.object): ...
class Match(builtins.object): ...

class Modality(builtins.object):
    def name(self) -> retval:
        """"""

    def process(self, src, mask=...) -> retval:
        """
        * \brief Form a quantized image pyramid from a source image.
        *
        * \param[in] src  The source image. Type depends on the modality.
        * \param[in] mask Optional mask. If not empty, unmasked pixels are set to zero
        *                 in quantized image and cannot be extracted as features.
        """

    def read(self, fn) -> None:
        """"""

    @overload
    def create(self, modality_type) -> retval:
        """
        * \brief Create modality by name.
        *
        * The following modality types are supported:
        * - "ColorGradient"
        * - "DepthNormal"
        """

    @overload
    def create(self, fn) -> retval:
        """
        * \brief Load a modality from file.
        """

class QuantizedPyramid(builtins.object):
    def extractTemplate(self) -> tuple[retval, templ]:
        """
        * \brief Extract most discriminant features at current pyramid level to form a new template.
        *
        * \param[out] templ The new template.
        """

    def pyrDown(self) -> None:
        """
        * \brief Go to the next pyramid level.
        *
        * \todo Allow pyramid scale factor other than 2
        """

    def quantize(self, dst=...) -> dst:
        """
        * \brief Compute quantized image at current pyramid level for online detection.
        *
        * \param[out] dst The destination 8-bit image. For each pixel at most one bit is set,
        *                 representing its classification.
        """

class Template(builtins.object): ...

def ColorGradient_create(weak_threshold, num_features, strong_threshold) -> retval:
    """
    * \brief Constructor.
       *
       * \param weak_threshold   When quantizing, discard gradients with magnitude less than this.
       * \param num_features     How many features a template must contain.
       * \param strong_threshold Consider as candidate features only gradients whose norms are
       *                         larger than this.
    """

def DepthNormal_create(distance_threshold, difference_threshold, num_features, extract_threshold) -> retval:
    """
    * \brief Constructor.
       *
       * \param distance_threshold   Ignore pixels beyond this distance.
       * \param difference_threshold When computing normals, ignore contributions of pixels whose
       *                             depth difference with the central pixel is above this threshold.
       * \param num_features         How many features a template must contain.
       * \param extract_threshold    Consider as candidate feature only if there are no differing
       *                             orientations within a distance of extract_threshold.
    """

@overload
def Modality_create(modality_type) -> retval:
    """
    * \brief Create modality by name.
       *
       * The following modality types are supported:
       * - "ColorGradient"
       * - "DepthNormal"
    """

@overload
def Modality_create(modality_type) -> retval:
    """
    * \brief Load a modality from file.
    """

def colormap(quantized, dst=...) -> dst:
    """
    * \brief Debug function to colormap a quantized image for viewing.
    """

def drawFeatures(img, templates, tl, size=...) -> img:
    """
    * \brief Debug function to draw linemod features
    @param img
    @param templates see @ref Detector::addTemplate
    @param tl template bbox top-left offset see @ref Detector::addTemplate
    @param size marker size see @ref cv::drawMarker
    """

def getDefaultLINE() -> retval:
    """
    * \brief Factory function for detector using LINE algorithm with color gradients.

    Default parameter settings suitable for VGA images.
    """

def getDefaultLINEMOD() -> retval:
    """
    * \brief Factory function for detector using LINE-MOD algorithm with color gradients
    and depth normals.

    Default parameter settings suitable for VGA images.
    """
