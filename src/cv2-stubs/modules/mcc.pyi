import builtins
from typing import Any, Final, TypeAlias

from .. import functions as cv2

img: TypeAlias = Any
retval: TypeAlias = Any

class CChecker(builtins.object):
    def getBox(self) -> retval:
        """"""

    def getCenter(self) -> retval:
        """"""

    def getChartsRGB(self) -> retval:
        """"""

    def getChartsYCbCr(self) -> retval:
        """"""

    def getCost(self) -> retval:
        """"""

    def getTarget(self) -> retval:
        """"""

    def setBox(self, _box) -> None:
        """"""

    def setCenter(self, _center) -> None:
        """"""

    def setChartsRGB(self, _chartsRGB) -> None:
        """"""

    def setChartsYCbCr(self, _chartsYCbCr) -> None:
        """"""

    def setCost(self, _cost) -> None:
        """"""

    def setTarget(self, _target) -> None:
        """"""

    def create(self) -> retval:
        """
        \brief Create a new CChecker object.
        * \return A pointer to the implementation of the CChecker
        """

class CCheckerDetector(cv2.Algorithm):
    def getBestColorChecker(self) -> retval:
        """
        \brief Get the best color checker. By the best it means the one
        *         detected with the highest confidence.
        * \return checker A single colorchecker, if atleast one colorchecker
        *                 was detected, 'nullptr' otherwise.
        """

    def getListColorChecker(self) -> retval:
        """
        \brief Get the list of all detected colorcheckers
        * \return checkers vector of colorcheckers
        """

    def process(self, image, chartType, nc=..., useNet=..., params=...) -> retval:
        """
        \brief Find the ColorCharts in the given image.
        *
        * Differs from the above one only in the arguments.
        *
        * This version searches for the chart in the full image.
        *
        * The found charts are not returned but instead stored in the
        * detector, these can be accessed later on using getBestColorChecker()
        * and getListColorChecker()
        * \param image image in color space BGR
        * \param chartType type of the chart to detect
        * \param nc number of charts in the image, if you don't know the exact
        *           then keeping this number high helps.
        * \param useNet if it is true the network provided using the setNet()
        *               is used for preliminary search for regions where chart
        *               could be present, inside the regionsOfInterest provied.
        * \param params parameters of the detection system. More information
        *               about them can be found in the struct DetectorParameters.
        * \return true if atleast one chart is detected otherwise false
        """

    def processWithROI(self, image, chartType, regionsOfInterest, nc=..., useNet=..., params=...) -> retval:
        """
        \brief Find the ColorCharts in the given image.
        *
        * The found charts are not returned but instead stored in the
        * detector, these can be accessed later on using getBestColorChecker()
        * and getListColorChecker()
        * \param image image in color space BGR
        * \param chartType type of the chart to detect
        * \param regionsOfInterest regions of image to look for the chart, if
        *                          it is empty, charts are looked for in the
        *                          entire image
        * \param nc number of charts in the image, if you don't know the exact
        *           then keeping this number high helps.
        * \param useNet if it is true the network provided using the setNet()
        *               is used for preliminary search for regions where chart
        *               could be present, inside the regionsOfInterest provied.
        * \param params parameters of the detection system. More information
        *               about them can be found in the struct DetectorParameters.
        * \return true if atleast one chart is detected otherwise false
        """

    def setNet(self, net) -> retval:
        """
        \brief Set the net which will be used to find the approximate
        *         bounding boxes for the color charts.
        *
        * It is not necessary to use this, but this usually results in
        * better detection rate.
        *
        * \param net the neural network, if the network in empty, then
        *            the function will return false.
        * \return true if it was able to set the detector's network,
        *         false otherwise.
        """

    def create(self) -> retval:
        """
        \brief Returns the implementation of the CCheckerDetector.
        *
        """

class CCheckerDraw(builtins.object):
    def draw(self, img) -> img:
        """
        \brief Draws the checker to the given image.
        * \param img image in color space BGR
        * \return void
        """

    def create(self, pChecker, color=..., thickness=...) -> retval:
        """
        \brief Create a new CCheckerDraw object.
        * \param pChecker The checker which will be drawn by this object.
        * \param color The color by with which the squares of the checker
        *              will be drawn
        * \param thickness The thickness with which the sqaures will be
        *                  drawn
        * \return A pointer to the implementation of the CCheckerDraw
        """

class DetectorParameters(builtins.object):
    def create(self) -> retval:
        """"""

def CCheckerDetector_create() -> retval:
    """
    \brief Returns the implementation of the CCheckerDetector.
        *
    """

def CCheckerDraw_create(pChecker, color=..., thickness=...) -> retval:
    """
    \brief Create a new CCheckerDraw object.
        * \param pChecker The checker which will be drawn by this object.
        * \param color The color by with which the squares of the checker
        *              will be drawn
        * \param thickness The thickness with which the sqaures will be
        *                  drawn
        * \return A pointer to the implementation of the CCheckerDraw
    """

def CChecker_create() -> retval:
    """
    \brief Create a new CChecker object.
        * \return A pointer to the implementation of the CChecker
    """

def DetectorParameters_create() -> retval:
    """
    .
    """

MCC24: Final[int]
SG140: int
VINYL18: Final[int]
