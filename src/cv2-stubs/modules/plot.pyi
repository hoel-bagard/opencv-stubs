from typing import Any, overload, TypeAlias

from .. import functions as cv2

_plotResult: TypeAlias = Any

retval: TypeAlias = Any

class Plot2d(cv2.Algorithm):
    def render(self, _plotResult=...) -> _plotResult:
        """"""

    def setGridLinesNumber(self, gridLinesNumber) -> None:
        """"""

    def setInvertOrientation(self, _invertOrientation) -> None:
        """"""

    def setMaxX(self, _plotMaxX) -> None:
        """"""

    def setMaxY(self, _plotMaxY) -> None:
        """"""

    def setMinX(self, _plotMinX) -> None:
        """"""

    def setMinY(self, _plotMinY) -> None:
        """"""

    def setNeedPlotLine(self, _needPlotLine) -> None:
        """
        * @brief Switches data visualization mode
        *
        * @param _needPlotLine if true then neighbour plot points will be connected by lines. * In other case data will be plotted as a set of standalone points.
        """

    def setPlotAxisColor(self, _plotAxisColor) -> None:
        """"""

    def setPlotBackgroundColor(self, _plotBackgroundColor) -> None:
        """"""

    def setPlotGridColor(self, _plotGridColor) -> None:
        """"""

    def setPlotLineColor(self, _plotLineColor) -> None:
        """"""

    def setPlotLineWidth(self, _plotLineWidth) -> None:
        """"""

    def setPlotSize(self, _plotSizeWidth, _plotSizeHeight) -> None:
        """"""

    def setPlotTextColor(self, _plotTextColor) -> None:
        """"""

    def setPointIdxToPrint(self, pointIdx) -> None:
        """
        * @brief Sets the index of a point which coordinates will be printed on the top left corner of the plot (if ShowText flag is true).
        *
        * @param pointIdx index of the required point in data array.
        """

    def setShowGrid(self, needShowGrid) -> None:
        """"""

    def setShowText(self, needShowText) -> None:
        """"""

    @overload
    def create(self, data) -> retval:
        """
        * @brief Creates Plot2d object
        *
        * @param data \f$1xN\f$ or \f$Nx1\f$ matrix containing \f$Y\f$ values of points to plot. \f$X\f$ values * will be equal to indexes of correspondind elements in data matrix.
        """

    @overload
    def create(self, dataX, dataY) -> retval:
        """
        * @brief Creates Plot2d object
        *
        * @param dataX \f$1xN\f$ or \f$Nx1\f$ matrix \f$X\f$ values of points to plot.
        * @param dataY \f$1xN\f$ or \f$Nx1\f$ matrix containing \f$Y\f$ values of points to plot.
        """

@overload
def Plot2d_create(data) -> retval:
    """
    * @brief Creates Plot2d object
                 *
                 * @param data \f$1xN\f$ or \f$Nx1\f$ matrix containing \f$Y\f$ values of points to plot. \f$X\f$ values
                 * will be equal to indexes of correspondind elements in data matrix.
    """

@overload
def Plot2d_create(data) -> retval:
    """
    * @brief Creates Plot2d object
                 *
                 * @param dataX \f$1xN\f$ or \f$Nx1\f$ matrix \f$X\f$ values of points to plot.
                 * @param dataY \f$1xN\f$ or \f$Nx1\f$ matrix containing \f$Y\f$ values of points to plot.
    """
