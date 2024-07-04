from typing import Any, TypeAlias

from .. import functions as cv2

boundingBox: TypeAlias = Any

retval: TypeAlias = Any

class MultiTracker(cv2.Algorithm):
    def add(self, newTracker, image, boundingBox) -> retval:
        """
        * \brief Add a new object to be tracked.
        *
        * @param newTracker tracking algorithm to be used
        * @param image input image
        * @param boundingBox a rectangle represents ROI of the tracked object
        """

    def getObjects(self) -> retval:
        """
        * \brief Returns a reference to a storage for the tracked objects, each object corresponds to one tracker algorithm
        """

    def update(self, image) -> tuple[retval, boundingBox]:
        """
        * \brief Update the current tracking status.
        * @param image input image
        * @param boundingBox the tracking result, represent a list of ROIs of the tracked objects.
        """

    def create(self) -> retval:
        """
        * \brief Returns a pointer to a new instance of MultiTracker
        """

class Tracker(cv2.Algorithm):
    def init(self, image, boundingBox) -> retval:
        """
        @brief Initialize the tracker with a known bounding box that surrounded the target
        @param image The initial frame
        @param boundingBox The initial bounding box  @return True if initialization went succesfully, false otherwise
        """

    def update(self, image) -> tuple[retval, boundingBox]:
        """
        @brief Update the tracker, find the new most likely bounding box for the target
        @param image The current frame
        @param boundingBox The bounding box that represent the new target location, if true was returned, not modified otherwise  @return True means that target was located and false means that tracker cannot locate target in current frame. Note, that latter *does not* imply that tracker has failed, maybe target is indeed missing from the frame (say, out of sight)
        """

class TrackerBoosting(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters BOOSTING parameters TrackerBoosting::Params
        """

class TrackerCSRT(Tracker):
    def setInitialMask(self, mask) -> None:
        """"""

    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters CSRT parameters TrackerCSRT::Params
        """

class TrackerKCF(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters KCF parameters TrackerKCF::Params
        """

class TrackerMIL(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters MIL parameters TrackerMIL::Params
        """

class TrackerMOSSE(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        """

class TrackerMedianFlow(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters Median Flow parameters TrackerMedianFlow::Params
        """

class TrackerTLD(Tracker):
    def create(self) -> retval:
        """
        @brief Constructor
        @param parameters TLD parameters TrackerTLD::Params
        """

def MultiTracker_create() -> retval:
    """
    * \brief Returns a pointer to a new instance of MultiTracker
    """

def TrackerBoosting_create() -> retval:
    """
    @brief Constructor
        @param parameters BOOSTING parameters TrackerBoosting::Params
    """

def TrackerCSRT_create() -> retval:
    """
    @brief Constructor
      @param parameters CSRT parameters TrackerCSRT::Params
    """

def TrackerKCF_create() -> retval:
    """
    @brief Constructor
      @param parameters KCF parameters TrackerKCF::Params
    """

def TrackerMIL_create() -> retval:
    """
    @brief Constructor
        @param parameters MIL parameters TrackerMIL::Params
    """

def TrackerMOSSE_create() -> retval:
    """
    @brief Constructor
    """

def TrackerMedianFlow_create() -> retval:
    """
    @brief Constructor
        @param parameters Median Flow parameters TrackerMedianFlow::Params
    """

def TrackerTLD_create() -> retval:
    """
    @brief Constructor
        @param parameters TLD parameters TrackerTLD::Params
    """

def upgradeTrackingAPI(legacy_tracker) -> retval:
    """
    .
    """
