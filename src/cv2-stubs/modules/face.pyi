import builtins
from typing import Any, overload, TypeAlias

from .. import functions as cv2

confidence: TypeAlias = Any
facePoints: TypeAlias = Any
points: TypeAlias = Any
image: TypeAlias = Any
label: TypeAlias = Any
landmarks: TypeAlias = Any
faces: TypeAlias = Any
features: TypeAlias = Any

retval: TypeAlias = Any

class BIF(cv2.Algorithm):
    def compute(self, image, features=...) -> features:
        """
        Computes features sby input image.
        *  @param image Input image (CV_32FC1).
        *  @param features Feature vector (CV_32FC1).
        """

    def getNumBands(self) -> retval:
        """
        @returns The number of filter bands used for computing BIF.
        """

    def getNumRotations(self) -> retval:
        """
        @returns The number of image rotations.
        """

    def create(self, num_bands=..., num_rotations=...) -> retval:
        """
        * @param num_bands The number of filter bands (<=8) used for computing BIF.
        * @param num_rotations The number of image rotations for computing BIF. * @returns Object for computing BIF.
        """

class BasicFaceRecognizer(FaceRecognizer):
    def getEigenValues(self) -> retval:
        """"""

    def getEigenVectors(self) -> retval:
        """"""

    def getLabels(self) -> retval:
        """"""

    def getMean(self) -> retval:
        """"""

    def getNumComponents(self) -> retval:
        """
        @see setNumComponents
        """

    def getProjections(self) -> retval:
        """"""

    def getThreshold(self) -> retval:
        """
        @see setThreshold
        """

    def setNumComponents(self, val) -> None:
        """
        @copybrief getNumComponents @see getNumComponents
        """

    def setThreshold(self, val) -> None:
        """
        @copybrief getThreshold @see getThreshold
        """

class EigenFaceRecognizer(BasicFaceRecognizer):
    def create(self, num_components=..., threshold=...) -> retval:
        """
        @param num_components The number of components (read: Eigenfaces) kept for this Principal Component Analysis. As a hint: There's no rule how many components (read: Eigenfaces) should be kept for good reconstruction capabilities. It is based on your input data, so experiment with the number. Keeping 80 components should almost always be sufficient.
        @param threshold The threshold applied in the prediction.  ### Notes:  -   Training and prediction must be done on grayscale images, use cvtColor to convert between the color spaces. -   **THE EIGENFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your input data has the correct shape, else a meaningful exception is thrown. Use resize to resize the images. -   This model does not support updating.  ### Model internal data:  -   num_components see EigenFaceRecognizer::create. -   threshold see EigenFaceRecognizer::create. -   eigenvalues The eigenvalues for this Principal Component Analysis (ordered descending). -   eigenvectors The eigenvectors for this Principal Component Analysis (ordered by their eigenvalue). -   mean The sample mean calculated from the training data. -   projections The projections of the training data. -   labels The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.
        """

class FaceRecognizer(cv2.Algorithm):
    def getLabelInfo(self, label) -> retval:
        """
        @brief Gets string information by label.

        If an unknown label id is provided or there is no label information associated with the specified
        label id the method returns an empty string.
        """

    def getLabelsByString(self, str) -> retval:
        """
        @brief Gets vector of labels by string.

        The function searches for the labels containing the specified sub-string in the associated string
        info.
        """

    def predict(self, src) -> tuple[label, confidence]:
        """
        @brief Predicts a label and associated confidence (e.g. distance) for a given input image.

        @param src Sample image to get a prediction from.
        @param label The predicted label for the given image.
        @param confidence Associated confidence (e.g. distance) for the predicted label.  The suffix const means that prediction does not affect the internal model state, so the method can be safely called from within different threads.  The following example shows how to get a prediction from a trained model:  @code using namespace cv; // Do your initialization here (create the cv::FaceRecognizer model) ... // ... // Read in a sample image: Mat img = imread("person1/3.jpg", IMREAD_GRAYSCALE); // And get a prediction from the cv::FaceRecognizer: int predicted = model->predict(img); @endcode  Or to get a prediction and the associated confidence (e.g. distance):  @code using namespace cv; // Do your initialization here (create the cv::FaceRecognizer model) ... // ... Mat img = imread("person1/3.jpg", IMREAD_GRAYSCALE); // Some variables for the predicted label and associated confidence (e.g. distance): int predicted_label = -1; double predicted_confidence = 0.0; // Get the prediction and associated confidence from the model model->predict(img, predicted_label, predicted_confidence); @endcode
        """

    def predict_collect(self, src, collector) -> None:
        """
        @brief - if implemented - send all result of prediction to collector that can be used for somehow custom result handling
        @param src Sample image to get a prediction from.
        @param collector User-defined collector object that accepts all results  To implement this method u just have to do same internal cycle as in predict(InputArray src, CV_OUT int &label, CV_OUT double &confidence) but not try to get "best@ result, just resend it to caller side with given collector
        """

    def predict_label(self, src) -> retval:
        """
        @overload
        """

    def read(self, filename) -> None:
        """
        @brief Loads a FaceRecognizer and its model state.

        Loads a persisted model and state from a given XML or YAML file . Every FaceRecognizer has to
        overwrite FaceRecognizer::load(FileStorage& fs) to enable loading the model state.
        FaceRecognizer::load(FileStorage& fs) in turn gets called by
        FaceRecognizer::load(const String& filename), to ease saving a model.
        """

    def setLabelInfo(self, label, strInfo) -> None:
        """
        @brief Sets string info for the specified model's label.

        The string info is replaced by the provided value if it was set before for the specified label.
        """

    def train(self, src, labels) -> None:
        """
        @brief Trains a FaceRecognizer with given data and associated labels.

        @param src The training images, that means the faces you want to learn. The data has to be given as a vector\<Mat\>.
        @param labels The labels corresponding to the images have to be given either as a vector\<int\> or a Mat of type CV_32SC1.  The following source code snippet shows you how to learn a Fisherfaces model on a given set of images. The images are read with imread and pushed into a std::vector\<Mat\>. The labels of each image are stored within a std::vector\<int\> (you could also use a Mat of type CV_32SC1). Think of the label as the subject (the person) this image belongs to, so same subjects (persons) should have the same label. For the available FaceRecognizer you don't have to pay any attention to the order of the labels, just make sure same persons have the same label:  @code // holds images and labels vector<Mat> images; vector<int> labels; // using Mat of type CV_32SC1 // Mat labels(number_of_samples, 1, CV_32SC1); // images for first person images.push_back(imread("person0/0.jpg", IMREAD_GRAYSCALE)); labels.push_back(0); images.push_back(imread("person0/1.jpg", IMREAD_GRAYSCALE)); labels.push_back(0); images.push_back(imread("person0/2.jpg", IMREAD_GRAYSCALE)); labels.push_back(0); // images for second person images.push_back(imread("person1/0.jpg", IMREAD_GRAYSCALE)); labels.push_back(1); images.push_back(imread("person1/1.jpg", IMREAD_GRAYSCALE)); labels.push_back(1); images.push_back(imread("person1/2.jpg", IMREAD_GRAYSCALE)); labels.push_back(1); @endcode  Now that you have read some images, we can create a new FaceRecognizer. In this example I'll create a Fisherfaces model and decide to keep all of the possible Fisherfaces:  @code // Create a new Fisherfaces model and retain all available Fisherfaces, // this is the most common usage of this specific FaceRecognizer: // Ptr<FaceRecognizer> model =  FisherFaceRecognizer::create(); @endcode  And finally train it on the given dataset (the face images and labels):  @code // This is the common interface to train all of the available cv::FaceRecognizer // implementations: // model->train(images, labels); @endcode
        """

    def update(self, src, labels) -> None:
        """
        @brief Updates a FaceRecognizer with given data and associated labels.

        @param src The training images, that means the faces you want to learn. The data has to be given as a vector\<Mat\>.
        @param labels The labels corresponding to the images have to be given either as a vector\<int\> or a Mat of type CV_32SC1.  This method updates a (probably trained) FaceRecognizer, but only if the algorithm supports it. The Local Binary Patterns Histograms (LBPH) recognizer (see createLBPHFaceRecognizer) can be updated. For the Eigenfaces and Fisherfaces method, this is algorithmically not possible and you have to re-estimate the model with FaceRecognizer::train. In any case, a call to train empties the existing model and learns a new model, while update does not delete any model data.  @code // Create a new LBPH model (it can be updated) and use the default parameters, // this is the most common usage of this specific FaceRecognizer: // Ptr<FaceRecognizer> model =  LBPHFaceRecognizer::create(); // This is the common interface to train all of the available cv::FaceRecognizer // implementations: // model->train(images, labels); // Some containers to hold new image: vector<Mat> newImages; vector<int> newLabels; // You should add some images to the containers: // // ... // // Now updating the model is as easy as calling: model->update(newImages,newLabels); // This will preserve the old model data and extend the existing model // with the new features extracted from newImages! @endcode  Calling update on an Eigenfaces model (see EigenFaceRecognizer::create), which doesn't support updating, will throw an error similar to:  @code OpenCV Error: The function/feature is not implemented (This FaceRecognizer (FaceRecognizer.Eigenfaces) does not support updating, you have to use FaceRecognizer::train to update it.) in update, file /home/philipp/git/opencv/modules/contrib/src/facerec.cpp, line 305 terminate called after throwing an instance of 'cv::Exception' @endcode  @note The FaceRecognizer does not store your training images, because this would be very memory intense and it's not the responsibility of te FaceRecognizer to do so. The caller is responsible for maintaining the dataset, he want to work with.
        """

    def write(self, filename) -> None:
        """
        @brief Saves a FaceRecognizer and its model state.

        Saves this model to a given filename, either as XML or YAML.
        @param filename The filename to store this FaceRecognizer to (either XML/YAML).  Every FaceRecognizer overwrites FaceRecognizer::save(FileStorage& fs) to save the internal model state. FaceRecognizer::save(const String& filename) saves the state of a model to the given filename.  The suffix const means that prediction does not affect the internal model state, so the method can be safely called from within different threads.
        """

class Facemark(cv2.Algorithm):
    def fit(self, image, faces, landmarks=...) -> tuple[retval, landmarks]:
        """
        @brief Detect facial landmarks from an image.
        @param image Input image.
        @param faces Output of the function which represent region of interest of the detected faces. Each face is stored in cv::Rect container.
        @param landmarks The detected landmark points for each faces.  <B>Example of usage</B> @code Mat image = imread("image.jpg"); std::vector<Rect> faces; std::vector<std::vector<Point2f> > landmarks; facemark->fit(image, faces, landmarks); @endcode
        """

    def loadModel(self, model) -> None:
        """
        @brief A function to load the trained model before the fitting process.
        @param model A string represent the filename of a trained model.  <B>Example of usage</B> @code facemark->loadModel("../data/lbf.model"); @endcode
        """

class FacemarkAAM(FacemarkTrain): ...
class FacemarkKazemi(Facemark): ...
class FacemarkLBF(FacemarkTrain): ...
class FacemarkTrain(Facemark): ...

class FisherFaceRecognizer(BasicFaceRecognizer):
    def create(self, num_components=..., threshold=...) -> retval:
        """
        @param num_components The number of components (read: Fisherfaces) kept for this Linear Discriminant Analysis with the Fisherfaces criterion. It's useful to keep all components, that means the number of your classes c (read: subjects, persons you want to recognize). If you leave this at the default (0) or set it to a value less-equal 0 or greater (c-1), it will be set to the correct number (c-1) automatically.
        @param threshold The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.  ### Notes:  -   Training and prediction must be done on grayscale images, use cvtColor to convert between the color spaces. -   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your input data has the correct shape, else a meaningful exception is thrown. Use resize to resize the images. -   This model does not support updating.  ### Model internal data:  -   num_components see FisherFaceRecognizer::create. -   threshold see FisherFaceRecognizer::create. -   eigenvalues The eigenvalues for this Linear Discriminant Analysis (ordered descending). -   eigenvectors The eigenvectors for this Linear Discriminant Analysis (ordered by their eigenvalue). -   mean The sample mean calculated from the training data. -   projections The projections of the training data. -   labels The labels corresponding to the projections.
        """

class LBPHFaceRecognizer(FaceRecognizer):
    def getGridX(self) -> retval:
        """
        @see setGridX
        """

    def getGridY(self) -> retval:
        """
        @see setGridY
        """

    def getHistograms(self) -> retval:
        """"""

    def getLabels(self) -> retval:
        """"""

    def getNeighbors(self) -> retval:
        """
        @see setNeighbors
        """

    def getRadius(self) -> retval:
        """
        @see setRadius
        """

    def getThreshold(self) -> retval:
        """
        @see setThreshold
        """

    def setGridX(self, val) -> None:
        """
        @copybrief getGridX @see getGridX
        """

    def setGridY(self, val) -> None:
        """
        @copybrief getGridY @see getGridY
        """

    def setNeighbors(self, val) -> None:
        """
        @copybrief getNeighbors @see getNeighbors
        """

    def setRadius(self, val) -> None:
        """
        @copybrief getRadius @see getRadius
        """

    def setThreshold(self, val) -> None:
        """
        @copybrief getThreshold @see getThreshold
        """

    def create(self, radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
        """
        @param radius The radius used for building the Circular Local Binary Pattern. The greater the radius, the smoother the image but more spatial information you can get.
        @param neighbors The number of sample points to build a Circular Local Binary Pattern from. An appropriate value is to use `8` sample points. Keep in mind: the more sample points you include, the higher the computational cost.
        @param grid_x The number of cells in the horizontal direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
        @param grid_y The number of cells in the vertical direction, 8 is a common value used in publications. The more cells, the finer the grid, the higher the dimensionality of the resulting feature vector.
        @param threshold The threshold applied in the prediction. If the distance to the nearest neighbor is larger than the threshold, this method returns -1.  ### Notes:  -   The Circular Local Binary Patterns (used in training and prediction) expect the data given as grayscale images, use cvtColor to convert between the color spaces. -   This model supports updating.  ### Model internal data:  -   radius see LBPHFaceRecognizer::create. -   neighbors see LBPHFaceRecognizer::create. -   grid_x see LLBPHFaceRecognizer::create. -   grid_y see LBPHFaceRecognizer::create. -   threshold see LBPHFaceRecognizer::create. -   histograms Local Binary Patterns Histograms calculated from the given training data (empty if none was given). -   labels Labels corresponding to the calculated Local Binary Patterns Histograms.
        """

class MACE(cv2.Algorithm):
    def salt(self, passphrase) -> None:
        """
        @brief optionally encrypt images with random convolution
        @param passphrase a crc64 random seed will get generated from this
        """

    def same(self, query) -> retval:
        """
        @brief correlate query img and threshold to min class value
        @param query  a Mat with query image
        """

    def train(self, images) -> None:
        """
        @brief train it on positive features
        compute the mace filter: `h = D(-1) * X * (X(+) * D(-1) * X)(-1) * C`
        also calculate a minimal threshold for this class, the smallest self-similarity from the train images
        @param images  a vector<Mat> with the train images
        """

    def create(self, IMGSIZE=...) -> retval:
        """
        @brief constructor
        @param IMGSIZE  images will get resized to this (should be an even number)
        """

    def load(self, filename, objname=...) -> retval:
        """
        @brief constructor
        @param filename  build a new MACE instance from a pre-serialized FileStorage
        @param objname (optional) top-level node in the FileStorage
        """

class PredictCollector(builtins.object): ...

class StandardCollector(PredictCollector):
    def getMinDist(self) -> retval:
        """
        @brief Returns minimal distance value
        """

    def getMinLabel(self) -> retval:
        """
        @brief Returns label with minimal distance
        """

    def getResults(self, sorted=...) -> retval:
        """
        @brief Return results as vector
        @param sorted If set, results will be sorted by distance Each values is a pair of label and distance.
        """

    def create(self, threshold=...) -> retval:
        """
        @brief Static constructor
        @param threshold set threshold
        """

def BIF_create(num_bands=..., num_rotations=...) -> retval:
    """
    * @param num_bands The number of filter bands (<=8) used for computing BIF.
         * @param num_rotations The number of image rotations for computing BIF.
         * @returns Object for computing BIF.
    """

@overload
def EigenFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """
    @param num_components The number of components (read: Eigenfaces) kept for this Principal
    """

@overload
def EigenFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """ """

@overload
def EigenFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """ """

@overload
def EigenFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """
    @param threshold The threshold applied in the prediction.

    ### Notes:

    -   Training and prediction must be done on grayscale images, use cvtColor to convert between the
        color spaces.
    -   **THE EIGENFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
        SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
        input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
        the images.
    -   This model does not support updating.

    ### Model internal data:

    -   num_components see EigenFaceRecognizer::create.
    -   threshold see EigenFaceRecognizer::create.
    -   eigenvalues The eigenvalues for this Principal Component Analysis (ordered descending).
    -   eigenvectors The eigenvectors for this Principal Component Analysis (ordered by their
        eigenvalue).
    -   mean The sample mean calculated from the training data.
    -   projections The projections of the training data.
    -   labels The threshold applied in the prediction. If the distance to the nearest neighbor is
        larger than the threshold, this method returns -1.
    """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """
    @param num_components The number of components (read: Fisherfaces) kept for this Linear
    """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """ """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """ """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """ """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """
    @param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
    """

@overload
def FisherFaceRecognizer_create(num_components=..., threshold=...) -> retval:
    """

    ### Notes:

    -   Training and prediction must be done on grayscale images, use cvtColor to convert between the
        color spaces.
    -   **THE FISHERFACES METHOD MAKES THE ASSUMPTION, THAT THE TRAINING AND TEST IMAGES ARE OF EQUAL
        SIZE.** (caps-lock, because I got so many mails asking for this). You have to make sure your
        input data has the correct shape, else a meaningful exception is thrown. Use resize to resize
        the images.
    -   This model does not support updating.

    ### Model internal data:

    -   num_components see FisherFaceRecognizer::create.
    -   threshold see FisherFaceRecognizer::create.
    -   eigenvalues The eigenvalues for this Linear Discriminant Analysis (ordered descending).
    -   eigenvectors The eigenvectors for this Linear Discriminant Analysis (ordered by their
        eigenvalue).
    -   mean The sample mean calculated from the training data.
    -   projections The projections of the training data.
    -   labels The labels corresponding to the projections.
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """
    @param radius The radius used for building the Circular Local Binary Pattern. The greater the
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """
    @param neighbors The number of sample points to build a Circular Local Binary Pattern from. An
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """ """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """
    @param grid_x The number of cells in the horizontal direction, 8 is a common value used in
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """ """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """
    @param grid_y The number of cells in the vertical direction, 8 is a common value used in
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """ """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """
    @param threshold The threshold applied in the prediction. If the distance to the nearest neighbor
    """

@overload
def LBPHFaceRecognizer_create(radius=..., neighbors=..., grid_x=..., grid_y=..., threshold=...) -> retval:
    """

    ### Notes:

    -   The Circular Local Binary Patterns (used in training and prediction) expect the data given as
        grayscale images, use cvtColor to convert between the color spaces.
    -   This model supports updating.

    ### Model internal data:

    -   radius see LBPHFaceRecognizer::create.
    -   neighbors see LBPHFaceRecognizer::create.
    -   grid_x see LLBPHFaceRecognizer::create.
    -   grid_y see LBPHFaceRecognizer::create.
    -   threshold see LBPHFaceRecognizer::create.
    -   histograms Local Binary Patterns Histograms calculated from the given training data (empty if
        none was given).
    -   labels Labels corresponding to the calculated Local Binary Patterns Histograms.
    """

def MACE_create(IMGSIZE=...) -> retval:
    """
    @brief constructor
        @param IMGSIZE  images will get resized to this (should be an even number)
    """

def MACE_load(filename, objname=...) -> retval:
    """
    @brief constructor
        @param filename  build a new MACE instance from a pre-serialized FileStorage
        @param objname (optional) top-level node in the FileStorage
    """

def StandardCollector_create(threshold=...) -> retval:
    """
    @brief Static constructor
        @param threshold set threshold
    """

def createFacemarkAAM() -> retval:
    """
    .
    """

def createFacemarkKazemi() -> retval:
    """
    .
    """

def createFacemarkLBF() -> retval:
    """
    .
    """

@overload
def drawFacemarks(image, points, color=...) -> image:
    """
    @brief Utility to draw the detected facial landmark points

    @param image The input image to be processed.
    @param points Contains the data of points which will be drawn.
    @param color The color of points in BGR format represented by cv::Scalar.

    <B>Example of usage</B>
    @code
    std::vector<Rect> faces;
    std::vector<std::vector<Point2f> > landmarks;
    facemark->getFaces(img, faces);
    facemark->fit(img, faces, landmarks);
    for(int j=0;j<rects.size();j++){
    """

@overload
def drawFacemarks(image, points, color=...) -> image:
    """
    }
    @endcode
    """

@overload
def getFacesHAAR(image, face_cascade_name, faces=...) -> tuple[retval, faces]:
    """
    @brief Default face detector
    This function is mainly utilized by the implementation of a Facemark Algorithm.
    End users are advised to use function Facemark::getFaces which can be manually defined
    and circumvented to the algorithm by Facemark::setFaceDetector.

    @param image The input image to be processed.
    @param faces Output of the function which represent region of interest of the detected faces.
    Each face is stored in cv::Rect container.
    @param params detector parameters

    <B>Example of usage</B>
    @code
    std::vector<cv::Rect> faces;
    CParams params("haarcascade_frontalface_alt.xml");
    cv::face::getFaces(frame, faces, &params);
    for(int j=0;j<faces.size();j++){
    """

@overload
def getFacesHAAR(image, face_cascade_name, faces=...) -> tuple[retval, faces]:
    """
    }
    cv::imshow("detection", frame);
    @endcode
    """

def loadDatasetList(imageList, annotationList, images, annotations) -> retval:
    """
    @brief A utility to load list of paths to training image and annotation file.
    @param imageList The specified file contains paths to the training images.
    @param annotationList The specified file contains paths to the training annotations.
    @param images The loaded paths of training images.
    @param annotations The loaded paths of annotation files.

    Example of usage:
    @code
    String imageFiles = "images_path.txt";
    String ptsFiles = "annotations_path.txt";
    std::vector<String> images_train;
    std::vector<String> landmarks_train;
    loadDatasetList(imageFiles,ptsFiles,images_train,landmarks_train);
    @endcode
    """

def loadFacePoints(filename, points=..., offset=...) -> tuple[retval, points]:
    """
    @brief A utility to load facial landmark information from a given file.

    @param filename The filename of file contains the facial landmarks data.
    @param points The loaded facial landmark points.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    std::vector<Point2f> points;
    face::loadFacePoints("filename.txt", points, 0.0f);
    @endcode

    The annotation file should follow the default format which is
    @code
    version: 1
    n_points:  68
    {
    212.716603 499.771793
    230.232816 566.290071
    ...
    }
    @endcode
    where n_points is the number of points considered
    and each point is represented as its position in x and y.
    """

@overload
def loadTrainingData(filename, images, facePoints=..., delim=..., offset=...) -> tuple[retval, facePoints]:
    """
    @brief A utility to load facial landmark dataset from a single file.

    @param filename The filename of a file that contains the dataset information.
    Each line contains the filename of an image followed by
    pairs of x and y values of facial landmarks points separated by a space.
    Example
    @code
    /home/user/ibug/image_003_1.jpg 336.820955 240.864510 334.238298 260.922709 335.266918 ...
    /home/user/ibug/image_005_1.jpg 376.158428 230.845712 376.736984 254.924635 383.265403 ...
    @endcode
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.
    @param delim Delimiter between each element, the default value is a whitespace.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0f);
    @endcode
    """

@overload
def loadTrainingData(filename, images, facePoints=..., delim=..., offset=...) -> tuple[retval, facePoints]:
    """
    @brief A utility to load facial landmark information from the dataset.

    @param imageList A file contains the list of image filenames in the training dataset.
    @param groundTruth A file contains the list of filenames
    where the landmarks points information are stored.
    The content in each file should follow the standard format (see face::loadFacePoints).
    @param images A vector where each element represent the filename of image in the dataset.
    Images are not loaded by default to save the memory.
    @param facePoints The loaded landmark points for all training data.
    @param offset An offset value to adjust the loaded points.

    <B>Example of usage</B>
    @code
    cv::String imageFiles = "../data/images_train.txt";
    cv::String ptsFiles = "../data/points_train.txt";
    std::vector<String> images;
    std::vector<std::vector<Point2f> > facePoints;
    loadTrainingData(imageFiles, ptsFiles, images, facePoints, 0.0f);
    @endcode

    example of content in the images_train.txt
    @code
    /home/user/ibug/image_003_1.jpg
    /home/user/ibug/image_004_1.jpg
    /home/user/ibug/image_005_1.jpg
    /home/user/ibug/image_006.jpg
    @endcode

    example of content in the points_train.txt
    @code
    /home/user/ibug/image_003_1.pts
    /home/user/ibug/image_004_1.pts
    /home/user/ibug/image_005_1.pts
    /home/user/ibug/image_006.pts
    @endcode
    """

@overload
def loadTrainingData(filename, images, facePoints=..., delim=..., offset=...) -> tuple[retval, facePoints]:
    """
    @brief This function extracts the data for training from .txt files which contains the corresponding image name and landmarks.
    *The first file in each file should give the path of the image whose
    *landmarks are being described in the file. Then in the subsequent
    *lines there should be coordinates of the landmarks in the image
    *i.e each line should be of the form x,y
    *where x represents the x coordinate of the landmark and y represents
    *the y coordinate of the landmark.
    *
    *For reference you can see the files as provided in the
    *<a href="http://www.ifp.illinois.edu/~vuongle2/helen/">HELEN dataset</a>
    *
    * @param filename A vector of type cv::String containing name of the .txt files.
    * @param trainlandmarks A vector of type cv::Point2f that would store shape or landmarks of all images.
    * @param trainimages A vector of type cv::String which stores the name of images whose landmarks are tracked
    * @returns A boolean value. It returns true when it reads the data successfully and false otherwise
    """
