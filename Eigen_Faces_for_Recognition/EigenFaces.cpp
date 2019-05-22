#include "EigenFaces.h"

// Checks if files containing eigen face data, eigen value data, and average face
// are available/created for reading. If they are available returns true and reads them
// else returns false.
bool savedFacesExist(cv::Mat1f &averageFace, cv::Mat1f &eigenFaces, cv::Mat1f &eigenValues,
                     const char *path)
{
    char fileName[50];
    sprintf(fileName, "%s/eigen_faces.yml", path);
    cv::FileStorage fsEigenFaces(fileName, cv::FileStorage::READ);

    if(fsEigenFaces.isOpened()) {
        fsEigenFaces["Eigen Faces"] >> eigenFaces;
    }
    else {
        return false;
    }

    sprintf(fileName, "%s/eigen_values.yml", path);
    cv::FileStorage fsEigenValues(fileName, cv::FileStorage::READ);

    if(fsEigenFaces.isOpened()) {
        fsEigenFaces["Eigen Values"] >> eigenFaces;
    }
    else {
        return false;
    }

    sprintf(fileName, "%s/average_face.yml", path);
    cv::FileStorage fsAverageFace(fileName, cv::FileStorage::READ);

    if(fsEigenFaces.isOpened()) {
        fsEigenFaces["Average Face"] >> eigenFaces;
    }
    else {
        return false;
    }
    return true;
}

// Reads in images from directory specified by path variable into vector that
// includes the filename in a pair with the image
void readFaces(const char *path, std::vector<std::pair<std::string, cv::Mat1f> > &faces)
{
    std::vector<cv::String> fileNames;
    cv::Mat1f temp;
    glob(path, fileNames, false);
    size_t count = fileNames.size(); //number of png files in images folder
    for (size_t i = 0; i < count; i++) {
        temp = imread(fileNames[i], cv::IMREAD_GRAYSCALE);
        temp.reshape(1,1);

        faces.push_back(std::pair<std::string, cv::Mat1f>(fileNames[i], temp));
    }
}

// Converts the face vector to an image normalizes and then writes it out to file
void writeEigenFace(cv::Mat1f theImage, char *fileName)
{
    cv::Mat1f temp = theImage;
    temp.reshape(1,48);
    cv::normalize(theImage, temp, 255, 0, cv::NORM_MINMAX, -1);
    cv::imwrite(fileName, temp);
}

// Normalizes every eigenface using cv::normalize
void normalizeEigenFaces(cv::Mat1f &eigenfaces, int numberOfEigenFaces)
{
    cv::Mat1f subImage;
    for(size_t i = 0; i < numberOfEigenFaces; i++)
    {
        subImage = eigenfaces.col(i);
        // normalizes the values by dividing by the unit vector
        normalize(subImage, subImage, 1, 0, cv::NORM_L1, -1 );
    }
}


// Computes eigenfaces, average face and eigen values for the
void computeEigenFaces(std::vector<std::pair<std::string, cv::Mat1f> > trainingFaces, cv::Mat1f &averageFace,
                       cv::Mat1f &eigenFaces, cv::Mat1f &eigenValues, const char *path)
{
    char fileName[150];
    // Initializes the Average Face Mat to size of training data with zeros i to j
    averageFace = cv::Mat::zeros(trainingFaces[0].second.size(), trainingFaces[0].second.type());
    // Creates an average of all faces by adding respective matrices together and then
    // dividing them by the amount of faces
    for(size_t i = 0; i < trainingFaces.size(); i++)
    {
        add(averageFace, trainingFaces[0].second, averageFace);
    }
    averageFace /= trainingFaces.size();

    // Writes out Average Face data to yml file
    sprintf(fileName, "%s/average_face.yml", path);
    cv::FileStorage fsAvg(fileName, cv::FileStorage::WRITE);
    fsAvg << "Average Face" << averageFace;

    // Subtracts average face from the training face and then
    // concats all face images column wise to create Matrix A for SVD
    cv::Mat1f A(trainingFaces[0].second.rows * trainingFaces[0].second.cols, trainingFaces.size(),
                trainingFaces[0].second.type());
    for(size_t i = 0; i < trainingFaces.size(); i++)
    {
        cv::hconcat(A, trainingFaces[i].second.t() - averageFace, A);
    }

    // Performs matrix multiplication A transpose by A to get decomposed matrix and then
    // computes eigenvalues and eigenVectors
    cv::Mat1f eigenVectors;
    cv::eigen(A.t()*A, eigenValues, eigenVectors);

    eigenFaces = A * eigenVectors;


    // Saves Eigen values to yml file
    sprintf(fileName, "%s/eigen_values.yml", path);
    cv::FileStorage fsEigenValues(fileName, cv::FileStorage::WRITE);
    fsEigenValues << "Eigen Values" << eigenValues;


    // Save Eigen Faces to yml file
    sprintf(fileName, "%s/eigen_faces.yml", path);
    cv::FileStorage fsEigenFaces(fileName, cv::FileStorage::WRITE);
    fsEigenFaces << "Eigen Faces" << eigenFaces;
}
