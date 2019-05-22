// Libraries used
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace Eigen;
using namespace std;
using namespace cv;
// Files used
#include "Classifier.h"
#include "MLE.h"

const char* original_window = "Original Image";
const char* reference_window = "Reference image";
const char* classified_window = "Classified Image";

struct MLStats{
    Vector2f muSkin;
    Matrix2f varianceSkin;
    Vector2f muSkinYCC;
    Matrix2f varianceSkinYCC;
};

void calculate_ML_image(Mat& training_image, Mat& reference_image, MLStats &stats);
void threshold_classification(Mat& test_image, Mat& reference_image, Mat& classified_image,
                              MLStats &stats, float thresMin, float thresMax, const char filename[][20]);

int main()
{
    Mat classifiedImage;
    // Loading in images to perform the classification on
    Mat training_image = imread("../P2_Data/Data_Proj2/Training_1.ppm", IMREAD_COLOR);
    Mat test_image_1 = imread("../P2_Data/Data_Proj2/Training_3.ppm", IMREAD_COLOR);
    Mat test_image_2 = imread("../P2_Data/Data_Proj2/Training_6.ppm", IMREAD_COLOR);
    // Loading in defined reference images for training of the classifier and comparison
    // of results against the classified results. For each pixel in the image, it has been
    // manually determined which pixel is a skin pixel.
    Mat reference_image_tr = imread("../P2_Data/Data_Proj2/ref1.ppm", IMREAD_COLOR);
    Mat reference_image_1 = imread("../P2_Data/Data_Proj2/ref3.ppm", IMREAD_COLOR);
    Mat reference_image_2 = imread("../P2_Data/Data_Proj2/ref6.ppm", IMREAD_COLOR);
    // are calculated using calculateMLSkin
    MLStats statsML;
    calculate_ML_image(training_image, reference_image_tr, statsML);
    const char filenametr3[2][20] = {"train3_RGB_ROC.txt", "train3_YCC_ROC.txt"};
    const char filenametr6[2][20] = {"train6_RGB_ROC.txt", "train6_YCCS_ROC.txt"};
    threshold_classification(test_image_1, reference_image_1, classifiedImage, statsML, -1, 0, filenametr3);
    threshold_classification(test_image_1, reference_image_1, classifiedImage, statsML, -1, 0, filenametr6);
    return 0;
}


void calculate_ML_image(Mat& training_image, Mat& reference_image, MLStats &stats)
{
    vector<Vector2f> skinSamplesRGB, nonSkinSamplesRGB, skinSamplesYCC, nonSkinSamplesYCC;
    float totalRGB, normR, normG, normCb, normCr;
    totalRGB = normR = normG = normCb = normCr = 0;
    for (int i = 0; i < training_image.rows; i++) {
        for (int j = 0; j < training_image.cols; j++) {
            // using RGB color space for calculations
            totalRGB = (float)training_image.at<Vec3b>(i,j)[0] +
                       (float)training_image.at<Vec3b>(i,j)[1] +
                       (float)training_image.at<Vec3b>(i,j)[2];
            if (totalRGB != 0) {
                normR = (float)training_image.at<Vec3b>(i,j)[2] / totalRGB;
                normG = (float)training_image.at<Vec3b>(i,j)[1] / totalRGB;
            }
            // using YCC color space for calculations
            normCb = -0.169 * (float)training_image.at<Vec3b>(i,j)[2] -
                     0.332 * (float)training_image.at<Vec3b>(i,j)[1] +
                     0.5 * (float)training_image.at<Vec3b>(i,j)[0];
            normCr = 0.5 * (float)training_image.at<Vec3b>(i,j)[2] -
                     0.419 * (float)training_image.at<Vec3b>(i,j)[1] -
                     0.081 * (float)training_image.at<Vec3b>(i,j)[0];
            if ( ( (float)reference_image.at<Vec3b>(i,j)[2] ) != 0 &&
                 ( (float)reference_image.at<Vec3b>(i,j)[1] ) != 0 &&
                 ( (float)reference_image.at<Vec3b>(i,j)[0] ) != 0 != 0 ) {
                skinSamplesRGB.push_back(Vector2f(normR, normG));
                skinSamplesYCC.push_back(Vector2f(normCb, normCr));
            }
        }
    }
    stats.muSkin = MLE::sampleMean(skinSamplesRGB);
    stats.varianceSkin = MLE::sampleCovariance(skinSamplesRGB, stats.muSkin);
    stats.muSkinYCC = MLE::sampleMean(skinSamplesYCC);
    stats.varianceSkinYCC = MLE::sampleCovariance(skinSamplesYCC, stats.muSkinYCC);
}


void threshold_classification(Mat& test_image, Mat& reference_image, Mat& classified_image, MLStats &stats, float thresMin, float thresMax, const char filename[][20])
{
    vector <float> falseNegatives, falsePositives, falseNegativesYCC, falsePositivesYCC;
    float totalRGB, normR, normG, normCb, normCr;
    float skinTotal, nonSkinTotal, falseNegative, falsePositive,
            skinTotalYCC, nonSkinTotalYCC, falseNegativeYCC, falsePositiveYCC;
    bool classifiedAsSkin, isSkin;
    totalRGB = normR = normG = normCb = normCr = skinTotalYCC = nonSkinTotalYCC =
    falseNegativeYCC = falsePositiveYCC = 0;
    for(float threshold = thresMin; threshold <= thresMax+0.02; threshold+=.05) {
        skinTotal = nonSkinTotal = falseNegative = falsePositive = 0;
        for (int i = 0; i < test_image.rows; i++) {
            for (int j = 0; j < test_image.cols; j++) {
                // using RGB color space for calculations
                totalRGB = (float) test_image.at<Vec3b>(i, j)[0] +
                           (float) test_image.at<Vec3b>(i, j)[1] +
                           (float) test_image.at<Vec3b>(i, j)[2];
                if (totalRGB != 0) {
                    normR = (float) test_image.at<Vec3b>(i, j)[2] / totalRGB;
                    normG = (float) test_image.at<Vec3b>(i, j)[1] / totalRGB;
                }
                // using YCC color space for calculations
                normCb = -0.169 * (float) test_image.at<Vec3b>(i, j)[2] -
                         0.332 * (float) test_image.at<Vec3b>(i, j)[1] +
                         0.5 * (float) test_image.at<Vec3b>(i, j)[0];
                normCr = 0.5 * (float) test_image.at<Vec3b>(i, j)[2] -
                         0.419 * (float) test_image.at<Vec3b>(i, j)[1] -
                         0.081 * (float) test_image.at<Vec3b>(i, j)[0];
                classifiedAsSkin = Classifier::thresholdCaseThree(Vector2f(normR, normG), stats.muSkinYCC, stats.varianceSkinYCC, threshold);

                isSkin = ( (float)reference_image.at<Vec3b>(i,j)[2] ) != 0 &&
                         ( (float)reference_image.at<Vec3b>(i,j)[1] ) != 0 &&
                         ( (float)reference_image.at<Vec3b>(i,j)[0] ) != 0;

                if(classifiedAsSkin) {
                    skinTotal++;
                }
                else {
                    nonSkinTotal++;
                }

                if(isSkin && !classifiedAsSkin) {
                    falseNegative++;
                }
                else if(!isSkin && classifiedAsSkin) {
                    falsePositive++;
                }
                // YCC classification
                classifiedAsSkin = Classifier::thresholdCaseThree(Vector2f(normCb, normCr), stats.muSkin, stats.varianceSkin, threshold);

                if(classifiedAsSkin) {
                    skinTotalYCC++;
                }
                else {
                    nonSkinTotalYCC++;
                }
                if(isSkin && !classifiedAsSkin) {
                    falseNegativeYCC++;
                }
                else if(!isSkin && classifiedAsSkin) {
                    falsePositiveYCC++;
                }

            }
        }
        falseNegatives.push_back(falseNegative / nonSkinTotal);
        falsePositives.push_back(falsePositive / skinTotal);
        falseNegativesYCC.push_back(falseNegativeYCC / nonSkinTotalYCC);
        falsePositivesYCC.push_back(falsePositiveYCC / skinTotalYCC);
        /*
        cout << "Threshold: " << threshold << ": " << endl;
        cout << falsePositive << endl;
        cout << skinTotal << endl;
        cout << "\tFalse Negative Rate: \t" << falseNegative / nonSkinTotal << endl;
        cout << "\tFalse Positive Rate: \t" << falsePositive / skinTotal << endl;
        cout << "Threshold: " << threshold << ": " << endl;
        cout << falsePositiveYCC << endl;
        cout << skinTotalYCC << endl;
        cout << "\tFalse Negative Rate: \t" << falseNegativeYCC / nonSkinTotalYCC << endl;
        cout << "\tFalse Positive Rate: \t" << falsePositiveYCC / skinTotalYCC << endl;
        */
    }
    ofstream outputFile;

    outputFile.open(filename[0]);

    outputFile << "Threshold\tFalseNegative\tFalsePositive" << endl;

    for(float threshold = thresMin, i = 0; threshold <= thresMax+0.02; threshold+=.05, i++)
    {
        outputFile << threshold << "\t" << falseNegatives[i] << "\t" << falsePositives[i] << endl;
    }

    outputFile.close();

    outputFile.open(filename[1]);

    outputFile << "Threshold\tFalseNegative\tFalsePositive" << endl;

    for(float threshold = thresMin, i = 0; threshold <= thresMax+0.02; threshold+=.05, i++)
    {
        outputFile << threshold << "\t" << falseNegativesYCC[i] << "\t" << falsePositivesYCC[i] << endl;
    }

    outputFile.close();
}
