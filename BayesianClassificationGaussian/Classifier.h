#ifndef CLASSIFIER_H
#define CLASSIFIER_H

// Libraries used
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <math.h>
using namespace Eigen;
using namespace std;

extern float box_muller(float, float);

class Classifier
{
public:
    // Generate 100,000 samples from a 2D Gaussian distribution
    vector<Vector2f> generateSamples(Vector2f mu, Matrix2f sigma);

    // ** CLASSIFIERS **

    // Baye's cases
    int caseOne(Vector2f, Vector2f, Vector2f, float, float, float, float);
    int caseTwo(Vector2f, Vector2f, Vector2f, Matrix2f, Matrix2f, float, float);
    int caseThree(Vector2f, Vector2f, Vector2f, Matrix2f, Matrix2f, float, float);

    // Minimum distance
    int minimumDistance(Vector2f, Vector2f, Vector2f);

    // ** BOUNDS **

    // Chernoff bound
    pair<float, float> chernoffBound(Vector2f, Vector2f, Matrix2f, Matrix2f);

    // Bhattacharyya bound
    float bhattacharyyaBound(Vector2f, Vector2f, Matrix2f, Matrix2f);

private:
    // Calculates squared of 2x1 vector
    float squared(Vector2f);

    //Calculates error
    float error(float, Vector2f, Vector2f, Matrix2f, Matrix2f);
};
#endif
