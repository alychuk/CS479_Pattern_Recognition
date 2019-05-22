#ifndef MLE_H
#define MLE_H

// Libraries used
#include <Eigen/Dense>
#include <vector>
using namespace std;
using namespace Eigen;

class MLE
{
public:
    // Sample mean
    static Vector2f sampleMean(vector<Vector2f>);

    // Sample covariance
    static Matrix2f sampleCovariance(vector<Vector2f>, Vector2f);
};
#endif
