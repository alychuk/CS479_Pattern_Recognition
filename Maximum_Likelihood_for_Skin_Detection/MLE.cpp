// Libraries used
#include "MLE.h"
#include <iostream>

// Sample mean
Vector2f MLE::sampleMean(vector<Vector2f> data)
{
    Vector2f sum;
    sum << 0.0, 0.0;
    for(vector<int>::size_type i = 0; i < data.size(); i++)
    {
        sum += data[i];
    }
    return sum / data.size();
}

// Covariance
Matrix2f MLE::sampleCovariance(vector<Vector2f> data, Vector2f sampleMean)
{
    Matrix2f sum;
    sum <<        	0.0, 0.0, 0.0, 0.0;
    for(vector<int>::size_type i = 0; i <  data.size(); i++)
    {
        sum += (sampleMean - data[i])*((sampleMean - data[i]).transpose());
    }
    return sum / data.size();
}
