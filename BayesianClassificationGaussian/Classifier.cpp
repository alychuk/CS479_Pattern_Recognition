//
// Created by adaml on 5/16/2019.
//

#include "Classifier.h"

#include "Classifier.h"

// Generate a gaussian distribution using the Eigen libraries box_muller function
// mu specifies the mean vector and sigma the covariance matrix for the prior probablities
vector<Vector2f> Classifier::generateSamples(Vector2f mu, Matrix2f sigma)
{
    vector<Vector2f> samples;

    for(int i = 0; i < 100000; i++)
    {
        samples.push_back(Vector2f(box_muller(mu(0,0), sigma(0,0)), box_muller(mu(1,0), sigma(1,1))));
    }

    return samples;
}

// Baye's case one
int Classifier::caseOne(Vector2f x, Vector2f muOne, Vector2f muTwo, float varianceOne, float varianceTwo, float priorOne, float priorTwo)
{
    float discrimOne = (((1.0/varianceOne) * muOne).transpose() * x) - (1.0/(2*varianceOne)) * squared(muOne);
    float discrimTwo = (((1.0/varianceTwo) * muTwo).transpose() * x) - (1.0/(2*varianceTwo)) * squared(muTwo);

    if(priorOne != priorTwo)
    {
        discrimOne += log(priorOne);
        discrimTwo += log(priorTwo);
    }

    if(discrimOne > discrimTwo)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

// Baye's case two
int Classifier::caseTwo(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne, float priorTwo)
{
    float discrimOne = ((sigmaOne.inverse() * muOne).transpose() * x)(0) - (0.5 * muOne.transpose() * sigmaOne.inverse() * muOne);
    float discrimTwo = ((sigmaTwo.inverse() * muTwo).transpose() * x)(0) - (0.5 * muTwo.transpose() * sigmaTwo.inverse() * muTwo);

    if(priorOne != priorTwo)
    {
        discrimOne += log(priorOne);
        discrimTwo += log(priorTwo);
    }

    if(discrimOne > discrimTwo)
        return 1;
    else
        return 2;
}

// Baye's case three
int Classifier::caseThree(Vector2f x, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo, float priorOne,  float priorTwo)
{
    float discrimOne = (x.transpose() * (-0.5 * sigmaOne.inverse()) * x) + ((sigmaOne.inverse() * muOne).transpose() * x)(0) + (-0.5 * muOne.transpose() * sigmaOne.inverse() * muOne) + (-0.5 * log(sigmaOne.determinant()));
    float discrimTwo = (x.transpose() * (-0.5 * sigmaTwo.inverse()) * x) + ((sigmaTwo.inverse() * muTwo).transpose() * x)(0) + (-0.5 * muTwo.transpose() * sigmaTwo.inverse() * muTwo) + (-0.5 * log(sigmaTwo.determinant()));

    if(priorOne != priorTwo)
    {
        discrimOne += log(priorOne);
        discrimTwo += log(priorTwo);
    }

    if(discrimOne > discrimTwo)
        return 1;
    else
        return 2;
}

// Minimum distance classifier
int Classifier::minimumDistance(Vector2f x, Vector2f muOne, Vector2f muTwo)
{
    float discrimOne = -1.0 * squared(x-muOne);
    float discrimTwo = -1.0 * squared(x-muTwo);

    if(discrimOne > discrimTwo)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

// Chernoff
pair<float, float> Classifier::chernoffBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{
    float chernoffIndex = 0.0;
    float chernoffValue = error(chernoffIndex, muOne, muTwo, sigmaOne, sigmaTwo);
    for(float i = 0.0; i <= 1; i += 0.00001)
    {
        float curChernoffValue = error(i, muOne, muTwo, sigmaOne, sigmaTwo);
        if(curChernoffValue < chernoffValue)
        {
            chernoffIndex = i;
            chernoffValue = curChernoffValue;
        }
    }

    return pair<float, float>(chernoffIndex, chernoffValue);
}

// Bhattacharyya
float Classifier::bhattacharyyaBound(Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{
    return error(0.5, muOne, muTwo, sigmaOne, sigmaTwo);
}

// Norm squared
float Classifier::squared(Vector2f x)
{
    return x.transpose() * x;
}

// Error
float Classifier::error(float beta, Vector2f muOne, Vector2f muTwo, Matrix2f sigmaOne, Matrix2f sigmaTwo)
{

    float kb = (beta*(1-beta))/2.0;
    kb *= (muOne - muTwo).transpose() * ((1-beta)*sigmaOne + (beta)*sigmaTwo).inverse() * (muOne-muTwo);
    kb += 0.5 * log( ((1-beta)*sigmaOne + (beta)*sigmaTwo).determinant() / (pow(sigmaOne.determinant(), 1-beta) * pow(sigmaTwo.determinant(), beta)));

    return exp(-1.0 * kb);
}
