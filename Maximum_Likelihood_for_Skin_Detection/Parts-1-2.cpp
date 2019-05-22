// Libraries used
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <iostream>
#include <fstream>
using namespace Eigen;
using namespace std;

// Files used
#include "Classifier.h"
#include "MLE.h"

// Return random index (for 1/100 samples)
int randIndex(int size)
{
    return (rand() / (float)RAND_MAX) * size;
}

int main()
{
    srand(time(NULL));

    // Vectors used
    vector<Vector2f> one;
    vector<Vector2f> two;
    vector<Vector2f> misclassified;

    // Sigma/Mu
    Matrix2f sigmaOne;
    Vector2f muOne;
    Matrix2f sigmaTwo;
    Vector2f muTwo;

    // Estimated Sigma/Mu
    Matrix2f estimatedSigOne;
    Vector2f estimatedMuOne;
    Matrix2f estimatedSigTwo;
    Vector2f estimatedMuTwo;

    // For generations
    Classifier generator;

    // ** QUESTION 1 SAMPLE GENERATION **

    // Set up parameters
    muOne << 1.0, 1.0;
    sigmaOne << 1.0, 0.0, 0.0, 1.0;
    muTwo << 4.0, 4.0;
    sigmaTwo << 1.0, 0.0, 0.0, 1.0;

    // Variables
    int misclassOne = 0;
    int misclassTwo = 0;

    // Let's make 'em
    one = generator.generateSamples(muOne, sigmaOne);
    two = generator.generateSamples(muTwo, sigmaTwo);

    // For reference since not saved from PA1
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseOne(one[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseOne(two[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0)) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output to terminal
    cout << "QUESTION 1 REFERENCE FROM PA1:" << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;

    // ** QUESTION 1(A) **

    // Estimate stats
    estimatedMuOne = MLE::sampleMean(one);
    estimatedMuTwo = MLE::sampleMean(two);
    estimatedSigOne = MLE::sampleCovariance(one, estimatedMuOne);
    estimatedSigTwo = MLE::sampleCovariance(two, estimatedMuTwo);

    // Reconfigure variables
    misclassified.clear();
    misclassOne = 0;
    misclassTwo = 0;

    // Classify based on MLE
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseOne(one[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseOne(two[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output the estimates/stats
    cout << "\nQUESTION 1(A):" << endl;
    cout << "Estimated Sample Mean and Covariance for first distribution: \nmuOne= [" << estimatedMuOne(0) << ", " << estimatedMuOne(1) << "]" << endl;
    cout << "sigmaOne=" << endl;
    cout << estimatedSigOne << endl;
    cout << "Estimated Sample Mean and Covariance for second distribution: \nmuTwo= [" <<  estimatedMuTwo(0) << ", " << estimatedMuTwo(1) << "]" << endl;
    cout << "sigmaTwo=" << endl;
    cout << estimatedSigTwo << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;

    // ** QUESTION 1(B) **

    // 1/100 of samples
    vector<Vector2f> smallone;
    vector<Vector2f> smalltwo;

    // Randomly select from one and two for smaller sample size
    for(int i = 0; i < 1000; i++)
    {
        int idxOne = randIndex(one.size());
        int idxTwo = randIndex(two.size());
        smallone.push_back(one[idxOne]); one.erase(one.begin() + idxOne);
        smalltwo.push_back(two[idxTwo]); two.erase(two.begin() + idxTwo);
    }

    // Estimate stats using small sample size
    estimatedMuOne = MLE::sampleMean(smallone);
    estimatedMuTwo = MLE::sampleMean(smalltwo);
    estimatedSigOne = MLE::sampleCovariance(smallone, estimatedMuOne);
    estimatedSigTwo = MLE::sampleCovariance(smalltwo, estimatedMuTwo);

    // Reconfigure variables
    misclassified.clear();
    misclassOne = 0;
    misclassTwo = 0;

    // Classify based on MLE
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseOne(one[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseOne(two[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output the estimates/stats
    cout << "\nQUESTION 1(B):" << endl;
    cout << "Estimated Sample Mean and Covariance for first distribution: \nmuOne= [" << estimatedMuOne(0) << ", " << estimatedMuOne(1) << "]" << endl;
    cout << "sigmaOne=" << endl;
    cout << estimatedSigOne << endl;
    cout << "Estimated Sample Mean and Covariance for second distribution: \nmuTwo= [" <<  estimatedMuTwo(0) << ", " << estimatedMuTwo(1) << "]" << endl;
    cout << "sigmaTwo=" << endl;
    cout << estimatedSigTwo << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;

    // ** QUESTION 2 SAMPLE GENERATION **

    // Set up parameters
    muOne << 1.0, 1.0;
    sigmaOne << 1.0, 0.0, 0.0, 1.0;
    muTwo << 4.0, 4.0;
    sigmaTwo << 4.0, 0.0, 0.0, 8.0;

    // Reconfigure variables
    misclassOne = 0;
    misclassTwo = 0;
    misclassified.clear();

    // Let's make 'em
    one = generator.generateSamples(muOne, sigmaOne);
    two = generator.generateSamples(muTwo, sigmaTwo);

    // For reference since since not saved from PA1
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseThree(one[i], muOne, muTwo, sigmaOne, sigmaTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseThree(two[i], muOne, muTwo, sigmaOne, sigmaTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output to terminal
    cout << "\n\nQUESTION 2 REFERENCE FROM PA1:" << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;

    // ** QUESTION 2(A) **

    // Estimate stats
    estimatedMuOne = MLE::sampleMean(one);
    estimatedMuTwo = MLE::sampleMean(two);
    estimatedSigOne = MLE::sampleCovariance(one, estimatedMuOne);
    estimatedSigTwo = MLE::sampleCovariance(two, estimatedMuTwo);

    // Reconfigure variables
    misclassified.clear();
    misclassOne = 0;
    misclassTwo = 0;

    // Classify based on MLE
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseThree(one[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne, estimatedSigTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseThree(two[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne, estimatedSigTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output the estimates/stats
    cout << "\nQUESTION 2(A):" << endl;
    cout << "Estimated Sample Mean and Covariance for first distribution: \nmuOne= [" << estimatedMuOne(0) << ", " << estimatedMuOne(1) << "]" << endl;
    cout << "sigmaOne=" << endl;
    cout << estimatedSigOne << endl;
    cout << "Estimated Sample Mean and Covariance for second distribution: \nmuTwo= [" <<  estimatedMuTwo(0) << ", " << estimatedMuTwo(1) << "]" << endl;
    cout << "sigmaTwo=" << endl;
    cout << estimatedSigTwo << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;

    // ** QUESTION 2(B) **

    // Reconfigure variables
    smallone.clear();
    smalltwo.clear();

    // Generate 1/100 of samples
    for(int i = 0; i < 1000; i++)
    {
        int idxOne = randIndex(one.size());
        int idxTwo = randIndex(two.size());
        smallone.push_back(one[idxOne]); one.erase(one.begin() + idxOne);
        smalltwo.push_back(two[idxTwo]); two.erase(two.begin() + idxTwo);
    }

    // Estimate stats based on small sample size
    estimatedMuOne = MLE::sampleMean(smallone);
    estimatedMuTwo = MLE::sampleMean(smalltwo);
    estimatedSigOne = MLE::sampleCovariance(smallone, estimatedMuOne);
    estimatedSigTwo = MLE::sampleCovariance(smalltwo, estimatedMuTwo);

    // Reconfigure variables
    misclassified.clear();
    misclassOne = 0;
    misclassTwo = 0;

    // Classify based on MLE
    for(int i = 0; i < 100000; i++)
    {
        if(Classifier::caseOne(one[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(Classifier::caseOne(two[i], estimatedMuOne, estimatedMuTwo, estimatedSigOne(0,0), estimatedSigTwo(0,0)) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Output the estimates/stats
    cout << "\nQUESTION 2(B):" << endl;
    cout << "Estimated Sample Mean and Covariance for first distribution: \nmuOne= [" << estimatedMuOne(0) << ", " << estimatedMuOne(1) << "]" << endl;
    cout << "sigmaOne=" << endl;
    cout << estimatedSigOne << endl;
    cout << "Estimated Sample Mean and Covariance for second distribution: \nmuTwo= [" <<  estimatedMuTwo(0) << ", " << estimatedMuTwo(1) << "]" << endl;
    cout << "sigmaTwo=" << endl;
    cout << estimatedSigTwo << endl;
    cout << "Samples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "Samples from the second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "Total misclassified: " << misclassOne + misclassTwo << endl;
}
