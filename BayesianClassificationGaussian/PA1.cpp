// Libraries used
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <fstream>
using namespace Eigen;
using namespace std;

// Files used
#include "Classifier.h"

// Function used to write to results folder
void samplesFile(const char* fileName, vector<Vector2f> one, vector<Vector2f> two)
{
    // Open file
    ofstream output;
    output.open(fileName);

    output << "x1,y1,x2,y2\n";

    // Output and separate the vectors by spaces
    for(unsigned int i = 0; i < one.size(); i++)
    {
        output << one[i](0) << ",";
        output << one[i](1) << ",";
        output << two[i](0) << ",";
        output << two[i](1) << "\n";
    }

    // Close file
    output.close();
}

int main()
{
    srand(time(NULL));

    // LET'S BEGIN
    Classifier classifier;

    // Vectors used
    vector<Vector2f> one;
    vector<Vector2f> two;
    vector<Vector2f> misclassified;

    // Sigma/Mu
    Matrix2f sigmaOne;
    Vector2f muOne;
    Matrix2f sigmaTwo;
    Vector2f muTwo;

    // For Chernoff
    pair<float, float> chernoffBound;

    // ** QUESTION 1 SAMPLE GENERATION **

    // Set up parameters
    muOne << 1, 1;
    sigmaOne << 1, 0, 0, 1;
    muTwo << 4, 4;
    sigmaTwo << 1, 0, 0, 1;

    // Variables
    float priorOne = 0.5;
    float priorTwo = 0.5;
    int misclassOne = 0;
    int misclassTwo = 0;

    // Let's make 'em
    one = classifier.generateSamples(muOne, sigmaOne);
    two = classifier.generateSamples(muTwo, sigmaTwo);

    // ** QUESTION 1(A) **
    for(int i = 0; i < 100000; i++)
    {
        if(classifier.caseOne(one[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(classifier.caseOne(two[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Write to file
    samplesFile("./ForReport/1A-Misclassified.csv", misclassified, misclassified);

    // Answer questions
    cout << "QUESTION 1(A):" << endl;
    cout << "(i) Design a Bayes classifier for minimum error - DONE" << endl;
    cout << "(ii) Plot the Bayes decision boundary together with the generated samples to better visualize and interpret the classification results - REPORT" << endl;
    cout << "(iii) Report the number of misclassified samples for each class separately and the total number of misclassified samples..." << endl;
    cout << "\tSamples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "\tSamples from second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "\tTotal misclassified: " << misclassOne + misclassTwo << endl;
    cout << "(iv) Plot the Chernoff bound as a function B and find the optimum B for the minimum - REPORT" << endl;
    cout << "(v) Calculate the Bhattacharyya bound. Is it close to the experimental error - REPORT" << endl;

    // ** QUESTION 1(B) **

    // Reconfigure according to question specs
    misclassOne = 0;
    misclassTwo = 0;
    misclassified.clear();
    priorOne = 0.2;
    priorTwo = 0.8;

    // Calculate
    for(int i = 0; i < 100000; i++)
    {
        if(classifier.caseOne(one[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(classifier.caseOne(two[i], muOne, muTwo, sigmaOne(0,0), sigmaTwo(0,0), priorOne, priorTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    chernoffBound = classifier.chernoffBound(muOne, muTwo, sigmaOne, sigmaTwo);

    // Write to file
    samplesFile("./ForReport/Question1.csv", one, two);
    samplesFile("./ForReport/1B-Misclassified.csv", misclassified, misclassified);

    // Answer questions
    cout << "\nQUESTION 1(B):" << endl;
    cout << "(i) Design a Bayes classifier for minimum error - DONE" << endl;
    cout << "(ii) Plot the Bayes decision boundary together with the generated samples to better visualize and interpret the classification results - REPORT" << endl;
    cout << "(iii) Report the number of misclassified samples for each class separately and the total number of misclassified samples..." << endl;
    cout << "\tSamples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "\tSamples from second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "\tTotal misclassified: " << misclassOne + misclassTwo << endl;
    cout << "(iv) Plot the Chernoff bound as a function B and find the optimum B for the minimum - REPORT" << endl;
    cout << "(v) Calculate the Bhattacharyya bound. Is it close to the experimental error - REPORT" << endl;

    cout << "\nQuestion 1 error calculations:" << endl;
    cout << "With beta = " << chernoffBound.first << ", Chernoff Bound = " << chernoffBound.second << endl;
    cout << "With beta = 0.5, Bhattacharyya Bound = " << classifier.bhattacharyyaBound(muOne, muTwo, sigmaOne, sigmaTwo) << endl;

    // ** QUESTION 2 SAMPLE GENERATION **
    // Set up parameters
    muOne << 1, 1;
    sigmaOne << 1, 0, 0, 1;
    muTwo << 4, 4;
    sigmaTwo << 4, 0, 0, 8;

    // Variables
    priorOne = 0.5;
    priorTwo = 0.5;
    misclassOne = 0;
    misclassTwo = 0;

    // Let's make 'em
    one = classifier.generateSamples(muOne, sigmaOne);
    two = classifier.generateSamples(muTwo, sigmaTwo);

    // Clear misclassified
    misclassified.clear();

    // ** QUESTION 2(A) **
    for(int i = 0; i < 100000; i++)
    {
        if(classifier.caseThree(one[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(classifier.caseThree(two[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Write to file
    samplesFile("./ForReport/2A-Misclassified.csv", misclassified, misclassified);

    // Answer questions
    cout << "\nQUESTION 2(A):" << endl;
    cout << "(i) Design a Bayes classifier for minimum error - DONE" << endl;
    cout << "(ii) Plot the Bayes decision boundary together with the generated samples to better visualize and interpret the classification results - REPORT" << endl;
    cout << "(iii) Report the number of misclassified samples for each class separately and the total number of misclassified samples..." << endl;
    cout << "\tSamples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "\tSamples from second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "\tTotal misclassified: " << misclassOne + misclassTwo << endl;
    cout << "(iv) Plot the Chernoff bound as a function B and find the optimum B for the minimum - REPORT" << endl;
    cout << "(v) Calculate the Bhattacharyya bound. Is it close to the experimental error - REPORT" << endl;

    // ** QUESTION 2(B) **

    // Reconfigure according to question specs
    misclassOne = 0;
    misclassTwo = 0;
    misclassified.clear();
    priorOne = 0.2;
    priorTwo = 0.8;

    // Calculate
    for(int i = 0; i < 100000; i++)
    {
        if(classifier.caseThree(one[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(classifier.caseThree(two[i], muOne, muTwo, sigmaOne, sigmaTwo, priorOne, priorTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    chernoffBound = classifier.chernoffBound(muOne, muTwo, sigmaOne, sigmaTwo);

    // Write to file
    samplesFile("./ForReport/Question2.csv", one, two);
    samplesFile("./ForReport/2B-Misclassified.csv", misclassified, misclassified);

    // Answer questions
    cout << "\nQUESTION 2(B):" << endl;
    cout << "(i) Design a Bayes classifier for minimum error - DONE" << endl;
    cout << "(ii) Plot the Bayes decision boundary together with the generated samples to better visualize and interpret the classification results - REPORT" << endl;
    cout << "(iii) Report the number of misclassified samples for each class separately and the total number of misclassified samples..." << endl;
    cout << "\tSamples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "\tSamples from second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "\tTotal misclassified: " << misclassOne + misclassTwo << endl;
    cout << "(iv) Plot the Chernoff bound as a function B and find the optimum B for the minimum - REPORT" << endl;
    cout << "(v) Calculate the Bhattacharyya bound. Is it close to the experimental error - REPORT" << endl;

    cout << "\nQuestion 2 error calculations:" << endl;
    cout << "With beta = " << chernoffBound.first << ", Chernoff Bound = " << chernoffBound.second << endl;
    cout << "With beta = 0.5, Bhattacharyya Bound = " << classifier.bhattacharyyaBound(muOne, muTwo, sigmaOne, sigmaTwo) << endl;

    // ** QUESTION 3 **

    // Reconfigure according to question specs
    misclassOne = 0;
    misclassTwo = 0;
    misclassified.clear();

    // Calculate
    for(int i = 0; i < 100000; i++)
    {
        if(classifier.minimumDistance(one[i], muOne, muTwo) == 2)
        {
            misclassOne++;
            misclassified.push_back(one[i]);
        }
        if(classifier.minimumDistance(two[i], muOne, muTwo) == 1)
        {
            misclassTwo++;
            misclassified.push_back(two[i]);
        }
    }

    // Write to file
    samplesFile("./ForReport/3-Misclassified.csv", misclassified, misclassified);

    // Answer questions
    cout << "\nQUESTION 3:" << endl;
    cout << "(i) Design a Bayes classifier for minimum error - DONE" << endl;
    cout << "(ii) Plot the Bayes decision boundary together with the generated samples to better visualize and interpret the classification results - REPORT" << endl;
    cout << "(iii) Report the number of misclassified samples for each class separately and the total number of misclassified samples..." << endl;
    cout << "\tSamples from the first 2D Gaussian misclassified: " << misclassOne << endl;
    cout << "\tSamples from second 2D Gaussian misclassified: " << misclassTwo << endl;
    cout << "\tTotal misclassified: " << misclassOne + misclassTwo << endl;
    cout << "(iv) Plot the Chernoff bound as a function B and find the optimum B for the minimum - REPORT" << endl;
    cout << "(v) Calculate the Bhattacharyya bound. Is it close to the experimental error - REPORT" << endl;

    // Finish main
    return 0;
}
