#include "Identify.h"

// Global vars
extern float PCA_PERCENTAGE;

// Returns bool if x is bigger than y
bool Compare(pair<string, float> x, pair<string, float> y)
{
    return x.second < y.second;
}

// Returns float of norm of original space subtracted by new face
float distanceFaces(cv::Mat1f &originalFace, cv::Mat1f &newFace)
{
    return cv::norm(originalFace, newFace, cv::NORM_L2);
}

// Returns bool if a face as same ID as the search ID
bool faceIdentified(vector<pair<string, float> > similarFaces, int N, string searchID)
{
    for(int i = 0; i < N; i++)
    {
        if(similarFaces[i].first == searchID)
        {
            return true;
        }
    }
    return false;
}

// Calculates face projection
cv::Mat1f projectEigenFace(cv::Mat1f &newFace, cv::Mat1f &averageFace, cv::Mat1f &eigenFaces)
{
    cv::Mat1f normalizedFace = newFace - averageFace;
    // Initializes the Projected Face Mat to size of training data with zeros i to j
    cv::Mat1f projectedFace = cv::Mat::zeros(averageFace.size(), averageFace.type());
    cv::Mat1f subImage;
    for(size_t i = 0; i < eigenFaces.cols; i++)
    {
        subImage = eigenFaces.col(i);
        float faceCoefficients = normalizedFace.dot(subImage.t());
        projectedFace += (faceCoefficients * subImage);
    }
    return projectedFace + averageFace;
}

// Runs classifier cycling from 1 <= N <= 50
// Prints correctly/incorrectly matched faces and CMC curve data
void identify(const char *resultsPath, cv::Mat1f averageFace, cv::Mat1f eigenFaces, cv::Mat1f eigenValues,
              vector<pair<string, cv::Mat1f> > trainingFaces, vector<pair<string, cv::Mat1f> > queryFaces)
{
    // Perform PCA dimensionality reduction
    float eigenValuesSum = cv::sum(eigenValues)[0];
    float currentEigenTotal = 0;
    int count;
    char fileName[100];

    // Find the number of vectors to preserve PCA_PERCENTAGE of the information
    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenValues.rows; count++)
    {
        // y and x are swapped in cv.
        currentEigenTotal += eigenValues.at<uchar>(0, count);
    }

    cout << "Reducing Dimensionality from " << eigenFaces.cols << " to " << count << "!" << endl;

    cv::Mat1f reducedEigenFaces(eigenFaces, cv::Range::all(), cv::Range(0,count));

    // Project the faces onto the reduced eigenFaces
    vector<pair<string, cv::Mat1f> > projectedTrainingFaces, projectedQueryFaces;
    for(unsigned int i = 0; i < trainingFaces.size(); i++)
    {
        pair<string, cv::Mat1f> temp(trainingFaces[i].first,
                                     projectEigenFace(trainingFaces[i].second, averageFace, reducedEigenFaces));
        projectedTrainingFaces.push_back(temp);
    }
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        pair<string, cv::Mat1f> temp(queryFaces[i].first,
                                     projectEigenFace(queryFaces[i].second, averageFace, reducedEigenFaces));
        projectedQueryFaces.push_back(temp);
    }
    cout << "Faces projected" << endl;

    cv::Mat1f projQueryFace;
    int correct, incorrect;
    correct = incorrect = 0;
    bool querySaved = false;

    ofstream output;
    vector<float> N_Performances(50, 0);
    sprintf(fileName, "%s_%i_NImageNames.txt", resultsPath, (int)(PCA_PERCENTAGE*100));

    // Open file
    output.open(fileName);

    // Iterate through each query face and see if it can be classified correctly
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        projQueryFace = projectedQueryFaces[i].second;
        vector< pair<string, float> > queryPairs;
        querySaved = false;

        // Find the distances from this query face to every training face
        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<string, float> newPair(trainingFaces[t].first,
                                        distanceFaces(projQueryFace, projectedTrainingFaces[t].second));
            queryPairs.push_back(newPair);
        }

        // Sort the distances from least to greatest
        sort(queryPairs.begin(), queryPairs.end(), Compare);

        // Iterate from n = 1 to 50
        for(int n = 0; n < 50; n++)
        {
            if(faceIdentified(queryPairs, n + 1, projectedQueryFaces[i].first))
            {
                N_Performances[n] += 1;
                // Only save a correct match if N = 1 (0 in this case since we start at 0)
                if(correct < 3 && !querySaved && n == 0)
                {
                    output << "Correct Query Image " << correct << " ID: " << queryFaces[i].first;
                    output << " Correct Training Image " << correct << " ID: " << queryPairs[0].first;
                    output << endl << endl;
                    correct++;
                    querySaved = true;
                }
            }
            else
            {
                // Only save an incorrect match if N = 1 (0 in this case since we start at 0)
                if(incorrect < 3 && !querySaved && n == 0)
                {
                    output << "Incorrect Query Img " << incorrect << " ID: " << queryFaces[i].first;
                    output << " Incorrect Training Img " << incorrect << " ID: " << queryPairs[0].first;
                    output << endl << endl;
                    incorrect++;
                    querySaved = true;
                }
            }
        }
    }

    // Close file
    output.close();

    // Open next file
    sprintf(fileName, "%s_%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));
    output.open(fileName);

    // Print out the data for the CMC curve
    for(int n = 0; n < 50; n++)
    {
        output << n+1 << "\t" << (N_Performances[n] / (float)queryFaces.size()) << endl;
    }

    // Close file
    output.close();
}

// Runs threshold classifier!
// Varies the threshold to determine if a face can fit
// Prints the results to a series of files in directory and prints data for ROC curve for report.
void identifyThreshold(const char *resultsPath, cv::Mat1f averageFace, cv::Mat1f eigenFaces, cv::Mat1f eigenValues,
                       vector<pair<string, cv::Mat1f> > trainingFaces, vector<pair<string, cv::Mat1f> > queryFaces)
{
    // Perform PCA dimensionality reduction
    float eigenValuesSum = cv::sum(eigenValues)[0];
    float currentEigenTotal = 0;
    int count;
    char fileName[100];

    // Find the number of vectors to preserve PCA_PERCENTAGE of the information
    for(count = 0; currentEigenTotal / eigenValuesSum < PCA_PERCENTAGE && count < eigenValues.rows; count++)
    {
        // y and x are swapped in cv.
        currentEigenTotal += eigenValues.at<uchar>(0, count);
    }

    cout << "Reducing Dimensionality from " << eigenFaces.cols << " to " << count << "!" << endl;

    cv::Mat1f reducedEigenFaces(eigenFaces, cv::Range::all(), cv::Range(0,count));

    // Project the faces onto the reduced eigenFaces
    vector<pair<string, cv::Mat1f> > projectedTrainingFaces, projectedQueryFaces;
    for(unsigned int i = 0; i < trainingFaces.size(); i++)
    {
        pair<string, cv::Mat1f> temp(trainingFaces[i].first,
                                     projectEigenFace(trainingFaces[i].second, averageFace, reducedEigenFaces));
        projectedTrainingFaces.push_back(temp);
    }
    for(unsigned int i = 0; i < queryFaces.size(); i++)
    {
        pair<string, cv::Mat1f> temp(queryFaces[i].first,
                                     projectEigenFace(queryFaces[i].second, averageFace, reducedEigenFaces));
        projectedQueryFaces.push_back(temp);
    }
    cout << "Faces projected" << endl;

    cv::Mat1f projQueryFace;
    int TruePositiveCount, FalsePositiveCount;
    TruePositiveCount = FalsePositiveCount = 0;
    pair<int, int> temp(0,0);
    vector< pair<int, int> > counts(1800, temp);

    // Iterate through each query face and see if it can be classified correctly
    for(unsigned int i = 0; i < projectedQueryFaces.size(); i++)
    {
        cout << "\rQuery Face: " << i;
        projQueryFace = projectedQueryFaces[i].second;
        vector< pair<string, float> > queryPairs;

        for(unsigned int t = 0; t < trainingFaces.size(); t++)
        {
            pair<string, float> newPair(trainingFaces[t].first, distanceFaces(projQueryFace, trainingFaces[t].second));
            queryPairs.push_back(newPair);
        }

        sort(queryPairs.begin(), queryPairs.end(), Compare);
        cout << "\t" << queryPairs[0].second << endl;

        for(int threshold = 380; threshold < 1500; threshold+=2)
        {
            // Best match is less than the threshold
            if(queryPairs[0].second <= threshold)
            {
                // Check if true or false positive
                if(atoi(projectedQueryFaces[i].first.c_str()) <= 93)
                {
                    counts[threshold].first++;
                }
                else
                {
                    counts[threshold].second++;
                }
            }
        }
    }

    // Output info
    sprintf(fileName, "%s_%i.txt", resultsPath, (int)(PCA_PERCENTAGE*100));
    ofstream output;

    // Open file
    output.open(fileName);

    // Print ROC Curve info
    for(int threshold = 50; threshold < 600; threshold+=2)
    {
        float TruePositiveRate = counts[threshold].first / (float)trainingFaces.size();
        float FalsePositiveRate = counts[threshold].second / (float)(queryFaces.size() - trainingFaces.size());

        output << threshold << "\t" << TruePositiveRate<< "\t" << FalsePositiveRate << endl;
    }

    // Close file
    output.close();
}
