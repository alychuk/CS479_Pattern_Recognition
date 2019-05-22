#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "EigenFaces.h"
#include "Identify.h"

using namespace std;
using namespace cv;

float PCA_PERCENTAGE = .80;

int main(int argc, char* argv[])
{
    vector<pair<string, Mat1f> > trainingFaces, queryFaces;
    Mat1f eigenFaces, eigenValues, averageFace;

    if(argc < 2)
    {
        cout << "Need PCA percentage! aborting" << endl;
        return 1;
    }

    PCA_PERCENTAGE = atof(argv[1]);

    // Part A

    // Enter path as "../DirectoryPath/*.pgm
    readFaces("../Faces_FA_FB/fa_H/*.pgm", trainingFaces);
    readFaces("../Faces_FA_FB/fb_H/*.pgm", queryFaces);


    cout << "Reading" << endl;
    if(!savedFacesExist(averageFace, eigenFaces, eigenValues, "../Faces_FA_FB/fa_H")) // faces haven't been computed yet
    {
        cout << "No saved faces. Computing..." << endl;

        computeEigenFaces(trainingFaces, averageFace, eigenFaces, eigenValues, "../Faces_FA_FB/fa_H");
    }
    normalizeEigenFaces(eigenFaces, trainingFaces.size());
    cout << "Done." << endl;

    // Write average face to file
    writeEigenFace(averageFace, "averageFace.pgm");


    identify("../N-Results/NData", averageFace, eigenFaces, eigenValues, trainingFaces, queryFaces);


    // Print top 10 eigenvalues
    char faceFileName[150];
    for(int i = 0; i < 10; i++)
    {
        sprintf(faceFileName, "Part_A_largestFace%i.pgm", i + 1);
        writeEigenFace(eigenFaces.col(i), faceFileName);
    }

    // Print top 10 eigenvalues
    for(int i = eigenFaces.cols - 1; i > eigenFaces.cols - 1 - 10; i--)
    {
        sprintf(faceFileName, "Part_A_smallestFace%i.pgm", i - eigenFaces.cols + 2);
        writeEigenFace(eigenFaces.col(i), faceFileName);
    }

    // Part B

    trainingFaces.clear();
    queryFaces.clear();

    readFaces("../Faces_FA_FB/fa2_H/*.pgm", trainingFaces);
    readFaces("../Faces_FA_FB/fb_H/*.pgm", queryFaces);

    cout << "Reading Part B" << endl;
    if(!savedFacesExist(averageFace, eigenFaces, eigenValues, "fa2_H")) // faces haven't been computed yet
    {
        cout << "No saved faces. Computing..." << endl;

        computeEigenFaces(trainingFaces, averageFace, eigenFaces, eigenValues, "fa2_H");
    }

    normalizeEigenFaces(eigenFaces, trainingFaces.size());

    writeEigenFace(averageFace, "averageFace_PartB.pgm");


    PCA_PERCENTAGE = .95;

    identifyThreshold("../B_Results/BData", averageFace, eigenFaces, eigenValues, trainingFaces, queryFaces);

    return 0;
}
