#ifndef EIGENRECOGNITION_IDENTIFY_H
#define EIGENRECOGNITION_IDENTIFY_H
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;

bool Compare(pair<string, float> x, pair<string, float> y);

float distanceFaces(cv::Mat1f &originalFace, cv::Mat1f &newFace);

bool faceIdentified(vector<pair<string, float> > similarFaces, int N, string searchID);

cv::Mat1f projectEigenFace(cv::Mat1f &newFace, cv::Mat1f &averageFace, cv::Mat1f &eigenFaces);

void identify(const char *resultsPath, cv::Mat1f averageFace, cv::Mat1f eigenFaces, cv::Mat1f eigenValues,
              vector<pair<string, cv::Mat1f> > trainingFaces, vector<pair<string, cv::Mat1f> > queryFaces);

void identifyThreshold(const char *resultsPath, cv::Mat1f averageFace, cv::Mat1f eigenFaces, cv::Mat1f eigenValues,
                       vector<pair<string, cv::Mat1f> > trainingFaces, vector<pair<string, cv::Mat1f> > queryFaces);

#endif //EIGENRECOGNITION_IDENTIFY_H
