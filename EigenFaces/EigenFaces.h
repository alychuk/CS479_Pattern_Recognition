EigenFaces.h
#ifndef EIGENRECOGNITION_EIGENFACES_H
#define EIGENRECOGNITION_EIGENFACES_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdio.h>
#include "opencv2/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


bool savedFacesExist(cv::Mat1f &averageFace, cv::Mat1f &eigenFaces, cv::Mat1f &eigenValues,
                     const char *path);

void readFaces(const char *path, std::vector<std::pair<std::string, cv::Mat1f> > &faces);

void writeEigenFace(cv::Mat1f theImage, char *fileName);

void normalizeEigenFaces(cv::Mat1f &eigenfaces, int numberOfEigenFaces);

void computeEigenFaces(std::vector<std::pair<std::string, cv::Mat1f> > trainingFaces, cv::Mat1f &averageFace,
                       cv::Mat1f &eigenFaces, cv::Mat1f &eigenValues, const char *path);

#endif //EIGENRECOGNITION_EIGENFACES_H
