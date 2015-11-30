//
//  TrainSVM.cpp
//  DAT295
//
//  Created by Bjorn Persson Mattsson on 2015-11-30.
//  Copyright © 2015 Bjorn Persson Mattsson. All rights reserved.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include "SVM.h"

using namespace std;
using namespace cv;



int main(int argc, const char * argv[]) {

	const char * dataPath = "defaultPath";
	const char * targetPath = "defaultPath";

	if (argc != 3) {
		cout << "Number of arguments was not two!\n";
		return -1;
	} else {
		cout << "Correct number of arguments!\n";
	}

	dataPath =  argv[1];
	cout << "\nData path " << dataPath <<  "\n";
	targetPath =  argv[2];
	cout << "\nTarget path " << targetPath <<  "\n";


	// *** The part below (extracting training data) is what needs to be changed ***
	// Set up training data
    float labels[6] = {
    	1.0, 
    	1.0, 
    	1.0, 
    	-1.0, 
    	-1.0, 
    	-1.0};
    Mat labelsMat(6, 1, CV_32FC1, labels);

    float trainingData[6][2] = {
    	{110, 200}, 
    	{105, 190}, 
    	{90, 195}, 
    	{205, 180}, 
    	{201, 199}, 
    	{210, 201} };
    Mat trainingDataMat(6, 2, CV_32FC1, trainingData);
    // *** End of what needs to be changed ***



    // Set up SVM's parameters
    CvSVMParams params;
    SetSVMParams(&params);

    // Train the SVM
    CvSVM SVM;
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);


    SVM.save(targetPath);
}
