//
//  SVM.cpp
//  DAT295
//
//  Created by Bjorn Persson Mattsson on 2015-11-29.
//  Copyright © 2015 Bjorn Persson Mattsson. All rights reserved.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include "SVM.h"

using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {

	// Data for visual representation
	int width = 512;
	int height = 512;
	Mat image = Mat::zeros(height, width, CV_8UC3);

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


    if (argc != 2) {
        cout << "No trained SVM was specified!\n";
        return -1;
    } else {
        cout << "Correct number of arguments!\n";
    }

    const char * pathToTrainedSVM = "defaultPath";
    pathToTrainedSVM =  argv[1];
    cout << "\nPath to trained SVM: " << pathToTrainedSVM <<  "\n";

    CvSVM SVM;
    SVM.load(pathToTrainedSVM);


    Vec3b green(0,255,0), blue (255,0,0);
    // Show the decision regions given by the SVM
    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            Mat sampleMat = (Mat_<float>(1,2) << j,i);
            float response = SVM.predict(sampleMat);

            if (response == 1)
                image.at<Vec3b>(i,j)  = green;
            else if (response == -1)
                 image.at<Vec3b>(i,j)  = blue;
        }

    // Show the training data
    int thickness = -1;
    int lineType = 8;
    int nrElems = sizeof(trainingData)/sizeof(trainingData[0]);
    for (int i=0; i<nrElems; i++) {
    	circle( image, Point((int)trainingData[i][0], (int)trainingData[i][1]), 5, Scalar(  0,   0,   0), thickness, lineType);
    }
    //circle( image, Point(201,  10), 5, Scalar(  0,   0,   0), thickness, lineType);
    //circle( image, Point(205,  10), 5, Scalar(255, 255, 255), thickness, lineType);
    //circle( image, Point(201, 205), 5, Scalar(255, 255, 255), thickness, lineType);
    //circle( image, Point( 10, 201), 5, Scalar(255, 255, 255), thickness, lineType);

    // Show support vectors
    thickness = 2;
    lineType  = 8;
    int c     = SVM.get_support_vector_count();

    for (int i = 0; i < c; ++i)
    {
        const float* v = SVM.get_support_vector(i);
        //circle( image,  Point( (int) v[0], (int) v[1]),   6,  Scalar(128, 128, 128), thickness, lineType);
    }

    //imwrite("result.png", image);        // save the image

    imshow("SVM Simple Example", image); // show it to the user
    waitKey(0);
}