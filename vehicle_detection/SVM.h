//
//  SVM.h
//  DAT295
//
//  Created by Bjorn Persson Mattsson on 2015-11-30.
//  Copyright © 2015 Bjorn Persson Mattsson. All rights reserved.
//

#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>

using namespace cv;


//void SetSVMParams(CvSVMParams* params);
void SetSVMParams(CvSVMParams* params) {
    // Set up SVM's parameters
    params->svm_type    = CvSVM::C_SVC;
    //params->C = 1; params->gamma = 0.1; // validation: 0.0590035
    //params->C = 1; params->gamma = 0.01; // validation: 0.0598776
    //params->C = 0.1; params->gamma = 0.01; // validation: 0.0598776
    //params->C = 1; params->gamma = 1; // validation: 0.0598776
    //params->C = 10; params->gamma = 0.01; // validation: 0.0607517
    //params->C = 100; params->gamma = 0.001; // validation: 0.605263
    params->C = 100; params->gamma = 1000;

    params->kernel_type = CvSVM::RBF;
    params->term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 1e7, 1e-6);
}
