//
//  util.cpp
//  DAT295
//
//  Created by Bjorn Persson Mattsson on 2015-12-12.
//  Copyright © 2015 Bjorn Persson Mattsson. All rights reserved.
//

#pragma once
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <fstream>

using namespace cv;
using namespace std;


void SaveMatToFile(const char * filePath, Mat* matrix) {
	cout << "\nWriting matrix to path " << filePath << "\n\n";
    ofstream myFile(filePath);
    if (myFile.is_open()) {
        for(int i = 0; i < matrix->rows; i++){
        	for (int j=0; j<matrix->cols; j++) {
        		myFile << matrix->at<float>(i, j) << " ";
        		//cout << matrix->at<float>(i, j) << " ";
        	}
            myFile << "\n";
            //cout << "\n";
        }
        myFile.close();
    }
    else {
        cout << "Unable to save matrix to file";
    }
}
