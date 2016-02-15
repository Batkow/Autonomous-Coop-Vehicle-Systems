//
//  TrainSVM.cpp
//  DAT295
//
//  Created by Bjorn Persson Mattsson on 2015-11-30.
//  Copyright © 2015 Bjorn Persson Mattsson. All rights reserved.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include "SVM.h"
#include "util.cpp"
#include <math.h>
#include <fstream>
#include <vector>

using namespace std;
using namespace cv;


float calcMean(Mat input) {
    int nrElems = input.cols * input.rows;
    double sum = 0;
    for (int i=0; i<nrElems; i++) {
        sum += input.at<float>(i);
    }
    //cout << "Inside calcMean(): sum=" << sum << ", nrElems=" << nrElems << ", mean=" << sum/nrElems << "\n";
    return (float)(sum/nrElems);
}

float calcStd(Mat input, float mean) {
    int nrElems = input.cols * input.rows;
    double sum = 0;
    for (int i=0; i<nrElems; i++) {
        sum += pow(input.at<float>(i) - mean, 2);
    }
    return (float)sqrt(sum/nrElems);
}


int main(int argc, const char * argv[]) {

	const char * dataPath = "defaultPath";
    const char * targetPath = "defaultPath";
    const char * normConstPath = "svm_normalization_const.csv";

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

    // Load Haar classifier from file
    const char * haarClassPath = "haar_classifiers/currently_used_classifier.xml";
    const char * posSamplesPath = "positive_images";
    const char * negSamplesPath = "negative_images";
    CascadeClassifier haarClassifier = CascadeClassifier(haarClassPath);
    if (haarClassifier.empty()) {
    cout << "LOADED HAAR CLASSIFIER WAS EMPTY!!!\n";
    }
    int nrFeatures = -1;
    // load Haar classifier and check how many features will be extracted
    FileStorage fs(haarClassPath, FileStorage::READ);
    FileNode featuresNode = fs["cascade"]["features"];
    nrFeatures = featuresNode.size();
    cout << "Number of Haar features: " << nrFeatures << "\n";
    Ptr<FeatureEvaluator> ptrHaar = FeatureEvaluator::create(FeatureEvaluator::HAAR);
    ptrHaar->read(featuresNode);
    int imgWidth = 40;
    int imgHeight = 40;

    vector<String> posFileNames;
    vector<String> negFileNames;
    glob(posSamplesPath, posFileNames);
    glob(negSamplesPath, negFileNames);

    // total training size = sum of #positives and #negatives
    int totalTrainingSetSize = posFileNames.size() + negFileNames.size();
    cout << "Training data consists of " << totalTrainingSetSize << " data points: " << posFileNames.size() << " positives and " << negFileNames.size() << " negatives.\n";


    if (posFileNames.size() > 0 && negFileNames.size() > 0) {
        // valid
    } else {
        // non-valid!
        cout << "WARNING: There are either no positive samples or no negative samples. Aborting!\n";
        return -1;
    }

    RNG rng(getTickCount());


    // initialise trainingData matrix and labels vector
    float labels[totalTrainingSetSize];
    float trainingData[totalTrainingSetSize][nrFeatures];
    float featureVector[nrFeatures];

    Mat image;
    int dataPointCounter = 0;

    // for every positive file
    //   load it and run the haar classifier on it
    //   store Haar features as a row in trainingData matrix
    //   set label to 1 (for positive)
    // end for
    //cout << "Found positive files:\n";
    for (int i=0; i<posFileNames.size(); i++) {
        FileNodeIterator it = featuresNode.begin(), it_end = featuresNode.end();
        //cout << "  pos " << posFileNames[i] << "\n";
        image = imread(posFileNames[i]);
        //resize(image, image, Size(imgWidth, imgHeight), 0, 0, INTER_CUBIC);
        ptrHaar->setImage(image, image.size());
        //ptrHaar->setImage(image, Size(imgWidth, imgHeight));
        ptrHaar->setWindow(Point(0, 0));
        int idx = 0;
        while (it != it_end) {
            trainingData[dataPointCounter][idx] = (float)ptrHaar->calcOrd(idx);
            //featureVector[idx] = (float)ptrHaar->calcOrd(idx);
            //cout << "Feature number " << idx << " : " << ptrHaar->calcOrd(idx) << "\n";
            it++;
            idx++;
        }
        //for (int j=0; j<featureVector.size(); j++) {
        //    trainingData[i][j] = featureVector[j];
        //}
        labels[dataPointCounter] = 1;
        dataPointCounter++;
        imshow("Positives", image);

    }
    //cout << "\n";
    //waitKey(0);
    // for every negative file
    //   load it and run the haar classifier on it
    //   store Haar features as a row in trainingData matrix
    //   set label to -1 (for positive)
    // end for
    for (int i=0; i<negFileNames.size(); i++) {
        FileNodeIterator it = featuresNode.begin(), it_end = featuresNode.end();
        //cout << "  neg " << negFileNames[i] << "\n";
        image = imread(negFileNames[i]);

        // find a squared subpart (side larger than 40) of the image
        int left = rng.uniform(0.f, 1.f)*(image.cols - imgWidth);
        int top = rng.uniform(0.f, 1.f)*(image.rows - imgHeight);
        int maxWidth = image.cols - left;
        int maxHeight = image.rows - top;
        int maxSide = std::min(maxWidth, maxHeight);
        //int sampleSide = std::max(imgWidth, (int)(rng.uniform(0.f, 1.f)*maxSide));
        int sampleSide = rng.uniform(0.f, 1.f)*(maxSide-imgWidth) + imgWidth;
        //cout << "\n\ndebug: " << rng.uniform(0.f, 1.f)*maxSide << "\n\n";
        //cout << "\n\ndebug: " << max(5,rng.uniform(0.f, 1.f)*maxSide) << "\n\n";
        //int sampleSide = rng.uniform(0.f, 1.f)*maxSide;
        int right = left + sampleSide;
        int bottom = top + sampleSide;
        // crop that subpart and use it as "image" below
        //cout << "My subsample is: (" << left << ", " << top << ", " << right << ", " << bottom << ")\n";
        Rect sampleSubpart(left, top, sampleSide, sampleSide);
        image = image(sampleSubpart);

        ptrHaar->setImage(image, image.size());
        //ptrHaar->setImage(image, Size(imgWidth, imgHeight));
        /*
        float rnd1 = rng.uniform(0.f, 1.f);
        float rnd2 = rng.uniform(0.f, 1.f);
        ptrHaar->setWindow(Point(rnd1*(image.cols-imgWidth), rnd2*(image.rows-imgHeight)));
        */
        ptrHaar->setWindow(Point(0, 0));
        int idx = 0;
        while (it != it_end) {
            trainingData[dataPointCounter][idx] = (float)ptrHaar->calcOrd(idx);
            //featureVector[idx] = (float)ptrHaar->calcOrd(idx);
            //cout << "Feature number " << idx << " : " << ptrHaar->calcOrd(idx) << "\n";
            it++;
            idx++;
        }
        //for (int j=0; j<featureVector.size(); j++) {
        //    trainingData[i][j] = featureVector[j];
        //}
        labels[dataPointCounter] = -1;
        dataPointCounter++;
        imshow("Negatives", image);
    }
    //waitKey(0);


    Mat tmpLabelsMat(totalTrainingSetSize, 1, CV_32FC1, labels);
    Mat tmpDataMat(totalTrainingSetSize, nrFeatures, CV_32FC1, trainingData);


    float partValidation = 0.1;
    int nrValidationData = (int)(totalTrainingSetSize*partValidation);
    cout << "Nr data used for validation: " << nrValidationData << endl;
    vector<int> randomIndices;
    for (int i=0; i<totalTrainingSetSize; ++i) randomIndices.push_back(i);
    random_shuffle(randomIndices.begin(), randomIndices.end());
    
    Mat validationLabelsMat(nrValidationData, 1, CV_32FC1);
    Mat validationDataMat(nrValidationData, nrFeatures, CV_32FC1);
    Mat labelsMat(totalTrainingSetSize-nrValidationData, 1, CV_32FC1);
    Mat trainingDataMat(totalTrainingSetSize-nrValidationData, nrFeatures, CV_32FC1);
    for (int i=0; i<nrValidationData; i++) {
        // put from tmp data/labels into validation
        validationLabelsMat.at<float>(i) = tmpLabelsMat.at<float>(randomIndices.at(i));
        for (int j=0; j<nrFeatures; j++) {
            validationDataMat.at<float>(i, j) = tmpDataMat.at<float>(randomIndices.at(i), j);
        }
    }
    for (int i=nrValidationData; i<totalTrainingSetSize; i++) {
        // put from tmp data/labels into training
        labelsMat.at<float>(i-nrValidationData) = tmpLabelsMat.at<float>(randomIndices.at(i));
        for (int j=0; j<nrFeatures; j++) {
            trainingDataMat.at<float>(i-nrValidationData, j) = tmpDataMat.at<float>(randomIndices.at(i), j);
        }
    }




    // rescale the data and store subtraction and scaling constants for each feature to file.
    float rescalingConstants[nrFeatures][2];
    for (int j=0; j<trainingDataMat.cols; j++) {
        // for every feature, calculate mean and stddev, and then standardize the data
        //cout << "Before\n";
        //meanStdDev(trainingDataMat.col(j), m, stdv);
        double featureMean = calcMean(trainingDataMat.col(j));
        double featureStd = calcStd(trainingDataMat.col(j),  featureMean);
        //cout << "After\n";
        //cout << "Feature " << j << ". (mean, std) = (" << featureMean << ", " << featureStd << ")";
        for (int i=0; i<trainingDataMat.rows; i++) {
            double prevValue = trainingDataMat.at<float>(i,j);
            double newValue = prevValue;
            if (featureStd>0) {
                newValue = (prevValue - featureMean)/featureStd;
            }
            //cout << i << ": (prev, new) - (" << prevValue << ", " << newValue << ")\n";
            //waitKey(0);
            trainingDataMat.at<float>(i,j) = (float)newValue;
        }

        rescalingConstants[j][0] = featureMean;
        rescalingConstants[j][1] = featureStd;

        
        //meanStdDev(trainingDataMat.col(j), m, stdv);
        featureMean = calcMean(trainingDataMat.col(j));
        featureStd = calcStd(trainingDataMat.col(j),  featureMean);
        //cout << "\t -> (" << featureMean << ", " << featureStd << ")\n";        
    }

    ofstream normConstFile(normConstPath);
    if (normConstFile.is_open()) {
        for(int i = 0; i < nrFeatures; i++){
            normConstFile << rescalingConstants[i][0] << " " << rescalingConstants[i][1] << "\n";
        }
        normConstFile.close();
    }
    else {
        cout << "Unable to save normalisation constants to file";
    }
/*
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
*/

    // *** End of what needs to be changed ***

    SaveMatToFile("svmTrainingData.csv", &trainingDataMat);
    SaveMatToFile("svmTrainingLabels.csv", &labelsMat);


    // Set up SVM's parameters
    CvSVMParams params;
    SetSVMParams(&params);

    // Train the SVM
    cout << "Training the SVM!\n";
    CvSVM SVM;
    int kfold = 5;
    CvParamGrid cGrid(0.1, 30, 10);
    CvParamGrid gammaGrid(0.01, 5, 10);
    SVM.train(trainingDataMat, labelsMat, Mat(), Mat(), params);
    //SVM.train_auto(trainingDataMat, labelsMat, Mat(), Mat(), params, kfold, cGrid, gammaGrid);

    cout << "SVM parameters:\n";
    cout << "C: " << SVM.get_params().C << "\n";
    cout << "gamma: " << SVM.get_params().gamma << "\n";


    SVM.save(targetPath);

    float correctlyClassified = 0;
    float truePositives = 0;
    float trueNegatives = 0;
    float falsePositives = 0;
    float falseNegatives = 0;
    for (int i=0; i<validationDataMat.rows; i++) {
        Mat input = validationDataMat.row(i);
        float expectedResult = validationLabelsMat.at<float>(i);
        float result = SVM.predict(input);
        if ((result < 0 && expectedResult < 0) || (result > 0 && expectedResult > 0)) {
            correctlyClassified++;
        }
        if (result < 0 && expectedResult < 0) {
            trueNegatives++;
        }
        if (result > 0 && expectedResult > 0) {
            truePositives++;
        }
        if (result < 0 && expectedResult > 0) {
            falseNegatives++;
        }
        if (result > 0 && expectedResult < 0) {
            falsePositives++;
        }
    }
    cout << "\n\nClassification rate on validation data: ";
    cout << correctlyClassified << "/" << nrValidationData << " = " << (correctlyClassified/nrValidationData) << "\n";
    cout << "\n\nFalse positives (should be low): ";
    cout << falsePositives << "/" << (trueNegatives + falsePositives) << " = " << (falsePositives/(trueNegatives + falsePositives)) << "\n";
    cout << "\n\nFalse negatives (should be low): ";
    cout << falseNegatives << "/" << (truePositives + falseNegatives) << " = " << (falseNegatives/(truePositives + falseNegatives)) << "\n";
}
