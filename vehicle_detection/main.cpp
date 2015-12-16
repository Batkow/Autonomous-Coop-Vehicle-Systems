//
//  main.cpp
//  DAT295
//
//  Created by Ivo Batkovic on 2015-11-19.
//  Copyright © 2015 Ivo Batkovic. All rights reserved.
//
#include <iostream>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include <fstream>
#include "DistanceEstimation.h"
#include <time.h>


using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;
const char * haarClassPath = "haar_classifiers/currently_used_classifier.xml";
const char * haarClassPath1 = "haar_classifiers/cars3.xml";
const char * svmPath = "trained_svm.xml";
const char * normConstPath = "svm_normalization_const.csv";
string haarImgExport = "tmp_img/";
string videoOutputPath = "recordedOutput.avi";
string videoSourcePath = "recordedSource.avi";
const std::string videoStreamAddress = "http://root:pass@192.168.0.90/axis-cgi/mjpg/video.cgi?user=USERNAME&password=PWD&channel=0&.mjpg";
const char * timestampPath= "timestamps.log"; 

bool exportHaarImages = false;
int haarExportCounter = 0;
int tmpCounter = 0;

int main(int argc, const char * argv[]) {
  cout << "\n";



  const char * videoPath = "defaultPath";
  if (argv[1]) {
    videoPath =  argv[1];
    cout << "Video path: " << argv[1] <<  "\n";
  } else {
    cout << "No video specified! \n";
  }
  cout << "Path to trained SVM: " << svmPath <<  "\n";
  
  CvCapture* capture;
  capture = cvCreateFileCapture(videoPath);

  /*
  VideoCapture capture;
  if(!capture.open(videoStreamAddress)) {
    cout << "Error opening video stream or file" << endl;
    return -1;
  }
  */

  Mat src,IMG,imageROI,ORIGSOURCE;

  // Load Haar classifier (used for generating hypotheses)
  CascadeClassifier haarClassifier = CascadeClassifier(haarClassPath);
  if (haarClassifier.empty()) {
    cout << "LOADED HAAR CLASSIFIER WAS EMPTY!!!\n";
  }
  
  // Load Feature evaluator (used for extracting Haar features)
  FileStorage fs(haarClassPath, FileStorage::READ);
  FileNode featuresNode = fs["cascade"]["features"];
  int nrOfFeatures = featuresNode.size();
  float featureVector[nrOfFeatures];
  Ptr<FeatureEvaluator> ptrHaar = FeatureEvaluator::create(FeatureEvaluator::HAAR);
  ptrHaar->read(featuresNode);

  // Load the Support Vector Machine used for verification
  CvSVM mySVM;
  mySVM.load(svmPath);
  float rescalingConstants[nrOfFeatures][2];
  ifstream normConstFile;
  normConstFile.open(normConstPath);
  if (!normConstFile) {
    cout << "Unable to open file " << normConstPath << "\n";
  }
  for (int i=0; i<nrOfFeatures; i++) {
    normConstFile >> rescalingConstants[i][0];
    normConstFile >> rescalingConstants[i][1];
    //cout << rescalingConstants[i][0] << " ";
    //cout << rescalingConstants[i][1] << "\n";
  }

  // time stamp stuff
  time_t raw_time;
  ofstream timestampFile;
  timestampFile.open(timestampPath);
  timestampFile << ctime(&raw_time) << "\n\n";


  int windowWidth = 640;
  int windowHeight = 480;

  vector<Rect> detectedVehicles;
  int roiY = 200;
  Rect regionOfInterest(0, roiY, windowWidth, windowHeight-roiY);

  initializeDistanceEstimation();

  //VideoWriter vidWriter(videoOutputPath, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(windowWidth, windowHeight), true);
  //VideoWriter vidSourceWriter(videoSourcePath, CV_FOURCC('M', 'J', 'P', 'G'), 10, Size(windowWidth, windowHeight), true);
  
  cout << "Just before video loop \n";
  time(&raw_time);
  clock_t begin = clock();
  while(1) {
    
    //cout << "Inside video loop \n";
    // Get frame
    int frameSkip = 1;
    for (int i=0; i<frameSkip; i++) {
      src = cvQueryFrame(capture);
    }
    //capture.read(src);
    resize(src, src, Size(windowWidth,windowHeight),0,0,INTER_CUBIC);
    IMG = src.clone();
    ORIGSOURCE = src.clone();
    //resize(src, src, Size(320,240),0,0,INTER_CUBIC);

    imageROI = src(regionOfInterest);
    // Process frame

    GaussianBlur(src, src, Size(3,3), 1);

    // trying Haar stuff
    //imshow("Vehicle detection", src);
    //waitKey(0);
    //cout << "Before detectMultiScale  \n";
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles);
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.03, 3, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(10, 10), Size(150, 150));
    haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.03, 1, 0, Size(10, 10), Size(150, 150));
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.1, 3, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(20, 20), Size(150, 150));
    //cout << "Just before for loop \n";
    if (exportHaarImages) {
      for (size_t i = 0; i < detectedVehicles.size(); i++) {
        Rect r = detectedVehicles[i];
        cout << "should have exported " << haarExportCounter << " images so far.\n";
        ostringstream ss;
        ss << haarImgExport << "img" << haarExportCounter << ".jpg\n";
        string myStr = ss.str();
        imwrite(myStr, imageROI(r));
        //cout << myStr;
        haarExportCounter++;
      }
    }
    for (size_t i = 0; i < detectedVehicles.size(); i++) {
      //cout << "Inside for loop \n";
      Rect r = detectedVehicles[i];
      r.y += roiY;
      // Extract Haar features and throw them into a trained SVM
      FileNodeIterator it = featuresNode.begin(), it_end = featuresNode.end();
      ptrHaar->setImage(src, Size(r.width, r.height));
      ptrHaar->setWindow(Point(r.x, r.y));
      //ptrHaar->setImage(src, Size(100, 100));
      //ptrHaar->setWindow(Point(0, 0));
      int idx = 0;
      //cout << "Before loop \n";
      while (it != it_end) { // TODO: possible to exchange this to an ordinary for loop.
        featureVector[idx] = (float)ptrHaar->calcOrd(idx);
        //cout << "Feature number " << idx << " : " << featureVector[idx];
        if (rescalingConstants[idx][1] > 0) {
          //cout << "currently rescaling feature " << idx << ": oldValue=" << featureVector[idx];
          float myMean = rescalingConstants[idx][0];
          float myStd = rescalingConstants[idx][1];
          featureVector[idx] = (featureVector[idx]-myMean)/myStd;
          //cout << "\tnewValue="<< featureVector[idx] << "\n";
          //cout << " - rescaled to -> " << featureVector[idx] << "\n";
        }
        //cout << "Feature number " << idx << " : " << featureVector[idx] << "\n";
        it++;
        idx++;
      }
      //cout << "After loop \n";
      /*
      if (abs(featureVector[140])>100 && abs(featureVector[141])>100 && abs(featureVector[142])>100) {
        cout << "This is a picture that gives weird numbers\n";
        ostringstream ss;
        ss << haarImgExport << "img" << tmpCounter << ".jpg\n";
        string myStr = ss.str();
        imwrite(myStr, src(r));
        //cout << myStr;
        tmpCounter++;
        cout << "Image saved to " << haarImgExport << "\n";
      }
      */
      
      Mat featureMat(nrOfFeatures, 1, CV_32FC1, featureVector);

      //cout << "Printing features: \n";
      //cout << featureMat << "\n";
      //cout << "Features printed... \n";
      //waitKey(0);


      // The following line will crash if wrong number of inputs to SVM!
      float response = mySVM.predict(featureMat);
      //float response = 0;
      //cout << "SVM response: " << response << "\n";
      //imshow("debug", src(r));
      //waitKey(0);

      if (response > 0) {
        // if this rectangle gets verified by SVM, draw green
        rectangle(IMG, r, Scalar(0,255,0));
      } else {
        // otherwise, if not verified, draw gray
        rectangle(IMG, r, Scalar(30,30,30));
      }

      float estDist = estimateDistance(r.x + r.width/2, windowHeight - (r.y + 0.8*r.width));
      ostringstream mySs;
      mySs << estDist << " m";
      string distText = mySs.str();
      putText(IMG, distText.c_str(), Point(r.x, r.y), FONT_HERSHEY_PLAIN, 1, Scalar(255, 255, 255));
    }
    
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    
    // display region of interest
    rectangle(IMG, regionOfInterest, Scalar(255, 0, 255));

    // Show image
    imshow("Vehicle detection", IMG);
    moveWindow("Vehicle detection", 100, 0);

    // Save to video
    //vidWriter.write(IMG);
    //vidSourceWriter.write(ORIGSOURCE);

    float timeTaken = (float) (clock()-begin)/CLOCKS_PER_SEC;
    timestampFile << timeTaken << "\t" << "\n";

    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
  }
  cout << "Just after video loop \n";
  //cvReleaseCapture(&capture);
  cvDestroyWindow("Vehicle detection");
  timestampFile.close();
  return 0;
}
