//
//  main.cpp
//  DAT295
//
//  Created by Ivo Batkovic on 2015-11-19.
//  Copyright © 2015 Ivo Batkovic. All rights reserved.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;
const char * haarClassPath = "haar_classifiers/currently_used_classifier.xml";
const char * svmPath = "trained_svm.xml";

int main(int argc, const char * argv[]) {
  cout << "\n";
  // Setting selection for two different videos
  Eigen::MatrixXd regions(1,4);
  CvCapture* capture;


  const char * videoPath = "defaultPath";
  if (argv[1]) {
    videoPath =  argv[1];
    cout << "Video path: " << argv[1] <<  "\n";
  } else {
    cout << "No video specified! \n";
  }
  cout << "Path to trained SVM: " << svmPath <<  "\n";
  
  //Highway
  ROWV = 293; COLV = 316;
  MINROW = 350; MAXROW = 450;
  T1 = 50; T2 = 150;
  capture = cvCreateFileCapture(videoPath);
  regions << -60, COLV,700,MAXROW;


  Mat src,IMG,imageROI;


  // Load Haar classifier (used for generating hypotheses)
  CascadeClassifier haarClassifier = CascadeClassifier(haarClassPath);
  if (haarClassifier.empty()) {
    cout << "LOADED HAAR CLASSIFIER WAS EMPTY!!!\n";
  }
  
  // Load Feature evaluator (used for extracting Haar features)
  FileStorage fs(haarClassPath, FileStorage::READ);
  FileNode featuresNode = fs["cascade"]["features"];
  int nrOfFeatures = featuresNode.size();
  double featureVector[nrOfFeatures];
  Ptr<FeatureEvaluator> ptrHaar = FeatureEvaluator::create(FeatureEvaluator::HAAR);
  ptrHaar->read(featuresNode);

  // Load the Support Vector Machine used for verification
  CvSVM mySVM;
  mySVM.load(svmPath);

  /*
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 50, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  Eigen::MatrixXd lines(regions.cols()-1,2);
  
  GetRegionLines(&regions,&lines,ROWV,COLV);
  */
  
  //cout << "Number of Haar features:  " << nrOfFeatures << " \n";
  //cout << "Length of feature vector: " << sizeof(featureVector)/sizeof(featureVector[0]) << " \n";

  vector<Rect> detectedVehicles;
  int roiY = 200;
  Rect regionOfInterest(0, roiY, 640, 480-roiY);
  
  cout << "Just before video loop \n";
  while(1) {
    
    //cout << "Inside video loop \n";
    // Get frame
    int frameSkip = 2;
    for (int i=0; i<frameSkip; i++) {
      src = cvQueryFrame(capture);
    }
    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    //resize(src, src, Size(320,240),0,0,INTER_CUBIC);

    imageROI = src(regionOfInterest);
    clock_t begin = clock();
    // Process frame

    //GaussianBlur(src, src, Size(3,3), 1);
    /*
    GaussianBlur(src, src, Size(5,5), 1);
    Canny(src, image, T1, T2);
    
    Eigen::MatrixXd recoveredPoints(nPoints,lines.rows()+2);
    
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    
    long nRegions = recoveredPoints.cols()-1;
    
    Eigen::MatrixXd K(nRegions,1), M(nRegions,1);
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints);
    */

    // trying Haar stuff
    //imshow("Vehicle detection", src);
    //waitKey(0);
    //cout << "Before detectMultiScale  \n";
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles);
    //haarClassifier.detectMultiScale(src, detectedVehicles, 1.05, 1, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(20, 20), Size(150, 150));
    haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.01, 3, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(10, 10), Size(150, 150));
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.1, 3, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(20, 20), Size(150, 150));
    //cout << "Just before for loop \n";
    for (size_t i = 0; i < detectedVehicles.size(); i++)
    {
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
      while (it != it_end) {
        featureVector[idx] = ptrHaar->calcOrd(idx);
        //cout << "Feature number " << idx << " : " << res << "\n";
        it++;
        idx++;
      }
      //cout << "After loop \n";
      Mat featureMat(nrOfFeatures, 1, CV_32FC1, featureVector);
      // The following line will crash if wrong number of inputs to SVM!
      float response = mySVM.predict(featureMat);
      //float response = 0;
      //cout << "SVM response: " << response << "\n";

      if (response > 0) {
        // if this rectangle gets verified by SVM, draw green
        rectangle(src, r, Scalar(0,255,0));
      } else {
        // otherwise, if not verified, draw red
        //rectangle(src, r, Scalar(0,0,255));
      }
    }
    
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    
    // display region of interest
    rectangle(src, regionOfInterest, Scalar(255, 0, 255));

    // Show image
    imshow("Vehicle detection", src);
    moveWindow("Vehicle detection", 100, 0);

    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
  }
  cout << "Just after video loop \n";
  cvReleaseCapture(&capture);
  cvDestroyWindow("Vehicle detection");
  return 0;
  }
