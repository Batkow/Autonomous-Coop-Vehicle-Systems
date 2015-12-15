//
//  main.cpp
//  DAT295
//
//  Created by Ivo Batkovic on 2015-11-19.
//  Copyright © 2015 Ivo Batkovic. All rights reserved.
//
#include <iostream>
//#include <sstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include <Eigen/Dense>
#include <fstream>

using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;
const char * haarClassPath = "haar_classifiers/currently_used_classifier.xml";
const char * haarClassPath1 = "haar_classifiers/cars3.xml";
const char * svmPath = "trained_svm.xml";
const char * normConstPath = "svm_normalization_const.csv";
bool exportHaarImages = false;
string haarImgExport = "tmp_img/";
int haarExportCounter = 0;
int tmpCounter = 0;

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

  int windowWidth = 800;
  int windowHeight = 600;

  vector<Rect> detectedVehicles;
  int roiY = 200;
  Rect regionOfInterest(0, roiY, windowWidth, windowHeight-roiY);
  
  cout << "Just before video loop \n";
  while(1) {
    
    //cout << "Inside video loop \n";
    // Get frame
    int frameSkip = 5;
    for (int i=0; i<frameSkip; i++) {
      src = cvQueryFrame(capture);
    }
    resize(src, src, Size(windowWidth,windowHeight),0,0,INTER_CUBIC);
    IMG = src.clone();
    //resize(src, src, Size(320,240),0,0,INTER_CUBIC);

    imageROI = src(regionOfInterest);
    clock_t begin = clock();
    // Process frame

    GaussianBlur(src, src, Size(3,3), 1);
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
    //haarClassifier.detectMultiScale(imageROI, detectedVehicles, 1.01, 1, 0 | CASCADE_FIND_BIGGEST_OBJECT, Size(5, 5), Size(150, 150));
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
      cout << "After loop \n";
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
      cout << "SVM response: " << response << "\n";
      imshow("debug", src(r));
      //waitKey(0);

      if (response > 0) {
        // if this rectangle gets verified by SVM, draw green
        rectangle(IMG, r, Scalar(0,255,0));
      } else {
        // otherwise, if not verified, draw gray
        rectangle(IMG, r, Scalar(30,30,30));
      }
    }
    
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    
    // display region of interest
    rectangle(IMG, regionOfInterest, Scalar(255, 0, 255));

    // Show image
    imshow("Vehicle detection", IMG);
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
