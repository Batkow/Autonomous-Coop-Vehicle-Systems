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
#include <ctime>
#include "Drawing.h"
#include "ExtractingRegions.h"
#include "ProcessImage.h"
#include "DecisionMaking.h"
#include <Eigen/Dense>
#include <armadillo>

using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;
Eigen::MatrixXd testPos(9,4);
int nPoint = 0;
int firstPoint = 1, secondPoint = 0;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  if  ( event == EVENT_LBUTTONDOWN )
  {
    cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    
    if (nPoint>=9) {
      //do nothing
    }
    else {
      
    
    if (firstPoint == 1)
    {
      testPos(nPoint,0) = x;
      testPos(nPoint,1) = y;
      firstPoint = 0;
      secondPoint = 1;
    } else if ( secondPoint == 1)
    {
      testPos(nPoint,2) = x;
      testPos(nPoint,3) = y;
      secondPoint = 0;
      firstPoint = 1;
      nPoint++;
    }
    }
  }
}

int main(int argc, const char * argv[]) {
  // Setting selection for two different videos
  Eigen::MatrixXd regions;
  CvCapture* capture;
  
  if (1)
  {
    //Rural
    ROWV = 100; COLV = 320;
    MINROW = 150; MAXROW = 300;
    T1 = 200; T2 = 300;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rural.avi");
    regions = *new Eigen::MatrixXd(7,4);
    //region = col1,row1, col2,row2
    /*regions <<  0,    250,  320,  100,
                220,  250,  320,  100,
                320,  250,  320,  100,
                420,  250,  320,  100,
                640,  250,  320,  100;
    */
    regions <<  178, 148,7,          207,
    214,          146,            3,          248,
    286,          150,          189,          301,
    321,          148,          321,          298,
    345,          150,          453,          300,
    427,          147,          622,          241,
    457,          147,          630,          213;
    
    
  } else if(0)
  {
    //Highway
    ROWV = 293; COLV = 316;
    MINROW = 340; MAXROW = 480;
    T1 = 50; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/highway.avi");
    regions = *new Eigen::MatrixXd(7,4);
    //region = col1,row1, col2,row2
    regions <<  -60,  450,  316,  293,
                60,   450,  316,  293,
                220,  450,  316,  293,
                316,  450,  316,  293,
                420,  450,  316,  293,
                580,  450,  316,  293,
                700,  450,  316,  293;
  
  } else
  {
    //Highway
    ROWV = 230; COLV = 400;
    MINROW = 280; MAXROW = 450;
    T1 = 80; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rostock.avi");
    regions = *new Eigen::MatrixXd(9,4);
    //region = col1,row1, col2,row2
    regions <<  211,  274,  4,    316,
                309,  279,  8,    425,
                335,  278,  44,   446,
                385,  281,  249,  449,
                398,  280,  337,  447,
                419,  281,  463,  452,
                465,  280,  613,  388,
                489,  278,  634,  339,
                579,  279,  636,  297;

  }
  int nFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
  arma::mat recordedArray = arma::zeros(nFrames,2);
  
  //Create a window
  namedWindow("1", 1);
  //set the callback function for any mouse event
  setMouseCallback("1", CallBackFunc, NULL);
  
  
  Mat image,src,IMG;
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 40000, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  // Each row contains k and m parameters for each line
  Eigen::MatrixXd lines(regions.rows(),2);
  GetRegionLinesV2(regions, lines);
  
  // Define previous k and m parameters used for momentum
  Eigen::MatrixXd kPrev = Eigen::MatrixXd::Zero(regions.rows()+1, 1);
  Eigen::MatrixXd mPrev = Eigen::MatrixXd::Zero(regions.rows()+1, 1);
  
  // Momentum parameter
  double alpha = 0.5;
  
  // Road parameter. TODO: Implement the nIslands to decide how many detected road markings found...
  int nTracks = 4;
  int iFrame = 0;
  while(1) {
    // Get frame
    src = cvQueryFrame(capture);
    
    if (src.empty() || iFrame == nFrames){
      cout<<"End of video file"<<endl;
      break;
    }
    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    
    // Process frame
    //cvtColor(src, IMG, CV_BGR2GRAY);
    //threshold(IMG, image, 150, 255, 0);
    GaussianBlur(src, src, Size(5,5), 1);
    Canny(src, image, T1, T2);
    
    Eigen::MatrixXd recoveredPoints(nPoints,lines.rows()+2);
    
    // Make local line searches of the image to get the points.
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    long nRegions = recoveredPoints.cols()-1;
    
    Eigen::MatrixXd K(nRegions,1), M(nRegions,1);
    Eigen::MatrixXd pointsPerRegion(nRegions,1);
    
    // Based on the points, extract the lines and get a solution based on the data
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints,&pointsPerRegion);
    
    
    Eigen::MatrixXd regionIndex(2,1);
    Eigen::VectorXd ppr = pointsPerRegion;
    Eigen::VectorXd laneLocation2 = Eigen::VectorXd::Zero(nRegions, 1);

    SelectLanesV2(ppr, laneLocation2, nTracks);

    cout<<ppr.transpose()<<endl;
    cout<<"---"<<endl;
    cout<<laneLocation2.transpose()<<endl;
    
    SelectLaneOrientation(regionIndex,laneLocation2,(int)recoveredPoints.cols());
    int p1 = regionIndex(0,0), p2 = regionIndex(1,0);
    
    //Add momentum
    AddMomentum(K,kPrev,M,mPrev,alpha,regionIndex);
    kPrev = K;
    mPrev = M;
    
    //DrawTracks(&src, &K, &M,MINROW,MAXROW,Scalar(0,0,255));
    DrawBorders(&src,1,MINROW,MAXROW,K(p1,0),K(p2,0),M(p1,0),M(p2,0));
    Eigen::MatrixXd k = lines.col(0);
    Eigen::MatrixXd m = lines.col(1);
    int midRegion = (int)(lines.rows()-1)/2;

    DrawTracks(&src, &k,&m,MINROW,MAXROW,Scalar(255,255,255));
    double laneOffset = GetLateralPosition(K(p1,0),M(p1,0),
                                           K(p2,0),M(p2,0),
                                           lines.col(0)(midRegion),lines.col(1)(midRegion),
                                           (MAXROW));
    cout<<"Lane offset:"<<laneOffset<<endl;
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    // Show image
    
    
    imshow("1", src);
    moveWindow("1", 0, 0);
    imshow("Canny",image);
    moveWindow("Canny", 640, 0);
  
    char key = (char)waitKey(0);

    if (key == 'y')
    {
      recordedArray(iFrame,0) = 1;
    } else if ( key == 27)
      break;
    
    iFrame++;
    cout<<iFrame/(double)nFrames<<endl;
    
    /*
    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
    else if (key =='s'){
      waitKey(0);
    cout<<testPos<<endl;
      waitKey(0);

    }*/
    
    //
  }
  int process = recordedArray.save("/Users/batko/Desktop/dataRural.mat",arma::raw_ascii);
  cout<<process<<endl;
  cout<<"nFrames"<<" "<<nFrames<<endl;
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  return 0;
  }
