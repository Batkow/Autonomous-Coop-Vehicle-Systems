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
#include <Eigen/Dense>

using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;


double GetLateralPosition(double K1,double K2, double M1,double M2,int MAXROW)
{
  double laneWidth = 3.5;
  double centerLane = ((K2 * MAXROW + M2) + (K1 * MAXROW + M1) ) / 2.0;
  return laneWidth / (320-centerLane);
}

int main(int argc, const char * argv[]) {
  // Setting selection for two different videos
  Eigen::MatrixXd regions(1,4);
  CvCapture* capture;
  
  if (0)
  {
    //Rural
    ROWV = 100; COLV = 320;
    MINROW = 150; MAXROW = 300;
    T1 = 200; T2 = 300;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rural.avi");
    regions << 0, 320,640,250;
    
    
  } else
  {
    //Highway
    ROWV = 293; COLV = 316;
    MINROW = 320; MAXROW = 480;
    T1 = 50; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/highway.avi");
    regions << -60, COLV,700,450;
  }
  
  Mat image,src,IMG;
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 100, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  Eigen::MatrixXd lines(regions.cols()-1,2);
  
  GetRegionLines(&regions,&lines,ROWV,COLV);
  
  while(1) {
    //clock_t begin = clock();
    // Get frame
    src = cvQueryFrame(capture);

    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    
    // Process frame
    GaussianBlur(src, src, Size(5,5), 1);
    Canny(src, image, T1, T2);
    
    Eigen::MatrixXd recoveredPoints(nPoints,lines.rows()+2);
    
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    
    long nRegions = recoveredPoints.cols()-1;
    
    Eigen::MatrixXd K(nRegions,1), M(nRegions,1);
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints);

    
    
    
    
    //DrawTracks(&src, &K, &M,MINROW,MAXROW);
    DrawBorders(&src,1,MINROW,MAXROW,K(1,0),K(2,0),M(1,0),M(2,0));
    Eigen::MatrixXd k = lines.col(0);
    Eigen::MatrixXd m = lines.col(1);
    //DrawTracks(&src, &k,&m,MINROW,MAXROW);
    double laneOffset = GetLateralPosition(K(1,0),K(2,0),M(1,0),M(2,0),MAXROW);
    cout<<"Lane offset:"<<laneOffset<<endl;
    // Show image
    imshow("SUPER MEGA ULTRA LANE DETECTION", src);
    moveWindow("SUPER MEGA ULTRA LANE DETECTION", 0, 0);
    imshow("Canny",image);
    moveWindow("Canny", 640, 0);

    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  return 0;
  }
