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

int main(int argc, const char * argv[]) {
  // Setting selection for two different videos
  Eigen::MatrixXd regions(1,4);
  CvCapture* capture;


  const char * videoPath = "defaultPath";
  if (argv[1]) {
    videoPath =  argv[1];
    cout << "\nVideo path " << argv[1] <<  "\n";
  } else {
    cout << "\nNo video specified! \n";
  }
  
  //Highway
  ROWV = 293; COLV = 316;
  MINROW = 350; MAXROW = 450;
  T1 = 50; T2 = 150;
  capture = cvCreateFileCapture(videoPath);
  regions << -60, COLV,700,MAXROW;


  
  Mat image,src,IMG;
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 50, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  Eigen::MatrixXd lines(regions.cols()-1,2);
  
  GetRegionLines(&regions,&lines,ROWV,COLV);
  
  
  while(1) {
    
    // Get frame
    src = cvQueryFrame(capture);

    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    clock_t begin = clock();
    // Process frame
    GaussianBlur(src, src, Size(5,5), 1);
    Canny(src, image, T1, T2);
    
    Eigen::MatrixXd recoveredPoints(nPoints,lines.rows()+2);
    
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    
    long nRegions = recoveredPoints.cols()-1;
    
    Eigen::MatrixXd K(nRegions,1), M(nRegions,1);
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints);

    
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    
    
    DrawTracks(&src, &K, &M,MINROW,MAXROW);
    DrawBorders(&src,1,MINROW,MAXROW,K(1,0),K(2,0),M(1,0),M(2,0));
    Eigen::MatrixXd k = lines.col(0);
    Eigen::MatrixXd m = lines.col(1);
    DrawTracks(&src, &k,&m,MINROW,MAXROW);

    
    // Show image
    imshow("SUPER MEGA ULTRA LANE DETECTION", src);
    moveWindow("SUPER MEGA ULTRA LANE DETECTION", 0, 0);
    imshow("Canny",image);
    moveWindow("Canny", 640, 0);

    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  return 0;
  }
