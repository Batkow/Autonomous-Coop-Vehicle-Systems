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
#include <armadillo>
#include <ctime>
#include "Drawing.h"
#include "ExtractingRegions.h"
#include "ProcessImage.h"

using namespace std;
using namespace cv;

int ROWV,COLV,MINROW,MAXROW,T1,T2;

int main(int argc, const char * argv[]) {
  // Setting selection for two different videos
  arma::mat regions;
  CvCapture* capture;
  
  if (0)
  {
    //Rural
    ROWV = 100; COLV = 320;
    MINROW = 150; MAXROW = 300;
    T1 = 200; T2 = 300;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rural.avi");
    regions << 0 << 320 << 640 <<250<<arma::endr;
    
  } else
  {
    //Rural
    ROWV = 293; COLV = 316;
    MINROW = 350; MAXROW = 450;
    T1 = 50; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/highway.avi");
    regions << -60 << COLV << 700 <<MAXROW<<arma::endr;
  }
  
  
  Mat image,src,IMG;
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 50, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  arma::mat lines(regions.n_cols-1,2);
  GetRegionLines(regions,&lines,ROWV,COLV);

  while(1) {
    
    // Get frame
    src = cvQueryFrame(capture);

    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    clock_t begin = clock();
    // Process frame
    GaussianBlur(src, src, Size(5,5), 1);
    Canny(src, image, T1, T2);
    
    arma::mat recoveredPoints(nPoints,lines.n_rows+2);
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    
    long nRegions = recoveredPoints.n_cols-1;
    arma::field<arma::mat> cell(nRegions,1);
    RemoveInvalidPoints(&recoveredPoints,&cell,nRegions,nPoints);
    
    arma::mat k(nRegions,1), m(nRegions,1);
    LinearSolve(&cell,&k,&m,nRegions);
    
    cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    
    DrawTracks(&src, &k, &m,MINROW,MAXROW);
    arma::mat K = lines.col(0);
    arma::mat M = lines.col(1);
    DrawTracks(&src, &K,&M,MINROW,MAXROW);
    DrawBorders(&src,1,MINROW,MAXROW,k(1,0),k(2,0),m(1,0),m(2,0));
    
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
