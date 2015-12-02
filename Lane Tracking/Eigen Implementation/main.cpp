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

void SelectClosestLines(Eigen::MatrixXd *pointsPerRegion,Eigen::MatrixXd *regionIndex)
{
  int midRegion = (pointsPerRegion->rows()) / 2;
  
   //Left lane
  bool leftLaneSet = 0;
  for (int i=midRegion-1; i>0; i--) {
    if ((pointsPerRegion->col(0)(i) > pointsPerRegion->col(0)(i-1)) || pointsPerRegion->col(0)(i)>10) {
      regionIndex->col(0)(0) = i;
      leftLaneSet = 1;
      break;
    }
  }
  if (!leftLaneSet)
    regionIndex->col(0)(0)=0;
  
  //Right lane
  bool rightLaneSet = 0;
  for (int i=midRegion;i<pointsPerRegion->rows()-1;i++){
    if ( (pointsPerRegion->col(0)(i) > pointsPerRegion->col(0)(i+1)) || pointsPerRegion->col(0)(i)>10){
      regionIndex->col(0)(1) = i;
      rightLaneSet = 1;
      break;
    }
  }
  if (!rightLaneSet)
    regionIndex->col(0)(1)=pointsPerRegion->rows()-1;
}


double GetLateralPosition(double K1,double K2, double M1,double M2,double MAXROW)
{
  double laneWidth = 3.5;
  double centerLane = ((K2 * MAXROW + M2) + (K1 * MAXROW + M1) ) / 2.0;
  double pixelLaneWidth = abs((K2 * MAXROW + M2) - (K1 * MAXROW + M1) );
  double pixelsPerMeter = pixelLaneWidth / laneWidth;
  double offset = 320-centerLane;
  
  return offset / (pixelsPerMeter);
  
  
}


void AddMomentum(Eigen::MatrixXd &K, Eigen::MatrixXd &kPrev,Eigen::MatrixXd &M,Eigen::MatrixXd &mPrev,double alpha)
{
  
  K = (1-alpha) * K + alpha * kPrev;
  M = (1-alpha) * M + alpha * mPrev;
  kPrev = K;
  mPrev = M;

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
    capture = cvCreateFileCapture("/Users/amritk/Desktop/rural.avi");
    regions << 0, 320,640,250;
    
    
  } else
  {
    //Highway
    ROWV = 293; COLV = 316;
    MINROW = 340; MAXROW = 480;
    T1 = 70; T2 = 150;
    capture = cvCreateFileCapture("/Users/amritk/Desktop/highway.avi");
    //regions << -60,100,200, COLV,440,540,700,450;
    regions << -60, COLV, 700, 450;
  }
  
  Mat image,src,IMG;
  // Set nPoints...but also check so it does not exceed.
  int nPoints = 50, nMaxPoints = MAXROW-MINROW;

  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  Eigen::MatrixXd lines(regions.cols()-1,2);
  GetRegionLines(&regions,&lines,ROWV,COLV);
  
  Eigen::MatrixXd kPrev = Eigen::MatrixXd::Zero(regions.cols(), 1);
  Eigen::MatrixXd mPrev = Eigen::MatrixXd::Zero(regions.cols(), 1);
  double alpha = 0.8;
  double countWithin = 0, countIter = 0;
  double percentWithin;
    
  
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
    Eigen::MatrixXd pointsPerRegion(nRegions,1);
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints,&pointsPerRegion);
    
    AddMomentum(K,kPrev,M,mPrev,alpha);
    
    //Eigen::MatrixXd regionIndex(2,1);
    //SelectClosestLines(&pointsPerRegion,&regionIndex);
    //int p1 = regionIndex(0,0), p2 = regionIndex(1,0);
    
    int p1 = 1, p2 = 2;
    DrawTracks(&src, &K, &M,MINROW,MAXROW);
    DrawBorders(&src,1,MINROW,MAXROW,K(p1,0),K(p2,0),M(p1,0),M(p2,0));
    Eigen::MatrixXd k = lines.col(0);
    Eigen::MatrixXd m = lines.col(1);
    //DrawTracks(&src, &k,&m,MINROW,MAXROW);
    double laneOffset = GetLateralPosition(K(p1,0),K(p2,0),M(p1,0),M(p2,0),(MAXROW+MINROW) /2.0);
    if(abs(laneOffset) < 0.5)
    {
        countWithin+=1;
    }
      countIter+=1;
    percentWithin = (countWithin/countIter)*100;
    cout<<"Lane offset:"<<laneOffset<<endl;
      cout<<"Percentage:"<<percentWithin<<endl;
    //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    // Show image
    imshow("SUPER MEGA ULTRA LANE DETECTION", src);
    imwrite( "/Users/Desktop/LinesImage.jpg", src );
    moveWindow("SUPER MEGA ULTRA LANE DETECTION", 0, 0);
    imshow("Canny",image);
    moveWindow("Canny", 640, 0);
    
    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
    //
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  return 0;
  }
