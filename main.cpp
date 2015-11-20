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
#include "Functions.h"
using namespace std;
using namespace cv;


#define ROWV 293
#define COLV 316
#define MINROW 350
#define MAXROW 400


void GetRegions(arma::mat *regions,arma::mat *lines,int currentRow)
{
  regions->at(0,0) = 0;
  regions->at(0,regions->n_cols-1)= 640;
  
  for (int i=0;i<lines->n_rows;i++)
  {
    regions->at(0,i+1) = (int)(lines->at(i,0)*currentRow+lines->at(i,1));
  }
  
}

void GetRegionLines(arma::mat regions, arma::mat *lines)
{
  // Col = K * row + m
  long width = regions.n_cols;
  for (int i=0; i<regions.n_cols-1; i++) {
    
    double k = (double)(COLV-regions(0,i)) / (double)(ROWV - regions(0,width-1));
    double m = (double)(COLV - k*ROWV);
    lines->at(i,0) = k;
    lines->at(i,1) = m;
  }
  
}

void ScanImage(cv::Mat *img, arma::mat *lines,arma::mat *recoveredPoints,int nPoints,int minRow, int maxRow)
{
  double deltaRow = (maxRow-minRow) / (double)nPoints;
  int currentRow;
  
  arma::mat regions(1,lines->n_rows+2);
  for (int i=0; i<nPoints; i++) {
    currentRow = minRow + i*deltaRow;
    
    // Calculate the start and end points for the regions
    GetRegions(&regions,lines,currentRow);
    Mat slice = img->row(currentRow);
    double meanPoint;
    for (int j=0; j<lines->n_rows+1; j++) {
      Mat indices,regionSlice;
      regionSlice = slice.colRange(regions.at(0,j),regions.at(0,j+1));
       findNonZero(regionSlice,indices);
       
      if (mean(indices)[0] == 0){
          meanPoint= 0;
       } else {
          meanPoint= mean(indices)[0]+regions(0,j);
       }

       recoveredPoints->at(i,j) = (int)meanPoint;
       
      
    }
    recoveredPoints->at(i,lines->n_rows+1) = currentRow;
  }
  
  
}

void RemoveInvalidPointsV2(arma::mat *recoveredPoints,arma::field<arma::mat> *cell,int nRegions,int nPoints)
{
  
  for (int i=0; i<nRegions; i++) {
    arma::mat tmpMat(nPoints,2);
    tmpMat= arma::join_horiz(recoveredPoints->col(i),recoveredPoints->col(nRegions));
    arma::uvec index = find(tmpMat.col(0)==0);
    
    if (size(index,0) != 0) {
      for (int i=0; i<size(index,0); i++) {
        tmpMat.shed_row(index(i)-1*i);
      }
    }
    cell->at(i,0) = tmpMat;
  }
}

void LinearSolveV2(arma::field<arma::mat> *cell, arma::mat *k, arma::mat *m,int nRegions)
{
  for (int i=0; i<nRegions; i++) {
    if (size(cell->at(i,0),0)>=1)
    {
      // Solve the system
      arma::mat Y = cell->at(i,0).col(0); // Retrieved points on the sampling line
      arma::mat X(cell->at(i,0).n_rows,2); // Matrix X = [1 xn]
      
      X.ones(); X.col(1) = cell->at(i,0).col(1); // Assign matrix X the values...
      arma::mat B = solve(X,Y); // X = rows, Y = columns, B = {m,k};
      // COL = k * ROW + m
      k->at(i,0) = B(1,0);
      m->at(i,0) = B(0,0);
    }
    else
    {
      k->at(i,0) = 0;
      m->at(i,0) = 0;
    }
  }
}


int main(int argc, const char * argv[]) {
  
  
  cvNamedWindow("Example3", CV_WINDOW_AUTOSIZE);
  CvCapture* capture = cvCreateFileCapture("/Users/batko/Downloads/highway.avi");
  
  Mat image,src;
  double k1,k2,m1,m2;
  // number of line searches
  int nPoints = 30;
  int rowV,colV,minRow = 350, maxRow = 400;
  
  
  // Define the different regions of interest and get the corresponding lines
  arma::mat regions;
  regions << 20 << COLV << 620 <<MAXROW<<arma::endr;
  int nLines = regions.n_cols-1;
  
  arma::mat lines(regions.n_cols-1,2);
  GetRegionLines(regions,&lines);
  while(1) {
    // Get frame
    src = cvQueryFrame(capture);
    cvtColor(src, src, CV_BGR2GRAY);
    // Resize frame
    resize(src, src, Size(640,480),0,0,INTER_CUBIC);
    
    // Process frame
    GaussianBlur(src, src, {5,5}, 1);
//    threshold(src, image, 0.1, 1, 0);
    Canny(src, image, 50, 150);
  
  Mat newSrc = src;
  arma::mat recoveredPoints(nPoints,lines.n_rows+2);
  ScanImage(&image,&lines,&recoveredPoints,nPoints,minRow,maxRow);
    
  int nRegions = 4;
  arma::field<arma::mat> cell(nRegions,1);
  RemoveInvalidPointsV2(&recoveredPoints,&cell,nRegions,nPoints);
  
  
  arma::mat k(nRegions,1), m(nRegions,1);
  LinearSolveV2(&cell,&k,&m,nRegions);

  /*
    // Initialize matrices
    arma::mat foundPoints(nPoints,3);
    arma::mat leftPoints(nPoints,2), rightPoints(nPoints,2);
    arma::mat dummy(nPoints,3);
  
  
    // Get the points from the line search
    GetPoints(&image, nPoints, &foundPoints,1);
    GetPoints(&src,nPoints,&dummy,0);
    
    // Remove invalid points
    RemoveInvalidPoints(&foundPoints,&leftPoints,&rightPoints);
  
     Left points N x 2 : column index , row index;
     Right points M x 2 : column index, row index;
  
    
    // Linear fit to the points
    LinearSolve(&leftPoints,&k1,&m1);
    LinearSolve(&rightPoints,&k2,&m2);
    
    // Calculate vanishing point
    VanishingPoint(&rowV,&colV,k1,k2,m1,m2);
    
    // Get min and max rows of the lines
    minRow = foundPoints(foundPoints.n_rows-1,2);
    maxRow = foundPoints(0,2);
  */

  DrawTrack(&src,4,minRow,maxRow,lines.at(0,0),lines.at(0,1));
  DrawTrack(&src,4,minRow,maxRow,lines.at(1,0),lines.at(1,1));
  DrawTrack(&src,4,minRow,maxRow,lines.at(2,0),lines.at(2,1));
  DrawTrack(&src,4,minRow,maxRow,lines.at(3,0),lines.at(3,1));

    
  DrawTrack(&src,4,minRow,maxRow,k.at(0,0),m.at(0,0));
  DrawTrack(&src,4,minRow,maxRow,k.at(1,0),m.at(1,0));
  DrawTrack(&src,4,minRow,maxRow,k.at(2,0),m.at(2,0));
  DrawTrack(&src,4,minRow,maxRow,k.at(3,0),m.at(3,0));
  
    // Draw left and right track
    /*
    int leftTrack = DrawTrack(&src,leftPoints.n_rows,minRow,maxRow,k1,m1);
    int rightTrack =  DrawTrack(&src,rightPoints.n_rows,minRow,maxRow,k2,m2);
    
    bool borderCondition = (leftTrack && rightTrack);
    DrawBorders(&src,borderCondition,minRow,maxRow,k1,k2,m1,m2);
    */
    
    
    // Show image
    imshow("detected lines", src);
  
    // Key press events
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
  }
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  return 0;
  }
