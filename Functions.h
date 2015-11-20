//
//  Functions.h
//  DAT295
//
//  Created by Ivo Batkovic on 2015-11-19.
//  Copyright © 2015 Ivo Batkovic. All rights reserved.
//
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <armadillo>

using namespace cv;

// Draw track lines
void MyLine( Mat img, Point start, Point end );

// Get row and column position for a line segment
void GetPoints(Mat *img, int nPoints, arma::mat *meanPoints, int booleanVar);

// Remove points without any found lines
void RemoveInvalidPoints(arma::mat *foundPoints, arma::mat *leftPoints, arma::mat *rightPoints);

// Fit the solution as col = row * k + m
void LinearSolve(arma::mat *points,double *k, double *m);

// Find the vanishing point
void VanishingPoint(int *rowV, int *colV,double k1,double k2,double m1,double m2);

// Draw the track
int DrawTrack(Mat *src,int nPoints,int minRow, int maxRow, double k1, double m1);

// Draw the borders
void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2);


/*
 FUNCTIONS HERE
*/

void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2)
{
  if (borderCondition)
  {
    MyLine(*src,Point((maxRow*k1+m1),maxRow), Point((maxRow*k2+m2),maxRow));
    MyLine(*src,Point((minRow*k1+m1),minRow), Point((minRow*k2+m2),minRow));
  }
  
}


int DrawTrack(Mat *src,int nPoints,int minRow, int maxRow, double k1, double m1)
{
  if (nPoints != 0) {
    
    MyLine(*src,Point((maxRow*k1+m1),maxRow), Point((minRow*k1+m1),minRow));
    return 1;
    
  } else {
    
    return 0;
  }
  
  
  
  
}

void VanishingPoint(int *rowV, int *colV,double k1,double k2,double m1,double m2)
{
  double xIntersect = (m2-m1) / (k1-k2);
  double yIntersect = k1 * xIntersect + m1;
  *rowV = (int)xIntersect, *colV = (int)yIntersect;
}



void LinearSolve(arma::mat *points, double *k, double *m)
{
  // Solve the system
  arma::mat Y = points->col(0); // Retrieved points on the sampling line
  arma::mat X(points->n_rows,2); // Matrix X = [1 xn]
  
  X.ones(); X.col(1) = points->col(1); // Assign matrix X the values...
  arma::mat B = solve(X,Y); // X = rows, Y = columns, B = {m,k};
  // COL = k * ROW + m
  *k = B(1,0);
  *m = B(0,0);
  
}

void RemoveInvalidPoints(arma::mat *foundPoints,arma::mat *leftPoints, arma::mat *rightPoints)
{
  
  leftPoints->col(0) = foundPoints->col(0);
  leftPoints->col(1) = foundPoints->col(2);
  
  rightPoints->col(0) = foundPoints->col(1);
  rightPoints->col(1) = foundPoints->col(2);
  
  
  
  arma::uvec leftTest = find(leftPoints->col(0)==0);
  arma::uvec rightTest = find(rightPoints->col(0)==0);
  if (size(leftTest,0) != 0) {
    for (int i=0; i<size(leftTest,0); i++) {
      leftPoints->shed_row(leftTest(i)-1*i);
    }
  }
  
  if (size(rightTest,0) != 0) {
    for (int i=0; i<size(rightTest,0); i++) {
      rightPoints->shed_row(rightTest(i)-1*i);
    }
  }
}

void GetPoints(Mat *img, int nPoints, arma::mat *meanPoints,int booleanVar)
{
  Mat slice, leftSlice, rightSlice, leftPoint, rightPoint;
  int width = (int)img->cols/2.2;
  double scaling = 0.6*(width) /(double)nPoints;
  
  double offset = 0;
  double meanLeftPoint, meanRightPoint;
  int colStart, colEnd, currentRow;
  for (int i = 0; i<nPoints;i++)
  {
    colStart = (int)(img->cols/2 - width + i*scaling + offset);
    
    colEnd = (int)(img->cols/2 + width -i*scaling + offset);
    currentRow = (int)(img->rows * 0.9) - i/(double)nPoints * img->rows * 0.2;
    
    if (booleanVar == 1)
    {
      slice = img->row(currentRow).colRange(colStart,colEnd);
      leftSlice = slice.colRange(1, slice.size[1]/2);
      rightSlice = slice.colRange(slice.size[1]/2,slice.size[1]);
      
      findNonZero(leftSlice, leftPoint);
      findNonZero(rightSlice, rightPoint);
      
      if (mean(leftPoint)[0] == 0){
        meanLeftPoint = 0;
      } else {
        meanLeftPoint = mean(leftPoint)[0]+colStart;
      }
      
      if (mean(rightPoint)[0] == 0){
        meanRightPoint = 0;
      } else {
        meanRightPoint = mean(rightPoint)[0]+colStart+slice.size[1]/2;
      }
      
      
      meanPoints->row(i).col(0) = (int)meanLeftPoint;
      meanPoints->row(i).col(1) = (int)meanRightPoint;
      meanPoints->row(i).col(2) = (int)currentRow;
    }
    img->row(currentRow).colRange(colStart,colEnd) = 255;
  }
}

void MyLine( Mat img, Point start, Point end )
{
  int thickness = 2;
  int lineType = 1;
  line( img,
       start,
       end,
       Scalar( 0, 255, 0 ),
       thickness,
       lineType );
}
