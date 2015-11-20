#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <armadillo>

using namespace std;
using namespace cv;

// Extract points for the different regions
void ScanImage(cv::Mat *img, arma::mat *lines,arma::mat *recoveredPoints,int nPoints,int minRow, int maxRow);

// Remove unwanted points
void RemoveInvalidPoints(arma::mat *recoveredPoints,arma::field<arma::mat> *cell,long nRegions,int nPoints);

// Solve RMS problem for points to a line
void LinearSolve(arma::field<arma::mat> *cell, arma::mat *k, arma::mat *m,long nRegions);

void ScanImage(cv::Mat *img, arma::mat *lines,arma::mat *recoveredPoints,int nPoints,int minRow, int maxRow)
{
  double deltaRow = (maxRow-minRow) / (double)nPoints;
  int currentRow;
  
  arma::mat regionPoints(1,lines->n_rows+2);
  for (int i=0; i<nPoints; i++) {
    currentRow = minRow + i*deltaRow;
    
    // Calculate the start and end points for the regions
    GetRegionIntervals(&regionPoints,lines,currentRow,img->cols);
    Mat slice = img->row(currentRow);
    double meanPoint;
    for (int j=0; j<lines->n_rows+1; j++) {
      Mat indices,regionSlice;
      
      if (regionPoints(0,j) != regionPoints(0,j+1))
      {
        regionSlice = slice.colRange(regionPoints.at(0,j),regionPoints.at(0,j+1));
        findNonZero(regionSlice,indices);
        
        if (abs(mean(indices)[0]) < 0.1){
          meanPoint= 0;
        } else {
          meanPoint= mean(indices)[0]+regionPoints(0,j);
        }
        
        recoveredPoints->at(i,j) = (int)meanPoint;
        
      } else
      {
        recoveredPoints->at(i,j) = 0;
      }
      
      
    }
    recoveredPoints->at(i,lines->n_rows+1) = currentRow;
  }
}

void RemoveInvalidPoints(arma::mat *recoveredPoints,arma::field<arma::mat> *cell,long nRegions,int nPoints)
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

void LinearSolve(arma::field<arma::mat> *cell, arma::mat *k, arma::mat *m,long nRegions)
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
