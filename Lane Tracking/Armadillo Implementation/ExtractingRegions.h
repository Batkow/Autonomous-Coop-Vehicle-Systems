#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <armadillo>

using namespace cv;

void GetRegionIntervals(arma::mat *regionPoints,arma::mat *lines,int currentRow, int nCols);
void GetRegionLines(arma::mat regionPoints, arma::mat *lines,int rowV, int colV);



void GetRegionIntervals(arma::mat *regionPoints,arma::mat *lines,int currentRow, int nCols)
{
  regionPoints->at(0,0) = 0;
  regionPoints->at(0,regionPoints->n_cols-1)= nCols;
  
  for (int i=0;i<lines->n_rows;i++)
  {
    regionPoints->at(0,i+1) = min(max((int)(lines->at(i,0)*currentRow+lines->at(i,1)),0),nCols);
    
  }
  
}

void GetRegionLines(arma::mat regionPoints, arma::mat *lines,int rowV, int colV)
{
  // Col = K * row + m
  long width = regionPoints.n_cols;
  for (int i=0; i<regionPoints.n_cols-1; i++) {
    
    double k = (double)(colV-regionPoints(0,i)) / (double)(rowV - regionPoints(0,width-1));
    double m = (double)(colV - k*rowV);
    lines->at(i,0) = k;
    lines->at(i,1) = m;
    
  }
  
}
