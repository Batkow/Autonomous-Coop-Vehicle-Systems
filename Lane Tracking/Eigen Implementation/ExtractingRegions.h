#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>
using namespace cv;

// Decide the region intervals
void GetRegionIntervals(Eigen::MatrixXd *regionPoints,Eigen::MatrixXd *lines,int currentRow, int nCols);

// Get the lines COL = k * ROW + m
void GetRegionLines(Eigen::MatrixXd *regionPoints, Eigen::MatrixXd *lines,int rowV, int colV);



void GetRegionIntervals(Eigen::MatrixXd *regionPoints,Eigen::MatrixXd *lines,int currentRow, int nCols)
{
  regionPoints->col(0)(0) = 0;
  regionPoints->col(regionPoints->cols()-1)(0)= nCols;
  
  for (int i=0;i<lines->rows();i++)
  {
    regionPoints->col(i+1)(0) = min(max((int)(lines->col(0)(i)*currentRow+lines->col(1)(i)),0),nCols);
    
  }
  
}


void GetRegionLines(Eigen::MatrixXd *regionPoints, Eigen::MatrixXd *lines,int rowV, int colV)
{
  // Col = K * row + m
  long width = regionPoints->cols();
  for (int i=0; i<regionPoints->cols()-1; i++) {
    
    double k = (double)(colV-regionPoints->col(i)(0)) / (double)(rowV - regionPoints->col(width-1)(0));
    double m = (double)(colV - k*rowV);
    lines->col(0)(i) = k;
    lines->col(1)(i) = m;
    
  }
  
}


