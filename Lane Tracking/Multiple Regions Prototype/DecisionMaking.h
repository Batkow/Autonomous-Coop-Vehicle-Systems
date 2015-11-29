#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

void SelectClosestLines(Eigen::MatrixXd *pointsPerRegion,Eigen::MatrixXd *regionIndex)
{
  int midRegion = (pointsPerRegion->rows()) / 2;
  
  //Left lane
  bool leftLaneSet = 0;
  for (int i=midRegion-1; i>0; i--) {
    if ((pointsPerRegion->col(0)(i) > pointsPerRegion->col(0)(i-1))) {
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
    if ( (pointsPerRegion->col(0)(i) > pointsPerRegion->col(0)(i+1))){
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

void SelectLanes(Eigen::VectorXd &numberOfPoints,Eigen::VectorXd &laneLocation, int maxTracks)
{
  int nIsland = 0;
  for (int i=0; i<numberOfPoints.rows();i++) {
    Eigen::MatrixXd indexHolder(2,1);
    int offset;
    if (numberOfPoints.row(i)(0) !=0){
      indexHolder(0,0) = i;
      
      if ( i == numberOfPoints.rows()-1) {
        indexHolder(1,0) = i;
        offset = i;
        break;
      } else if( numberOfPoints.row(i+1)(0) !=0){
        indexHolder(1,0) = i+1;
        offset = i;
        i = i+1;
      } else {
        indexHolder(1,0) = i;
        offset = i;
        i = i+1;
      }
    
      int pos;
      numberOfPoints.segment(indexHolder(0,0), indexHolder(1,0)-indexHolder(0,0)+1).maxCoeff(&pos);
      laneLocation.row(nIsland)(0) = pos+offset;
      nIsland++;
      if (nIsland == maxTracks) {
        break;
      }
      
    }
  }
  
  if (nIsland < maxTracks)
  {
    for (int i=laneLocation.rows()-1;i>nIsland-1;i--)
    {
      removeRow(laneLocation, i);
      
    }
  }

}




void SelectLaneOrientation(Eigen::MatrixXd &regionIndex,Eigen::VectorXd &laneLocation,int nRegions)
{
  int nTracks = laneLocation.rows();
  int nLeftTracks = 0;
  int nRightTracks = 0;
  for ( int i = 0;i<nTracks;i++)
  {
    if (laneLocation.row(i)(0)<=nRegions/2-1)
      nLeftTracks++;
  }
  nRightTracks = nTracks - nLeftTracks;
  regionIndex.row(0)(0) = laneLocation.row(nLeftTracks-1)(0);
  regionIndex.row(1)(0) = laneLocation.row(nTracks-nRightTracks)(0);
  /*
  if (laneLocation.row(0)(0) <= 3 && laneLocation.row(1)(0) <= 3){
    regionIndex.row(0)(0) = laneLocation.row(1)(0);
    regionIndex.row(1)(0) = laneLocation.row(2)(0);
  }
  else{
    regionIndex.row(0)(0) = laneLocation.row(0)(0);
    regionIndex.row(1)(0) = laneLocation.row(1)(0);
  }*/
}




