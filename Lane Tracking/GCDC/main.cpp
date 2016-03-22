//
//  main.cpp
//  DAT295
//
//  Created by Ivo Batkovic on 2015-11-19.
//  Copyright © 2015 Ivo Batkovic. All rights reserved.
//
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include "Drawing.h"
#include "ExtractingRegions.h"
#include "ProcessImage.h"
#include "DecisionMaking.h"
#include <Eigen/Dense>
#include <armadillo>
#include <time.h>
using namespace std;
using namespace cv;


Eigen::MatrixXd testPos(9,4);
int nPoint = 0;
int firstPoint = 1, secondPoint = 0;
void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
  if  ( event == EVENT_LBUTTONDOWN )
  {
    cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << endl;
    
    if (nPoint>=9) {
      //do nothing
    }
    else {
      
    
    if (firstPoint == 1)
    {
      testPos(nPoint,0) = x;
      testPos(nPoint,1) = y;
      firstPoint = 0;
      secondPoint = 1;
    } else if ( secondPoint == 1)
    {
      testPos(nPoint,2) = x;
      testPos(nPoint,3) = y;
      secondPoint = 0;
      firstPoint = 1;
      nPoint++;
    }
    }
  }
}

int main(int argc, const char * argv[]) {

  //-----------------------------
  // Definitions for the video choice
  //-----------------------------
  Eigen::MatrixXd regions;
  CvCapture* capture;
  VideoCapture vcap;
  int canny, ROWV,COLV,MINROW,MAXROW,T1,T2;
  int width = 640, height = 480;
  
  
  
  //-----------------------------
  // Choice of video source
  //-----------------------------
  if (0)
  {
    //-----------------------------
    //Rural.avi video
    //-----------------------------
    ROWV = 100; COLV = 320;
    MINROW = 150; MAXROW = 300;
    T1 = 200; T2 = 300; canny = 1;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rural.avi");
    regions = *new Eigen::MatrixXd(7,4);
    //region = col1,row1, col2,row2
    regions <<  178, 148, 7,  207,
                214, 146, 3,  248,
                286, 150, 189,301,
                320, 148, 320,298,
                345, 150, 453,300,
                427, 147, 622,241,
                457, 147, 630,213;
    canny = 1;
    
  } else if(0)
  {
    //-----------------------------
    //Highway.avi video
    //-----------------------------
    ROWV = 293; COLV = 316;
    MINROW = 350; MAXROW = 480;
    T1 = 100; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/highway.avi");
    regions = *new Eigen::MatrixXd(7,4);
    //region = col1,row1, col2,row2
    regions <<  -60,  450,  316,  293,
                60,   450,  316,  293,
                220,  450,  316,  293,
                316,  450,  316,  293,
                420,  450,  316,  293,
                580,  450,  316,  293,
                700,  450,  316,  293;
    canny = 0;
  
  } else if(0)
  {
    //-----------------------------
    // rostock.avi video
    //-----------------------------
    ROWV = 230; COLV = 400;
    MINROW = 280; MAXROW = 450;
    T1 = 80; T2 = 150;
    capture = cvCreateFileCapture("/Users/batko/Downloads/rostock.avi");
    regions = *new Eigen::MatrixXd(9,4);
    //region = col1,row1, col2,row2
    regions <<  211,  274,  4,    316,
                309,  279,  8,    425,
                335,  278,  44,   446,
                375,  281,  249,  449,
                398,  280,  337,  447,
                419,  281,  463,  452,
                465,  280,  613,  388,
                489,  278,  634,  339,
                579,  279,  636,  297;
    canny = 1;

  } else {
    MINROW = 300; MAXROW = 442;
    capture = cvCreateFileCapture("/Users/batko/Downloads/croppedVid3.avi");
    regions = *new Eigen::MatrixXd(7,4);
    regions <<      70, 299,   0, 309,
    188, 297,  28, 374,
    275, 299, 158, 443,
    318, 299, 319, 451,
    341, 300, 425, 444,
    431, 302, 567, 398,
    545, 290, 627, 310;
    canny = 1;
    
  }
  
  //-----------------------------
  // Live cam
  //-----------------------------
  bool liveCam = 0;
  if (liveCam){
    MINROW = 300; MAXROW = 442;
    regions = *new Eigen::MatrixXd(7,4);
    regions <<      70, 299,   0, 309,
    188, 297,  28, 374,
    275, 299, 158, 443,
    318, 299, 319, 451,
    341, 300, 425, 444,
    431, 302, 567, 398,
    545, 290, 627, 310;

    const std::string videoStreamAddress = "http://root:pass@192.168.0.90/axis-cgi/mjpg/video.cgi?user=USERNAME&password=PWD&channel=0&.mjpg";
    if(!vcap.open(videoStreamAddress)) {
     cout << "Error opening video stream or file" << endl;
     return -1;
     }
  }

  
  //-----------------------------
  // Scaling: Calibrations were made for 640x480 resolution
  //-----------------------------
  MINROW = MINROW * ( height / 480.0); MAXROW = MAXROW * (height / 480.0);
  regions.col(0) = regions.col(0) * ( width / 640.0);
  regions.col(2) = regions.col(2) * ( width / 640.0);
  regions.col(1) = regions.col(1) * ( height / 480.0);
  regions.col(3) = regions.col(3) * ( height / 480.0);
  
  //-----------------------------
  // Video properties
  //-----------------------------
  int nFrames = (int) cvGetCaptureProperty(capture, CV_CAP_PROP_FRAME_COUNT);
  
  //-----------------------------
  // Image processing parameters
  //-----------------------------
  int nPoints = 40000, nMaxPoints = MAXROW-MINROW;
  if (nPoints > nMaxPoints)
    nPoints = nMaxPoints;
  
  //-----------------------------
  // Initializations
  //-----------------------------
  
  // Matrix holding the lines col = row * k + m for the region lines
  Eigen::MatrixXd lines(regions.rows(),2);
  
  // Matrix holding the mean column for each region on each row
  Eigen::MatrixXd recoveredPoints(nPoints,lines.rows()+2);
  
  // Number of search regions
  long nRegions = recoveredPoints.cols()-1;
  
  // Holds the K and M parameters for each region
  Eigen::MatrixXd K(nRegions,1), M(nRegions,1);
  
  // Counts the number of points per each region
  Eigen::VectorXd pointsPerRegion(nRegions,1);
  
  // Holds the index that decides the left and right road track
  Eigen::MatrixXd regionIndex(2,1);
  
  // Holds the location of found lanes
  Eigen::VectorXd laneLocation2;
  
  // Holds the k and m values from the previous frame
  Eigen::MatrixXd kPrev = Eigen::MatrixXd::Zero(regions.rows()+1, 1);
  Eigen::MatrixXd mPrev = Eigen::MatrixXd::Zero(regions.rows()+1, 1);
  
  
  // OpenCV object matrices for loaded image
  Mat image,src,IMG;
  
  // Momentum parameterand lane offset parameter
  double alpha = 0.5;
  
  // Lane offset variable
  double laneOffset;
  
  // Frame counter
  int iFrame = 0;
  
  // Holds the index of the left and right road track
  int p1,p2;
  
  // Middle region line index
  int midRegion = (int)(lines.rows()-1)/2;
  
  // Armadillo matrix to store lane offset for analysis
  //arma::mat recordedArray = arma::zeros(nFrames,3);
  
  // Timing and lab sesssion
  time_t rawtime;
  ofstream myfile;
  myfile.open ("/Users/batko/Desktop/dataLog.txt");
  //string videoOutputPath = "/Users/batko/Desktop/testVid3.avi";
  //VideoWriter vidWriter(videoOutputPath,CV_FOURCC('M','J','P','G'),10, Size(640,480),true);
  
  //-----------------------------
  // Extracts the equation for the region lines
  //-----------------------------
  GetRegionLinesV2(regions, lines);
  
  // Holds the k and m values for the region lines
  Eigen::MatrixXd k = lines.col(0); 
  Eigen::MatrixXd m = lines.col(1);

  
  //-----------------------------
  // Initialize windows
  //-----------------------------
  namedWindow("1", 1);
  //namedWindow("Canny",1);
  
  
  moveWindow("1", 0, 0);
  //moveWindow("Canny", 640, 0);
  setMouseCallback("1", CallBackFunc, NULL);
  
  
  
  //-----------------------------
  // Main loop processing frames
  //-----------------------------
  //time (&rawtime);
  //myfile << ctime (&rawtime) << "\n";
  while(1) {
    
    //-----------------------------
    // Get frame and check if valid
    //-----------------------------
    // Live cam
    if (liveCam){
      vcap.read(src);
    } // Video
    else {
      src = cvQueryFrame(capture);
      if (src.empty() || iFrame == nFrames){
        cout<<"End of video file"<<endl; break; }
    }
    
    
    //-----------------------------
    // Re-size the source image
    //-----------------------------
    resize(src, src, Size(width,height),0,0,INTER_CUBIC);
    clock_t begin = clock();
    //-----------------------------
    // Choose processing type
    //-----------------------------
    if (canny)
    {
      GaussianBlur(src, src, Size(9,9), 1);
      Canny(src, image, 150, 160);
    }
    else
    {
      cvtColor(src, IMG, CV_BGR2GRAY);
      threshold(IMG, image, 105, 255, 0);
      //medianBlur(image, image, 3);
    }
    
    
    //-----------------------------
    // Local line search
    //-----------------------------
    ScanImage(&image,&lines,&recoveredPoints,nPoints,MINROW,MAXROW);
    
    
    //-----------------------------
    // Extract lines from points
    //-----------------------------
    ExtractLines(&recoveredPoints,&K,&M,nRegions,nPoints,&pointsPerRegion);

    
    //-----------------------------
    // Make decision based on lines
    //-----------------------------
    laneLocation2 = Eigen::VectorXd::Zero(nRegions, 1);
    SelectLanesV2(pointsPerRegion, laneLocation2);
    if (pointsPerRegion.sum() != 0){
      SelectLaneOrientation(regionIndex,laneLocation2,(int)recoveredPoints.cols());
      p1 = regionIndex(0,0), p2 = regionIndex(1,0);
      
      
      //-----------------------------
      //Add momentum & update previous lines
      //-----------------------------
      AddMomentum(K,kPrev,M,mPrev,alpha,regionIndex);
      kPrev = K;
      mPrev = M;
      
      //-----------------------------
      // Draw boarders/lines
      //-----------------------------
      //vidWriter.write(src);
      DrawBorders(&src,MINROW,MAXROW,K(p1,0),K(p2,0),M(p1,0),M(p2,0));
      //DrawTracks(&src, &K, &M,MINROW,MAXROW,Scalar(0,0,255));
      //DrawTracks(&src, &k,&m,MINROW,MAXROW,Scalar(255,255,255));
      
      
      
      //-----------------------------
      // Calculate lane offset
      //-----------------------------
      laneOffset = GetLateralPosition(K(p1,0),M(p1,0),
                                      K(p2,0),M(p2,0),
                                      lines.col(0)(midRegion),lines.col(1)(midRegion),
                                      (MAXROW));
      //cout<<"Lane offset:"<<laneOffset<<endl;
      float timeTaken = (float)(clock()-begin) / CLOCKS_PER_SEC;
      myfile << timeTaken << "\t" << laneOffset << "\t";
    }
    
    //-----------------------------
    // Show image
    //-----------------------------
    imshow("1", src);
    //imshow("Canny",image);
    
  
    //-----------------------------
    //Key press events
    //-----------------------------
    char key = (char)waitKey(1); //time interval for reading key input;
    if(key == 'q' || key == 'Q' || key == 27)
      break;
    else if (key =='s'){
      waitKey(0);
      cout<<testPos<<endl;
      waitKey(0);
    }
    
    /*
    char key = (char)waitKey(0);
    if (key == 'y')
      myfile << 1<<"\n"<<endl;
    else if (key == 'q')
      break;
    else
      myfile << 0<<"\n"<<endl;
     */
    
    
    
  }
  //int process = recordedArray.save("/Users/batko/Desktop/Master/Y2/LP2/DAT295/DAT295/Data/ruralOffset320.mat",arma::raw_ascii);
  //cout<<process<<endl;
  //cout<<"nFrames"<<" "<<nFrames<<endl;
  cvReleaseCapture(&capture);
  cvDestroyWindow("Example3");
  myfile.close();
  return 0;
  }
