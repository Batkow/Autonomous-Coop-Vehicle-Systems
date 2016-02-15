#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <armadillo>

using namespace cv;

// Draw track lines
void MyLine( Mat img, Point start, Point end );

// Draw the track
void DrawTracks(cv::Mat *src,arma::mat *k,arma::mat *m,int minRow, int maxRow);

// Draw the borders
void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2);


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

void DrawTracks(cv::Mat *src,arma::mat *k,arma::mat *m,int minRow, int maxRow)
{
  int nRows = src->rows;
  for (int i = 0; i<k->n_rows;i++)
  {
    if (k->at(i,0) !=0 || m->at(i,0) !=0) {
      MyLine(*src,Point((nRows*k->at(i,0)+m->at(i,0)),nRows), Point((minRow*k->at(i,0)+m->at(i,0)),minRow));
    }
  }
}

void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2)
{
  if (borderCondition)
  {
    MyLine(*src,Point((maxRow*k1+m1),maxRow), Point((maxRow*k2+m2),maxRow));
    MyLine(*src,Point((minRow*k1+m1),minRow), Point((minRow*k2+m2),minRow));
  }
  
}

