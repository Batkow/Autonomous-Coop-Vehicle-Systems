#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Dense>

using namespace cv;

// Draw track lines
void DrawLine( Mat img, Point start, Point end );

// Draw the borders
void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2);


void DrawLine( Mat img, Point start, Point end )
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

void DrawTracks(cv::Mat *src,Eigen::MatrixXd *k,Eigen::MatrixXd *m,int minRow, int maxRow)
{
  int nRows = src->rows;
  for (int i = 0; i<k->rows();i++)
  {
    if (k->col(0)(i) !=0 || m->col(0)(i) !=0) {
      DrawLine(*src,Point((nRows*k->col(0)(i)+m->col(0)(i)),nRows), Point((minRow*k->col(0)(i)+m->col(0)(i)),minRow));
    }
  }
}


void DrawBorders(Mat *src, bool borderCondition, int minRow, int maxRow, double k1,double k2,double m1,double m2)
{
  if (borderCondition)
  {
    DrawLine(*src,Point((maxRow*k1+m1),maxRow), Point((maxRow*k2+m2),maxRow));
    DrawLine(*src,Point((minRow*k1+m1),minRow), Point((minRow*k2+m2),minRow));
  }
  
}

