#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <ctime>
#include "Drawing.h"
#include "ExtractingRegions.h"
#include "ProcessImage.h"
#include <Eigen/Dense>

using namespace std;
using namespace cv;

int ROWV, COLV, MINROW, MAXROW, T1, T2;

int main(int argc, char** argv){
    
    const char * videoPath = "defaultPath";
    if (argv[1]) {
        videoPath = argv[1];
    }
    else{
        cout << "\n No input data detected \n";
    }
    
    Eigen::MatrixXd regions(1,4);
    CvCapture* capture;
    
    capture = cvCreateFileCapture(videoPath);
   /*
    Highway
    ROWV = 293; COLV = 316;
    MINROW = 350; MAXROW = 450;
    T1 = 50; T2 = 150;
    capture = cvCreateFileCapture(videoPath);
    regions << -60, COLV,700,MAXROW;
    cout << regions << "\n";
    */
    
    CascadeClassifier haarClassifier = CascadeClassifier("haar_classifiers/cars3.xml");
    
    Mat src;
    vector<Rect> detectedVehicles;
    //cout << detectedVehicles << "\n";
    
    while(1){
        
        src = cvQueryFrame(capture);
        /*
        int frameSkipRate = 5;
        for (int i = 0; i<frameSkipRate ; i++) {
            src = cvQueryFrame(capture);
        }
        */
        resize(src, src, Size(640,480),0,0,INTER_CUBIC);
        clock_t begin = clock();
        
        GaussianBlur(src, src, Size(5,5), 1);
        
       // haarClassifier.detectMultiScale(src, detectedVehicles,1.2,3,[30,30]);
        haarClassifier.detectMultiScale(src, detectedVehicles, 1.2,3,0,30,100);
        
        for (size_t i = 0; i < detectedVehicles.size(); i++)
        {
            rectangle(src, detectedVehicles[i], Scalar(0,255,0));
        }
        
        imshow("Blurry", src);
        moveWindow("Blurry", 0, 0);
        
        
        // Key press events
        char key = (char)waitKey(1); //time interval for reading key input;
        if(key == 'q' || key == 'Q' || key == 27)
            break;
        
    }
    cvReleaseCapture(&capture);
    cvDestroyWindow("Example3");
    return 0;
    
}

