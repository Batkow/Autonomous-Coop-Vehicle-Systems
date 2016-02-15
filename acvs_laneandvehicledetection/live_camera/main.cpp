#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <sstream>
#include <time.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


double mmToPixel (vector<Point2f> edgeLocation, int squareSize)
{
    //vector<Point2f> edgeLocation;
    // Begin calculating mm/pixel estimate
    float x1 = edgeLocation.at(1).x;
    cout << x1 << endl;
    float y1 = edgeLocation.at(1).y;
    cout << y1 << endl;
    float x2 = edgeLocation.at(2).x;
    cout << x2 << endl;
    float y2 = edgeLocation.at(2).y;
    cout << y2 << endl;
    
    float  point1[2] = {x1, y1};
    float  point2[2] = {x2, y2};
    
    double a = x2-x1;
    double b = y2-y1;
    double temp1 = pow(a,2);
    double temp2 = pow(b,2);
    double temp = temp1+temp2;
    cout << temp << endl;
    double distPixels = sqrt(temp);
    cout << distPixels << endl;
    double MM = squareSize/distPixels;
    // Ending mm/pixel estimate
    return MM;
}


int main(int argc, const char * argv[]) {
    
    // Start of reading the camera calibration parameters
    const char * camMatrixXMLFile = "default.xml";
    
    if (argv[1]) {
        camMatrixXMLFile =  argv[1];
        cout << "camMatrixXMLFile: " << argv[1] <<  "\n";
    } else {
        cout << "No camMatrixXMLFile specified! \n";
    }
    
    FileStorage fs;
    fs.open(camMatrixXMLFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        cout << "Could not open XML file : \""<< camMatrixXMLFile << "\"" << endl;
        return -1;
    }
    
    // first method: use (type) operator on FileNode.
    int frameCount = (int)fs["nrOfFrames"];
    
    std::string date;
    // second method: use FileNode::operator >>
    fs["calibration_Time"] >> date;
    int squareSize;
    fs["square_Size"] >> squareSize;
    Mat cameraMatrix2, distCoeffs2;
    fs["Camera_Matrix"] >> cameraMatrix2;
    fs["Distortion_Coefficients"] >> distCoeffs2;
    /*
    cout << "frameCount: " << frameCount << endl
    << "calibration date: " << date << endl
    << "camera matrix: " << cameraMatrix2 << endl
    << "distortion coeffs: " << distCoeffs2 << endl;
    */
    fs.release();
    
    //End of reading the camera calibration parameters
    
    VideoCapture vcap(0); // open the default camera
    if(!vcap.isOpened())  // check if we succeeded
        return -1;
    
    //VideoCapture vcap;
    Mat outImage;
    Mat inputImage;
    //vector<Point2f> edgeLocation;

    //const std::string videoStreamAddress = "http://root:pass@192.168.0.90/axis-cgi/mjpg/video.cgi?user=USERNAME&password=PWD&channel=0&.mjpg";
    
    /*if(!vcap.open(videoStreamAddress)) {
        cout << "Error opening video stream or file" << endl;
        return -1;
    }
     */
    for(;;) {
        
        Size patternSize(7,7);
        vector<Point2f> corners;
        int frameSkip = 5;
        for (int i = 0; i < frameSkip; i=i+1) {
            vcap.read(inputImage);
        }
        imshow("Input Window", inputImage);
        undistort(inputImage, outImage, cameraMatrix2, distCoeffs2, cameraMatrix2);
        
        bool patternFound = findChessboardCorners(outImage,patternSize ,corners,CALIB_CB_ADAPTIVE_THRESH+CALIB_CB_NORMALIZE_IMAGE+CALIB_CB_FAST_CHECK);
        
        drawChessboardCorners(outImage, patternSize, Mat(corners), patternFound);
        //edgeLocation = corners;
        //cout << edgeLocation << endl;
        /*if(!vcap.read(image)) {
            cout << "No frame" << endl;
            waitKey();
        }*/
        imshow("Output Window", outImage);

        if(waitKey(1) >= 0)
        {
            double distinMM = mmToPixel(corners, squareSize);
            cout << distinMM << "\t mm/pixel" << endl;
            break;
        }
    }
    
    vcap.release();
    
    
    return 0;
    

}
