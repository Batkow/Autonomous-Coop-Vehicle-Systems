#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;


int main(int argc, const char * argv[]) {

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

	Mat cameraMatrix2, distCoeffs2;
	fs["Camera_Matrix"] >> cameraMatrix2;
	fs["Distortion_Coefficients"] >> distCoeffs2;

	cout << "frameCount: " << frameCount << endl
    	 << "calibration date: " << date << endl
     	 << "camera matrix: " << cameraMatrix2 << endl
     	 << "distortion coeffs: " << distCoeffs2 << endl;

	fs.release();
	
	//camMatrix = cm.cam_mat_data;
	//cout << camMatrix;
	
	return 0;
}