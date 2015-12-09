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

	
class Camera_Matrix
{
	public:
	    void write(FileStorage& fs) const                        //Write serialization for this class
	     {
	         fs << "{" << "rows"  << rows
	                   << "cols" << cols
					   << "data" << data
	            << "}";
	     }
	     void read(const FileNode& node)                          //Read serialization for this class
	     {
	         node["rows" ] >> rows;
	         node["cols"] >> cols;
	         node["data"] >> data;
	     }
	public:
		int rows;
		int cols;
		Mat data;
};

static void write(FileStorage& fs, const std::string&, const Camera_Matrix& x)
{
    x.write(fs);
}

static void read(const FileNode& node, Camera_Matrix& x, const Camera_Matrix& default_value = Camera_Matrix())
{
	if(node.empty())
		x = default_value;
	else
		x.read(node);
}

const char * camMatrixXMLFile = "/Users/tempuser/Documents/camera_calibration/out_camera_data.xml";

int main(int argc, const char * argv[]) {
	Camera_Matrix cm;

    const char * camMatrixXMLFile = "/Users/tempuser/Documents/camera_calibration/out_camera_data.xml";
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
	//Mat camMatrix;
	FileNode camMatrix = fs["data"];
	//fs["Camera_Matrix"] >> cm;
	fs.release();
	
	//camMatrix = cm.cam_mat_data;
	cout << "Level1 \n";
	//cout << camMatrix;
	
	return 0;
}