#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/ml/ml.hpp>
#include <ctime>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

Settings s;
const string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
if (!fs.isOpened())
{
      cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
      return -1;
}
fs["Settings"] >> s;
fs.release();                                         // close Settings file

if (!s.goodInput)
{
      cout << "Invalid input detected. Application stopping. " << endl;
      return -1;
}

