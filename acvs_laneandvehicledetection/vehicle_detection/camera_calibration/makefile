CC= g++
CFLAGS = -o

SOURCES = camera_calibration.cpp
EXECUTABLE = camera_calibration.exe


LIBPATH = -L/usr/local/Cellar/opencv/2.4.12/lib

LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lopencv_ml
FRAMEWORK = -framework Accelerate

INCLUDEPATH = -I/usr/local/Cellar/opencv/2.4.12/include
INCLUDEPATH += -I/usr/local/Cellar/eigen/3.2.6/include/eigen3

ARGUMENTS = "/Users/tempuser/Documents/camera_calibration/in_VID6.xml“

	#$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)

make: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH)
	./$(EXECUTABLE) $(ARGUMENTS)

