CC= g++
CFLAGS = -o

SOURCES = main.cpp
EXECUTABLE = main.exe

SVM_SOURCES = SVM.cpp
SVM_EXECUTABLE = SVM.exe

LIBPATH_s = -L/usr/local/Cellar/opencv/2.4.11_1/lib
LIBPATH_b = -L/usr/include/opencv2

LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_objdetect -lopencv_ml
FRAMEWORK = -framework Accelerate

INCLUDEPATH_s = -I/usr/local/Cellar/opencv/2.4.11_1/include
INCLUDEPATH_b = -I/usr/include/opencv2
INCLUDEPATH_s += -I/usr/local/Cellar/eigen/3.2.6/include/eigen3
INCLUDEPATH_b += -I/usr/include/eigen3


	#$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)

all: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH_b) $(LIB) $(INCLUDEPATH_b)
	./$(EXECUTABLE)

