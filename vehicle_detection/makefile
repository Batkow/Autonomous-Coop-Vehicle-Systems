CC= g++
CFLAGS = -o

SOURCES = main.cpp
EXECUTABLE = main.exe

#LIBPATH = -L/usr/local/Cellar/opencv/2.4.11_1/lib
LIBPATH = -L/usr/include/opencv2

LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc
FRAMEWORK = -framework Accelerate

#INCLUDEPATH = -I/usr/local/Cellar/opencv/2.4.11_1/include
INCLUDEPATH = -I/usr/include/opencv2
#INCLUDEPATH += -I/usr/local/Cellar/eigen/3.2.6/include/eigen3
INCLUDEPATH += -I/usr/include/eigen3

ARGUMENTS = "/home/plankton/Videos/highway.avi"

	#$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)

all: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH)
	./$(EXECUTABLE) $(ARGUMENTS)

