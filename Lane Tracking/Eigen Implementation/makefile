CC= g++
CFLAGS = -o

SOURCES = main.cpp
EXECUTABLE = main

LIBPATH = -L/usr/local/Cellar/opencv/2.4.11_1/lib

LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc
FRAMEWORK = -framework Accelerate

INCLUDEPATH = -I/usr/local/Cellar/opencv/2.4.11_1/include
INCLUDEPATH += -I/usr/local/Cellar/eigen/3.2.6/include/eigen3


all: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)
	./$(EXECUTABLE)

