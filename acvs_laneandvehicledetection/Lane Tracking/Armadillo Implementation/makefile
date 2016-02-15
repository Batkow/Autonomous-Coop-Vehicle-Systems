CC= g++
CFLAGS = -o

SOURCES = main.cpp
EXECUTABLE = main

LIBPATH = -L/usr/local/Cellar/opencv/2.4.11_1/lib
LIBPATH += -L/usr/local/Cellar/armadillo/5.200.1/lib

LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc 
FRAMEWORK = -framework Accelerate

INCLUDEPATH = -I/usr/local/Cellar/opencv/2.4.11_1/include
INCLUDEPATH += -I/usr/local/Cellar/armadillo/5.200.1/include



all: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)
	./$(EXECUTABLE)

