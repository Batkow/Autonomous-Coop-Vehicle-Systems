CC= g++
CFLAGS = -o

SOURCES = main.cpp
EXECUTABLE = revereDemo

LIBPATH = -L/usr/local/lib
LIBPATH += -L/usr/local/Cellar/armadill/5.200.1/lib
LIB = -lopencv_core -lopencv_highgui -lopencv_imgproc

INCLUDEPATH = -I/usr/local/include
INCLUDEPATH += -I/usr/local/Cellar/eigen/3.2.6/include/eigen3
INCLUDEPATH += -I/usr/local/Cellar/armadillo/5.200.1/include

all: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH)
	./$(EXECUTABLE)

