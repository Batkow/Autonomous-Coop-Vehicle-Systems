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

#ARGUMENTS_b = "/home/plankton/Videos/data_1612.avi"
ARGUMENTS_b = "/home/plankton/Videos/data_1608.avi"
#ARGUMENTS_b = "/home/plankton/Videos/rostock.avi"
#ARGUMENTS_b = "/home/plankton/Videos/highway.avi"
ARGUMENTS_s = "/Users/tempuser/Documents/highway.avi"

	#$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH) $(LIB) $(INCLUDEPATH) $(FRAMEWORK)

bjorn: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH_b) $(LIB) $(INCLUDEPATH_b)
	./$(EXECUTABLE) $(ARGUMENTS_b)

selvin: 
	$(CC) $(CFLAGS) $(EXECUTABLE) $(SOURCES) $(LIBPATH_s) $(LIB) $(INCLUDEPATH_s)
	./$(EXECUTABLE) $(ARGUMENTS_s)

svm_example: 
	$(CC) $(CFLAGS) $(SVM_EXECUTABLE) $(SVM_SOURCES) $(LIBPATH_b) -lopencv_core -lopencv_highgui -lopencv_ml $(INCLUDEPATH_b)
	./$(SVM_EXECUTABLE) "trained_svm.xml"

svm_train: 
	$(CC) $(CFLAGS) TrainSVM.exe TrainSVM.cpp $(LIBPATH_b) $(LIB) $(INCLUDEPATH_b)
	./TrainSVM.exe "training_data_folder" "trained_svm.xml"
