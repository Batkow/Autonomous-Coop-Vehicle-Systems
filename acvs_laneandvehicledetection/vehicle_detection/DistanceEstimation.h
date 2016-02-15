
#pragma once
#include <iostream>
#include <math.h>

using namespace std;

void initializeDistanceEstimation();

float estimateDistance(float camXCoord, float camYCoord);



void initializeDistanceEstimation() {

}


float estimateDistance(float camXCoord, float camYCoord) {
	float h = 1.7410826179;
	float c = 0.00144671;//1.0/691.2256;
	float r = 1.24896677;//3.141593/2.5154;
	float output = h*tan(c*camYCoord + r);
	//cout << "Distance is: " << output << "\n";
	return output;
}