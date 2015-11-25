//
//  main.cpp
//  LaneD
//
//  Created by Amrit  Krishnan on 24/11/15.
//  Copyright © 2015 Amrit  Krishnan. All rights reserved.
//

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <ctime>
#include <Eigen/Dense>
#include <math.h>
#include "utils.h"

using namespace std;
using namespace cv;


int T1,T2;
enum{
SCAN_STEP = 5,			  // in pixels
HOUGH_TRESHOLD = 50,		// line approval vote threshold
HOUGH_MIN_LINE_LENGTH = 50,	// remove lines shorter than this treshold
BW_TRESHOLD = 250,
HOUGH_MAX_LINE_GAP = 100,   // join lines to one with smaller than this gaps
LINE_REJECT_DEGREES = 10, // in degrees
BORDERX = 10			  // px, skip this much from left & right borders
};

#define K_VARY_FACTOR 0.2f
#define B_VARY_FACTOR 20
#define MAX_LOST_FRAMES 30


struct Lane {
    Lane(){}
    Lane(CvPoint a, CvPoint b, float angle, float kl, float bl): p0(a),p1(b),angle(angle),
    votes(0),visited(false),found(false),k(kl),b(bl) { }
    
    CvPoint p0, p1;
    int votes;
    bool visited, found;
    float angle, k, b;
};

struct Status {
    Status():reset(true),lost(0){}
    ExpMovingAverage k, b;
    bool reset;
    int lost;
};

Status laneR, laneL;

void FindResponses(IplImage *img, int startX, int endX, int y, std::vector<int>& list)
{
    // scans for single response: /^\_
    
    const int row = y * img->width * img->nChannels;
    unsigned char* ptr = (unsigned char*)img->imageData;
    
    int step = (endX < startX) ? -1: 1;
    int range = (endX > startX) ? endX-startX+1 : startX-endX+1;
    
    for(int x = startX; range>0; x += step, range--)
    {
        if(ptr[row + x] <= BW_TRESHOLD) continue; // skip black: loop until white pixels show up
        
        // first response found
        int idx = x + step;
        
        // skip same response(white) pixels
        while(range > 0 && ptr[row+idx] > BW_TRESHOLD){
            idx += step;
            range--;
        }
        
        // reached black again
        if(ptr[row+idx] <= BW_TRESHOLD) {
            list.push_back(x);
        }
        
        x = idx; // begin from new pos
    }
}

void processSide(std::vector<Lane> lanes, IplImage *edges, bool right) {
    
    Status* side = right ? &laneR : &laneL;
    
    // response search
    int w = edges->width;
    int h = edges->height;
    const int BEGINY = 0;
    const int ENDY = h-1;
    const int ENDX = right ? (w-BORDERX) : BORDERX;
    int midx = w/2;
    int midy = edges->height/2;
    
    // show responses
    int* votes = new int[lanes.size()];
    for(int i=0; i<lanes.size(); i++) votes[i++] = 0;
    
    for(int y=ENDY; y>=BEGINY; y-=SCAN_STEP) {
        std::vector<int> rsp;
        FindResponses(edges, midx, ENDX, y, rsp);
        
        if (rsp.size() > 0) {
            int response_x = rsp[0]; // use first reponse (closest to screen center)
            
            float dmin = 9999999;
            float xmin = 9999999;
            int match = -1;
            for (int j=0; j<lanes.size(); j++) {
                // compute response point distance to current line
                float d = dist2line(
                                    cvPoint2D32f(lanes[j].p0.x, lanes[j].p0.y),
                                    cvPoint2D32f(lanes[j].p1.x, lanes[j].p1.y),
                                    cvPoint2D32f(response_x, y));
                
                // point on line at current y line
                int xline = (y - lanes[j].b) / lanes[j].k;
                int dist_mid = abs(midx - xline); // distance to midpoint
                
                // pick the best closest match to line & to screen center
                if (match == -1 || (d <= dmin && dist_mid < xmin)) {
                    dmin = d;
                    match = j;
                    xmin = dist_mid;
                    break;
                }
            }
            
            // vote for each line
            if (match != -1) {
                votes[match] += 1;
            }
        }
    }
    
    int bestMatch = -1;
    int mini = 9999999;
    for (int i=0; i<lanes.size(); i++) {
        int xline = (midy - lanes[i].b) / lanes[i].k;
        int dist = abs(midx - xline); // distance to midpoint
        
        if (bestMatch == -1 || (votes[i] > votes[bestMatch] && dist < mini)) {
            bestMatch = i;
            mini = dist;
        }
    }
    
    if (bestMatch != -1) {
        Lane* best = &lanes[bestMatch];
        float k_diff = fabs(best->k - side->k.get());
        float b_diff = fabs(best->b - side->b.get());
        
        bool update_ok = (k_diff <= K_VARY_FACTOR && b_diff <= B_VARY_FACTOR) || side->reset;
        
        printf("side: %s, k vary: %.4f, b vary: %.4f, lost: %s\n",
               (right?"RIGHT":"LEFT"), k_diff, b_diff, (update_ok?"no":"yes"));
        
        if (update_ok) {
            // update is in valid bounds
            side->k.add(best->k);
            side->b.add(best->b);
            side->reset = false;
            side->lost = 0;
        } else {
            // can't update, lanes flicker periodically, start counter for partial reset!
            side->lost++;
            if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
                side->reset = true;
            }
        }
        
    } else {
        printf("no lanes detected - lane tracking lost! counter increased\n");
        side->lost++;
        if (side->lost >= MAX_LOST_FRAMES && !side->reset) {
            // do full reset when lost for more than N frames
            side->reset = true;
            side->k.clear();
            side->b.clear();
        }
    }
    
    delete[] votes;
}

void processLanes(Vec4i* lines, IplImage* edges, IplImage* temp_frame) {
    
    // classify lines to left/right side
    std::vector<Lane> left, right;
    
    for(int i = 0; i < lines->size(); i++ )
    {
        CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
        int dx = line[1].x - line[0].x;
        int dy = line[1].y - line[0].y;
        float angle = atan2f(dy, dx) * 180/CV_PI;
        
        if (fabs(angle) <= LINE_REJECT_DEGREES) { // reject near horizontal lines
            continue;
        }
        
        // assume that vanishing point is close to the image horizontal center
        // calculate line parameters: y = kx + b;
        dx = (dx == 0) ? 1 : dx; // prevent DIV/0!
        float k = dy/(float)dx;
        float b = line[0].y - k*line[0].x;
        
        // assign lane's side based by its midpoint position
        int midx = (line[0].x + line[1].x) / 2;
        if (midx < temp_frame->width/2) {
            left.push_back(Lane(line[0], line[1], angle, k, b));
        } else if (midx > temp_frame->width/2) {
            right.push_back(Lane(line[0], line[1], angle, k, b));
        }
    }
    
    // show Hough lines
    for	(int i=0; i<right.size(); i++) {
        cvLine(temp_frame, right[i].p0, right[i].p1, CV_RGB(0, 0, 255), 2);
    }
    
    for	(int i=0; i<left.size(); i++) {
        cvLine(temp_frame, left[i].p0, left[i].p1, CV_RGB(255, 0, 0), 2);
    }
    
    processSide(left, edges, false);
    processSide(right, edges, true);
    
    // show computed lanes
    int x = temp_frame->width * 0.55f;
    int x2 = temp_frame->width;
    cvLine(temp_frame, cvPoint(x, laneR.k.get()*x + laneR.b.get()),
           cvPoint(x2, laneR.k.get() * x2 + laneR.b.get()), CV_RGB(255, 0, 255), 2);
    
    x = temp_frame->width * 0;
    x2 = temp_frame->width * 0.45f;
    cvLine(temp_frame, cvPoint(x, laneL.k.get()*x + laneL.b.get()), 
           cvPoint(x2, laneL.k.get() * x2 + laneL.b.get()), CV_RGB(255, 0, 255), 2);
}



int main(int argc, const char * argv[]) {
    // Setting selection for two different videos
    CvCapture* capture;
    T1 = 200;
    T2 = 300;
    
        //Highway
        capture = cvCreateFileCapture("/Users/amritk/Desktop/highway.avi");
    Mat src,edges,cdst;
    while(1) {
        //clock_t begin = clock();
        // Get frame
        src = cvQueryFrame(capture);
        
        resize(src, src, Size(640,480),0,0,INTER_CUBIC);
        
        // Process frame
        cvtColor(src, cdst, CV_GRAY2BGR);
        GaussianBlur(cdst, cdst, Size(5,5), 1);
        Canny(cdst, edges, T1, T2);
        //Hough lines
        double rho = 1;
        double theta = CV_PI/180;
        vector<Vec4i> lines;
        HoughLinesP(edges, lines, 1, CV_PI/180, 150, 100, 400 );
        processLanes(lines, edges, src);
        /*
#if 0
        vector<Vec2f> lines;
        HoughLines(image, lines, 1, CV_PI/180, 150, 0, 0 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
            float rho = lines[i][0], theta = lines[i][1];
            Point pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            line( src, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
        }
#else
        vector<Vec4i> lines;
        HoughLinesP(image, lines, 1, CV_PI/180, 150, 100, 400 );
        for( size_t i = 0; i < lines.size(); i++ )
        {
            Vec4i l = lines[i];
            line(src, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
            cout<<l<<endl;
        }
#endif
         
         */
        // Show image
        imshow("Canny",edges);
        moveWindow("Canny", 640, 0);
        imshow("detected lines", src);
        moveWindow("detected lines",0,0);
        
        
        // Key press events
        char key = (char)waitKey(1); //time interval for reading key input;
        if(key == 'q' || key == 'Q' || key == 27)
            break;
        //cout<<(float)(clock()-begin) / CLOCKS_PER_SEC<<endl;
    }
    cvReleaseCapture(&capture);
    cvDestroyWindow("Example3");
    return 0;
}




