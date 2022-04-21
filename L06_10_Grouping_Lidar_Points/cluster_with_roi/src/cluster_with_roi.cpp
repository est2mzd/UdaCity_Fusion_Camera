#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"
#include "dataStructures.h"

using namespace std;

void loadCalibrationData(cv::Mat &P_rect_00, cv::Mat &R_rect_00, cv::Mat &RT)
{
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;

}

void showLidarTopview(std::vector<LidarPoint> &lidarPoints, cv::Size worldSize, cv::Size imageSize)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.height / worldSize.height) + imageSize.width / 2;

        float zw = (*it).z; // world position in m with y facing left from sensor
        if(zw > -1.40){       

            float val = it->x;
            float maxVal = worldSize.height;
            int red = min(255, (int)(255 * abs((val - maxVal) / maxVal)));
            int green = min(255, (int)(255 * (1 - abs((val - maxVal) / maxVal))));
            cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, green, red), -1);
        }
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 7);
    cv::resize(topviewImg, topviewImg, cv::Size(), 0.45,0.45);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topViewImage(imageSize, CV_8UC3, cv::Scalar(255,255,255));

    for(auto itBBox = boundingBoxes.begin(); itBBox!=boundingBoxes.end(); ++itBBox)
    {
        // create randomized color for current 3D object
        cv::RNG rng(itBBox->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150),rng.uniform(0,150),rng.uniform(0,150));

        // plot Lidar points into top view image
        int top2D=1e8, left2D=1e8, bottom2D=0.0, right2D=0.0;
        float x3Dmin=1e8, y3Dmin=1e8, y3Dmax=-1e8;

        for(auto itLidar = itBBox->lidarPoints.begin(); itLidar!=itBBox->lidarPoints.end(); ++itLidar)
        {
            // world coordinate
            float x3D = itLidar->x; // world position in m with x facing forward from sensor
            float y3D = itLidar->y; // world position in m with y facing left from sensor
            x3Dmin = (x3Dmin < x3D) ? x3Dmin : x3D;
            y3Dmin = (y3Dmin < y3D) ? y3Dmin : y3D;
            y3Dmax = (y3Dmax > y3D) ? y3Dmax : y3D;

            // top view coordinates
            int y2D = (-x3D * imageSize.height / worldSize.height) + imageSize.height;
            int x2D = (-y3D * imageSize.width  / worldSize.width)  + imageSize.width / 2.0;

            // find enclosing rectangle
            top2D    = (top2D < y2D)  ? top2D  : y2D;
            left2D   = (left2D < x2D) ? left2D : x2D;
            bottom2D = (bottom2D > y2D) ? bottom2D : y2D;
            right2D  = (right2D > x2D) ? right2D : x2D;

            // draw individual point
            cv::circle(topViewImage, cv::Point(x2D, y2D), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topViewImage, cv::Point(left2D, top2D), cv::Point(right2D, bottom2D), cv::Scalar(0,0,0),2);

        // augment object with some kye data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", itBBox->boxID, (int)itBBox->lidarPoints.size());
        putText(topViewImage, str1, cv::Point2f(left2D-250, bottom2D+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, y=%2.2f m", x3Dmin, y3Dmax - y3Dmin);
        putText(topViewImage, str2, cv::Point2f(left2D-250, bottom2D+125), cv::FONT_ITALIC, 2, currColor);
    }

    // plot distance markers
    float lineSpacing = 2.0;
    int numMarkers = floor(worldSize.height / lineSpacing);
    for(size_t i=0; i < numMarkers; ++i)
    {
        int y = (- (i*lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topViewImage, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255,0,0));
    }

    // display image
    string windowName = "3D Objects";
    //cv::namedWindow(windowName, 1);
    cv::namedWindow(windowName, 7);
    cv::resize(topViewImage, topViewImage, cv::Size(), 0.45,0.45);
    cv::imshow(windowName, topViewImage);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}



void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints)
{
    // store calibration data in OpenCV matrices
    cv::Mat P_rect_xx(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_xx(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    loadCalibrationData(P_rect_xx, R_rect_xx, RT);

    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat Coord3D(4, 1, cv::DataType<double>::type);
    cv::Mat Coord2D(3, 1, cv::DataType<double>::type);

    // Loop-1 for Lidar points
    for (auto itLidar = lidarPoints.begin(); itLidar != lidarPoints.end(); ++itLidar)
    {
        // assemble vector for matrix-vector-multiplication
        Coord3D.at<double>(0, 0) = itLidar->x;
        Coord3D.at<double>(1, 0) = itLidar->y;
        Coord3D.at<double>(2, 0) = itLidar->z;
        Coord3D.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Coord2D = P_rect_xx * R_rect_xx * RT * Coord3D;
        cv::Point pointImagePlane;
        pointImagePlane.x = Coord2D.at<double>(0, 0) / Coord2D.at<double>(0, 2); // pixel coordinates
        pointImagePlane.y = Coord2D.at<double>(1, 0) / Coord2D.at<double>(0, 2);

        double shrinkFactor = 0.10;
        // pointers to all bounding boxes which enclose the current Lidar point
        vector<vector<BoundingBox>::iterator> enclosingBoxes;

        // Loop-2 for Bounding Boxes
        for (vector<BoundingBox>::iterator itBBox = boundingBoxes.begin(); itBBox != boundingBoxes.end(); ++itBBox)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x      = (*itBBox).roi.x + shrinkFactor * (*itBBox).roi.width / 2.0;
            smallerBox.y      = (*itBBox).roi.y + shrinkFactor * (*itBBox).roi.height / 2.0;
            smallerBox.width  = (*itBBox).roi.width  * (1 - shrinkFactor);
            smallerBox.height = (*itBBox).roi.height * (1 - shrinkFactor);

            // check wether a point is within current bounding box
            if (smallerBox.contains(pointImagePlane))
            {
                // Each lidar point belong to only one bounding box.
                // So, if lidar point is in the bounding box,
                // the point is registered to the bounding box
                // and erased not to be registerd later.
                itBBox->lidarPoints.push_back(*itLidar);
                //lidarPoints.erase(itLidar);
                //itLidar--;
                break;
            }
        } // eof loop over all bounding boxes
    } // eof loop over all Lidar points
}

int main()
{
    std::vector<LidarPoint> lidarPoints;
    readLidarPts("../dat/C53A3_currLidarPts.dat", lidarPoints);

    std::vector<BoundingBox> boundingBoxes;
    readBoundingBoxes("../dat/C53A3_currBoundingBoxes.dat", boundingBoxes);

    clusterLidarWithROI(boundingBoxes, lidarPoints);

    cv::Size worldSize = cv::Size(10.0, 25.0);
    cv::Size imageSize = cv::Size(2000, 2000);
    bool bWait = true;
    for (auto it = boundingBoxes.begin(); it != boundingBoxes.end(); ++it)
    {
        if (it->lidarPoints.size() > 0)
        {
            showLidarTopview(it->lidarPoints, cv::Size(10.0, 25.0), cv::Size(1000, 2000));
        }
    }

    show3DObjects(boundingBoxes, worldSize, imageSize, bWait);
    return 0;
}