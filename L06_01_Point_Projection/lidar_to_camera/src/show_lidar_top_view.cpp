#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "structIO.hpp"

using namespace std;

void showLidarTopview()
{
    std::vector<LidarPoint> lidarPoints;
    readLidarPts("../dat/C51_LidarPts_0000.dat", lidarPoints);

    cv::Size worldSize(10.0, 20.0); // width and height of sensor field in m
    cv::Size imageSize(1000, 2000); // corresponding top view image in pixel

    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(0, 0, 0));

    // plot Lidar points into image
    for (auto it = lidarPoints.begin(); it != lidarPoints.end(); ++it)
    {
        float xw = (*it).x; // world position in m with x facing forward from sensor
        float yw = (*it).y; // world position in m with y facing left from sensor

        int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
        int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

        //------------------------------------------------------------------------------------//
        // TODO:
        // 1. Change the color of the Lidar points such that
        // X=0.0m corresponds to red while X=20.0m is shown as green.
        float max_x     = worldSize.height;
        float red_ratio = 1.0 - (1.0 / max_x) * xw;
        int red_value   = (int)(255.0 * red_ratio);
        int green_value = (int)(255.0 * (1.0 - red_ratio));
        //std::cout << "xw , red , green = " << xw << " , " << red_value << " , " << green_value << endl;

        //------------------------------------------------------------------------------------//
        // 2. Remove all Lidar points on the road surface while preserving 
        // measurements on the obstacles in the scene.
        float zw = (*it).z;
        float min_z = -1.5; // Lidar z = 1.73 [m]
        if(zw < min_z)
        {
            std::cout << "zw = " << zw << std::endl;
            continue;
        }

        //------------------------------------------------------------------------------------//
        // color = B,G,R
        cv::circle(topviewImg, cv::Point(x, y), 5, cv::Scalar(0, green_value, red_value), -1);
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
      	//Syntax of cv::line()
      	// cv::line (InputOutputArray img, Point pt1, Point pt2, const Scalar &color, int thickness=1, int lineType=LINE_8, int shift=0)
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0), 3);
    }

    // display image
    string windowName = "Top-View Perspective of LiDAR data";
    cv::namedWindow(windowName, 2);
    cv::resize(topviewImg, topviewImg, cv::Size(), 0.4, 0.4);
    cv::imshow(windowName, topviewImg);
    cv::waitKey(0); // wait for key to be pressed
}

int main()
{
    showLidarTopview();
}