#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

using namespace std;

void cornernessHarris()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1.png");
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // convert to grayscale

    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    double k = 0.04;       // Harris parameter (see equation for details)

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    std::cout << "img size = (" << img.rows << " , " << img.cols << ")" << std::endl; //img size = (375 , 1242)
    std::cout << "dst size = (" << dst.rows << " , " << dst.cols << ")" << std::endl; //dst size = (375 , 1242)

    // visualize results
    string windowName = "Harris Corner Detector Response Matrix";
    cv::namedWindow(windowName, 4);
    cv::imshow(windowName, dst_norm_scaled);
    cv::waitKey(0);

    // TODO: Your task is to locate local maxima in the Harris response matrix 
    // and perform a non-maximum suppression (NMS) in a local neighborhood around 
    // each maximum. The resulting coordinates shall be stored in a list of keypoints 
    // of the type `vector<cv::KeyPoint>`.
    vector<cv::KeyPoint> keypoints;
    const int minResponse = 100;   // minimum value for a corner in the 8bit scaled response matrix
    const double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression

    // Loop for Y direction
    for(std::size_t j; j < dst_norm.rows; j++)
    {
        // Loop for X direction
        for(std::size_t i; i < dst_norm.cols; i++)
        {
            // Get New Response
            const int response = (int)dst_norm.at<float>(j,i);
            // If New Response > Old Response
            if(response > minResponse)
            {
                // Create KeyPoint
                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i,j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool bOverlap = false;
                //
                for(auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    double overlapRatio = cv::KeyPoint::overlap(newKeyPoint, *it);
                    // if there is overlap
                    if(overlapRatio > maxOverlap)
                    {
                        bOverlap = true;

                        // if (Overlap == true) && ()
                        // we replace (*it) with newKeyPointthe
                        // (*it) is the keypoint which is saved in keypoints array.
                        if(newKeyPoint.response > (*it).response)
                        {
                            *it = newKeyPoint;
                            break;
                        }
                    }
                }// end of for : find overlap

                if(!bOverlap)
                {
                    keypoints.push_back(newKeyPoint);
                }
            }
        } // end of for : X direction
    } // end of for : Y direction

    //
    windowName = "Harris corner detection results";
    cv::namedWindow( windowName, 5);
    cv::Mat visImage = dst_norm_scaled.clone();
    cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cv::imshow(windowName, visImage);
    cv::waitKey(0);
}

int main()
{
    cornernessHarris();
}