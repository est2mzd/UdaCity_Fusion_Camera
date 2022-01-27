#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void magnitudeSobel()
{
    // load image from file
    cv::Mat img;
    img = cv::imread("../images/img1gray.png");

    // convert image to grayscale
    cv::Mat imgGray;
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // apply smoothing using the GaussianBlur() function from the OpenCV
    // ToDo : Add your code here
    cv::Mat imgBlur     = imgGray.clone();            // create the same size array of imgGray
    cv::Size kernelBlur = cv::Size(5,5);
    double stdDevX      = 2.0;
    cv::GaussianBlur(imgGray, imgBlur, kernelBlur, stdDevX);

    // create filter kernels using the cv::Mat datatype both for x and y
    // ToDo : Add your code here
    float sobelX[9] = {-1,  0,  1, -2, 0, 2, -1, 0, 1};
    float sobelY[9] = {-1, -2, -1,  0, 0, 0,  1, 2, 1};
    cv::Mat kernelSobelX   = cv::Mat(3, 3, CV_32F, sobelX);
    cv::Mat kernelSobelY   = cv::Mat(3, 3, CV_32F, sobelY);

    // apply filter using the OpenCv function filter2D()
    // ToDo : Add your code here
    cv::Mat imgSobelX, imgSobelY;
    cv::filter2D(imgBlur, imgSobelX, -1, kernelSobelX, cv::Point(-1,-1),0, cv::BORDER_DEFAULT);
    cv::filter2D(imgBlur, imgSobelY, -1, kernelSobelY, cv::Point(-1,-1),0, cv::BORDER_DEFAULT);

    // compute magnitude image based on the equation presented in the lesson 
    // ToDo : Add your code here
    cv::Mat imgMagnitude = imgGray.clone();
    for(int r = 0; r < imgMagnitude.rows; r++)
    {
        for(int c = 0; c < imgMagnitude.cols; c++)
        {
            imgMagnitude.at<unsigned char>(r,c) = sqrt( pow(imgSobelX.at<unsigned char >(r,c),  2) +
                                                                 pow(imgSobelY.at<unsigned char >(r,c),  2));
        }
    }

    // show result
    string windowName = "Gaussian Blurring";
    cv::namedWindow(windowName, 1); // create window
    cv::imshow(windowName, imgMagnitude);
    cv::waitKey(0); // wait for keyboard input before continuing
}

int main()
{
    magnitudeSobel();
}