#define  MYENV

#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

using namespace std;

void CvTimeCount(double& t, bool FlagStart)
{
    if(FlagStart)
    {
        t = (double)cv::getTickCount();
    }
    else
    {
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    }
}

void descKeypoints1()
{
    // load image from file and convert to grayscale
    cv::Mat imgGray;
    cv::Mat img = cv::imread("../images/img1.png");
    cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

    // BRISK : Detector
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    vector<cv::KeyPoint> kptsBRISK;

    double t;
    CvTimeCount(t, true);
    detector->detect(imgGray, kptsBRISK);
    CvTimeCount(t, false);
    cout << "BRISK detector with n= " << kptsBRISK.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // BRISK : Descriptor
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::BRISK::create();
    cv::Mat descBRISK;

    CvTimeCount(t, true);
    descriptor->compute(imgGray, kptsBRISK, descBRISK);
    CvTimeCount(t, false);
    cout << "BRISK descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    cv::Mat visImageBRISK = img.clone();
    cv::drawKeypoints(img, kptsBRISK, visImageBRISK, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = "BRISK Results";
    cv::namedWindow(windowName, 1);
    imshow(windowName, visImageBRISK);

    // TODO: Add the SIFT detector / descriptor, compute the
#ifdef MYENV
    // After OpenCV v4.4.0
    // The patent of SIFT has been expired, so from OpenCV v4.4.0 we can use SIFT like "cv::SIFT".
    // SIFT : Detector
    cv::Ptr<cv::FeatureDetector> detectorSIFT   = cv::SIFT::create();
    vector<cv::KeyPoint> kptsSIFT;

    CvTimeCount(t, true);
    detectorSIFT->detect(imgGray, kptsSIFT);
    CvTimeCount(t, false);
    cout << "SIFT detector with n= " << kptsSIFT.size() << " keypoints in " << 1000.0 * t / 1.0 << " ms" << endl;

    // SIFT : Descriptor
    cv::Ptr<cv::DescriptorExtractor> descriptorSIFT = cv::SIFT::create();
    cv::Mat descSIFT;

    CvTimeCount(t, true);
    descriptorSIFT->compute(imgGray, kptsSIFT, descSIFT);
    CvTimeCount(t, false);
    cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    cv::Mat visImageSIFT = img.clone();
    cv::drawKeypoints(img, kptsSIFT, visImageSIFT, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "SIFT Results";
    cv::namedWindow(windowName, 2);
    imshow(windowName, visImageSIFT);
    cv::waitKey(0);
#else
    // Before OpenCV v4.4.0
    // SIFT : Detector
    cv::Ptr<cv::DescriptorExtractor> descriptorSIFT = cv::xfeatures2d::SIFT::create();

    CvTimeCount(t, true);
    descriptorSIFT->detect(imgGray, kptsSIFT);
    CvTimeCount(t, false);
    cout << "SIFT detector with n= " << kptsSIFT.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // SIFT : Descriptor
    cv::Ptr<cv::DescriptorExtractor> descriptorSIFT = cv::xfeatures2d::SiftDescriptorExtractor::create();
    cv::Mat descSIFT;

    CvTimeCount(t, true);
    descriptorSIFT->compute(imgGray, kptsSIFT, descSIFT);
    CvTimeCount(t, false);
    cout << "SIFT descriptor in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    visImage = img.clone();
    cv::drawKeypoints(img, kptsSIFT, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    windowName = "SIFT Results";
    cv::namedWindow(windowName, 2);
    imshow(windowName, visImage);
    cv::waitKey(0);

#endif
    cv::waitKey(0);
}

int main()
{
    descKeypoints1();
    return 0;
}