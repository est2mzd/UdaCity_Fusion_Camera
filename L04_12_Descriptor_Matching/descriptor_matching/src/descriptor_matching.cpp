#include <iostream>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>

#include "structIO.hpp"

using namespace std;

string CreateFileName(string matcherType, string selectorType, bool crossCheck)
{
    //string Dir = "~/HDD1/UdaCity/Fusion/SFND_Camera_Object_Tracking/L04_12_Descriptor_Matching/descriptor_matching/src/results/";
    //string Dir = "./resutls/";
    //string Dir = "/resutls/";
    string Dir = "";
    string Name = Dir + matcherType + "_" + selectorType + "_";

    if(crossCheck)
    {
        Name = Name + "CrossCheckOn";
    }
    else
    {
        Name = Name + "CrossCheckOff";
    }
    Name = Name + ".png";
    return Name;
}

void matchDescriptors(cv::Mat &imgSource, cv::Mat &imgRef, vector<cv::KeyPoint> &kPtsSource, vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      vector<cv::DMatch> &matches, string descriptorType, string matcherType, string selectorType, float figureSizeRatio=0.7)
{
    //-----------------------------------------------------------//
    // configure matcher
    //-----------------------------------------------------------//
    bool crossCheck = true;

    // Error Check
    if((matcherType == "MAT_BF") && (selectorType == "SEL_KNN"))
    {
        // https://stackoverflow.com/questions/33497331/knnmatch-does-not-work-with-k-1
        crossCheck = false;
    }

    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType == "MAT_BF")
    {
        // Brute Force Matching
        int normType = (descriptorType == "DES_BINARY") ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck << ": ";
    }
    else if (matcherType == "MAT_FLANN")
    {
        // FLANN
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }

        //... TODO : implement FLANN matching
        cout << "FLANN matching: ";
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    //-----------------------------------------------------------//
    // perform matching task
    //-----------------------------------------------------------//
    if (selectorType =="SEL_NN")
    {
        // nearest neighbor (best match)
        double t = (double)cv::getTickCount();
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "(NN)  with n = " << matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;
    }
    else if (selectorType == "SEL_KNN")
    {
        // k nearest neighbors (k=2)
        // TODO : implement k-nearest-neighbor matching
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        int k = 2;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << "(KNN) with n = " << knn_matches.size() << " matches in " << 1000.0 * t << " ms";

        // TODO : filter matches using descriptor distance ratio test
        double misDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if((*it)[0].distance < misDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
        cout << " / keypoints removed = " << knn_matches.size() - matches.size() << endl;
    }

    // visualize results
    cv::Mat matchImg = imgRef.clone();
    cv::drawMatches(imgSource, kPtsSource, imgRef, kPtsRef, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    string windowName = "Matching keypoints between two camera images (best 50)";
    cv::namedWindow(windowName, 7);
    cv::resize(matchImg, matchImg, cv::Size(), 0.7,0.7);
    cv::imshow(windowName, matchImg);
    //string fine_name = CreateFileName(matcherType, selectorType, crossCheck);
    //cv::imwrite(fine_name, matchImg);
    //cout << CreateFileName(matcherType, selectorType, crossCheck) << endl;
    cv::waitKey(0);
}

int main()
{
    // Load Images
    cv::Mat imgSource = cv::imread("../images/img1gray.png");
    cv::Mat imgRef = cv::imread("../images/img2gray.png");

    // Load Keypoints which are created before.
    // Keypoints are used to visualize images only.
    vector<cv::KeyPoint> kptsSource, kptsRef;
    bool bLoadLargeData = true;
    if(bLoadLargeData)
    {
        readKeypoints("../dat/C35A5_KptsSource_BRISK_large.dat", kptsSource);
        readKeypoints("../dat/C35A5_KptsRef_BRISK_large.dat", kptsRef);
    }
    else
    {
        readKeypoints("../dat/C35A5_KptsSource_BRISK_small.dat", kptsSource);
        readKeypoints("../dat/C35A5_KptsRef_BRISK_small.dat", kptsRef);
    }

    // Load Descriptors which are created before.
    // Descriptors are used to Keypoints Matching.
    cv::Mat descSource, descRef;
    if(bLoadLargeData)
    {
        readDescriptors("../dat/C35A5_DescSource_BRISK_large.dat", descSource);
        readDescriptors("../dat/C35A5_DescRef_BRISK_large.dat", descRef);
    }
    else
    {
        readDescriptors("../dat/C35A5_DescSource_BRISK_small.dat", descSource);
        readDescriptors("../dat/C35A5_DescRef_BRISK_small.dat", descRef);
    }

    // Perform Matching
    vector<cv::DMatch> matches;
    string matcherType = "MAT_FLANN"; // MAT_BF or MAT_FLANN
    string descriptorType = "DES_BINARY"; // DES_BINARY or others...
    string selectorType = "SEL_KNN"; // SEL_NN or SEL_KNN
    float figureSizeRatio = 0.7;
    //matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, figureSizeRatio);

    matches.clear();
    matcherType = "MAT_BF";
    selectorType = "SEL_NN";
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, figureSizeRatio);

    matches.clear();
    matcherType = "MAT_BF";
    selectorType = "SEL_KNN";
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, figureSizeRatio);

    matches.clear();
    matcherType = "MAT_FLANN";
    selectorType = "SEL_NN";
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, figureSizeRatio);

    matches.clear();
    matcherType = "MAT_FLANN";
    selectorType = "SEL_KNN";
    matchDescriptors(imgSource, imgRef, kptsSource, kptsRef, descSource, descRef, matches, descriptorType, matcherType, selectorType, figureSizeRatio);
}