#include <numeric>
#include <fstream>
#include "matching2D.hpp"

#include <sys/stat.h>


using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    //-----------------------------------------------------------//
    // configure matcher
    //-----------------------------------------------------------//
    bool crossCheck = false;

    // Error Check
    if((matcherType == "MAT_BF") && (selectorType == "SEL_KNN"))
    {
        // https://stackoverflow.com/questions/33497331/knnmatch-does-not-work-with-k-1
        crossCheck = false;
    }

    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // Error Check : data type
        if((descSource.type() != CV_32F) || (descRef.type() != CV_32F))
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    //-----------------------------------------------------------//
    // perform matching task
    //-----------------------------------------------------------//
    if (selectorType.compare("SEL_NN") == 0)
    {
        // nearest neighbor (best match)
        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {
        // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        int k = 2;
        matcher->knnMatch(descSource, descRef, knn_matches, k);

        // filter matchers using descriptor distance ratio
        double misDescDistRatio = 0.8;
        for(auto it = knn_matches.begin(); it != knn_matches.end(); ++it)
        {
            if((*it)[0].distance < misDescDistRatio * (*it)[1].distance)
            {
                matches.push_back((*it)[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {
        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else
    {
        // https://qiita.com/hmichu/items/f5f1c778a155c7c414fd
        if (descriptorType == "BRIEF")
        {
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
        else if(descriptorType == "ORB")
        {
            extractor = cv::ORB::create();
        }
        else if(descriptorType == "FREAK")
        {
            extractor = cv::xfeatures2d::FREAK::create();
        }
        else if(descriptorType == "AKAZE")
        {
            extractor = cv::AKAZE::create();
        }
        else if(descriptorType == "SIFT")
        {
            extractor = cv::SIFT::create();
        }
        else if(descriptorType == "SURF")
        {
            extractor = cv::xfeatures2d::SURF::create();
        }
        else
        {
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
        }
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        //cv::waitKey(0);
    }
}

void visualizeResults(cv::Mat& img, std::string window_name, vector<cv::KeyPoint> keypoints)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = window_name;
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    //cv::waitKey(0);
}

// Detect keypoints in image
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{

    //
    if(detectorType == "SIFT")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType == "FAST")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        const int threshold = 10.0;
        const bool bNMS = true;
        cv::FAST(img, keypoints, threshold, bNMS);
    }
    else if(detectorType == "BRISK")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        const int thresh  = 30;
        const int octaves = 3;
        const float patternScale = 1.0f;
        cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(thresh, octaves, patternScale);
        brisk->detect(img, keypoints);
    }
    else if(detectorType == "ORB")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        const int nfeatures=500;
        const float scaleFactor=1.2f;
        const int nlevels=8;
        const int edgeThreshold=31;
        cv::Ptr<cv::ORB> orb = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold);
        orb->detect(img, keypoints);
    }
    else if (detectorType == "AKAZE")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        cv::AKAZE::DescriptorType descriptor_type = cv::AKAZE::DESCRIPTOR_MLDB;
        const int descriptor_size = 0;
        const int descriptor_channels = 3;
        const float threshold = 0.001f;
        const int nOctaves = 4;
        const int nOctaveLayers = 4;
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2;
        cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create(descriptor_type, descriptor_size, descriptor_channels,
                                                     threshold, nOctaves, nOctaveLayers, diffusivity);
        akaze->detect(img, keypoints);
    }
    else if (detectorType == "HARRIS")
    {
        std::cout << "----> Call " << detectorType << std::endl;
        detKeypointsHarris(keypoints, img, bVis);
    }
    else
    {
        detectorType = "Detector Type is Wrong.";
        std::cout << "----> Call " << detectorType << std::endl;
        cv::Ptr<cv::FeatureDetector> detector = cv::SIFT::create();
        detector->detect(img, keypoints);
    }

    if(bVis)
    {
        cout << "Show keypoints " << endl;
        visualizeResults(img, detectorType, keypoints);
    }
}

// Detect keypoints in image using Harris
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int block_size    = 2;    // for every pixel, a block_size × block_size neighborhood is considered
    int aperture_size = 3;    // aperture parameter for Sobel operator (must be odd)
    double k          = 0.04; // Harris parameter (see equation for details)
    cv::Mat h_response, h_response_norm; // image to store the Harris detector responses. CV_32FC1

    // calc harris corner response R, R = det(H) - k*(trace(H))^2
    cv::cornerHarris(img, h_response, block_size, aperture_size, k, cv::BORDER_DEFAULT);

    // normalize harris corner response R
    cv::normalize(h_response, h_response_norm, 0,255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());

    // find keypoints with NMS (non-maximum supression)
    const int min_response   = 100;   // minimum value for a corner in the 8bit scaled response matrix
    const double max_overlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression

    // Loop for Y direction
    for(std::size_t j=0; j < h_response_norm.rows; j++)
    {
        for(std::size_t i=0; i < h_response_norm.cols; i++)
        {
            // Get new response
            const int new_response = (int)h_response_norm.at<float>(j,i);
            // if new_response > old_response
            if(new_response > min_response)
            {
                // Create keypoint
                cv::KeyPoint new_keypoint;
                new_keypoint.pt       = cv::Point2f(i,j);
                new_keypoint.size     = 2 * aperture_size;
                new_keypoint.response = new_response;

                // Do NMS in local neighbourhood
                bool b_overlap = false;

                for(auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    double overlap_ratio = cv::KeyPoint::overlap(new_keypoint, *it);
                    // if there is overlap
                    if(overlap_ratio > max_overlap)
                    {
                        b_overlap = true;
                        if(new_keypoint.response > (*it).response)
                        {
                            *it = new_keypoint;
                            break;
                        }
                    }
                }// end of find overlap

                if(!b_overlap)
                {
                    keypoints.push_back(new_keypoint);
                }
            }
        }// end of loop : X direction
    }// end of loop : Y direction
}

////------------------------------------------------------------------------------------------------------------------
void CvTimeCount(double& t, bool FlagStart)
{
    if(FlagStart)
    {
        t = (double)cv::getTickCount();
    }
    else
    {
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency() * 1000.0; // ms
    }
}

////------------------------------------------------------------------------------------------------------------------
ReportData::ReportData()
{
    numImage            = 0;
    numKeypoints        = 0;
    numMatchedKeypoints = 0;
    cpuTimeDetect       = 0.0;
    cpuTimeDescript     = 0.0;
    nameDetector        = "no_name";
    nameDescriptor      = "no_name";
    fileNameLog         = "no_name";
    exportDirectory     = "../results";
};

void ReportData::CreateFileNameLog()
{
    filePathLog = exportDirectory + "/" + fileNameLog;
}

void ReportData::CreateExportDir()
{
    const char* dir = exportDirectory.c_str();
    mkdir(dir, S_IRWXU); // mode = read/write is OK
}

void ReportData::CalcAverage()
{
    numKeypoints         /= numImage;
    numMatchedKeypoints  /= numImage;
    cpuTimeDetect  /= numImage;
    cpuTimeDescript  /= numImage;
}

void ReportData::exportReport(bool bAppendToFile)
{
    CreateExportDir();
    CreateFileNameLog();
    CalcAverage();
    string spacer = "|";
    std::ofstream file;
    //
    if(!bAppendToFile)
    {
        file.open(filePathLog, std::ios::out);
        file << spacer << "Detector" << spacer << "Descriptor"
             << spacer << "kpt"      << spacer << "kptMatched"
             << spacer << "TimeDetect[ms]" << spacer << "TimeDescript[ms]"
             << spacer << std::endl;

        file << spacer << ":---:" << spacer << ":---:"
             << spacer << ":---:" << spacer << ":---:"
             << spacer << ":---:" << spacer << ":---:"
             << spacer << std::endl;
    }
    else
    {
        file.open(filePathLog, std::ios::app);
    }
    file << spacer << nameDetector  << spacer << nameDescriptor
         << spacer << numKeypoints  << spacer << numMatchedKeypoints
         << spacer << cpuTimeDetect << spacer << cpuTimeDescript
         << spacer << std::endl;
    //
    file.close();
}
