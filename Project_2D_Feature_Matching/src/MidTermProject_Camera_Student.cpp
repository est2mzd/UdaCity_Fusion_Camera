/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"

#include <sys/stat.h>

using namespace std;

//----------------------------------------------------------------------//
int parameter_study(string detectorType, string descriptorType, bool bAppendToFile);
std::vector<string> detectorTypes = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
string detectorType = detectorTypes[2]; // 0 to 6;
std::vector<string> descriptorTypes = {"BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT", "SURF"};
string descriptorType = descriptorTypes[4];
bool bAppendToFile = false;

////---------------------------------------------------------------------------------------------------------
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    std::vector<string> detectorTypes   = { "HARRIS", "SHITOMASI", "ORB", "SIFT", "AKAZE", "BRISK", "FAST"};
    std::vector<string> descriptorTypes = {"SURF", "BRIEF", "ORB", "SIFT", "BRISK", "FREAK", "AKAZE"};
    bool bAppendToFile = false;
    for(string detectorType : detectorTypes)
    {
        for(string descriptorType : descriptorTypes)
        {
            if((detectorType=="SIFT") && (descriptorType == "ORB"))
            {
                continue;
            }
            /*
            struct stat statBuf;
            string dir = "../results/" + detectorType + "_" + descriptorType;
            if(stat(dir.c_str(), &statBuf) == 0)
            {
                bAppendToFile = true;
                continue;
            }
            */
            //
            parameter_study(detectorType, descriptorType, bAppendToFile);
            bAppendToFile = true;
            cout << "OK : " << detectorType << " / " << descriptorType << endl;
        }
    }
}

////---------------------------------------------------------------------------------------------------------
int parameter_study(string detectorType, string descriptorType, bool bAppendToFile)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix   = "KITTI/2011_09_26/image_00/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex  = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex    = 9; // last file index to load
    int imgFillWidth   = 4; // no. of digits which make up the file index (e.g. img-0001.png)

    // misc
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    //vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    RingBuffer<DataFrame> dataBuffer(dataBufferSize); // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results

    ////---------------------------------------------------------------------------------------------------------------
    // For Reporting
    ReportData Report;
    Report.fileNameLog     = "MidTermReport.txt";
    Report.exportDirectory = "../results";
    double cpuTimeDetect;
    double cpuTimeDescript;
    ////---------------------------------------------------------------------------------------------------------------

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
    {
        /* LOAD IMAGE INTO BUFFER */

        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFileName     = imgPrefix + imgNumber.str() + imgFileType;
        string imgFullFilename = imgBasePath + imgFileName;

        // load image from file and convert to grayscale
        cv::Mat img, imgGray;
        img = cv::imread(imgFullFilename);
        cv::cvtColor(img, imgGray, cv::COLOR_BGR2GRAY);

        //// STUDENT ASSIGNMENT ---------------------------------------------------------------------------------------
        //// TASK MP.1 -> replace the following code with ring buffer of size dataBufferSize

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = imgGray;
        dataBuffer.Write(frame);

        //// EOF STUDENT ASSIGNMENT -----------------------------------------------------------------------------------
        cout << "=============================================" << endl;
        cout << "#1 : LOAD IMAGE INTO BUFFER done:" <<  imgFileName << endl;

        /* DETECT IMAGE KEYPOINTS */

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image

        //// STUDENT ASSIGNMENT ---------------------------------------------------------------------------------------
        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection
        //// based on detectorType -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT


        CvTimeCount(cpuTimeDetect, true);
        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, bVis);
        }
        CvTimeCount(cpuTimeDetect, false);
        cout << "End of Task MP. 2" << endl;
        //// EOF STUDENT ASSIGNMENT -----------------------------------------------------------------------------------

        //// STUDENT ASSIGNMENT ---------------------------------------------------------------------------------------
        //// TASK MP.3 -> only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150);
        if (bFocusOnVehicle)
        {
            cout << "before : keypoints size = " << keypoints.size() << endl;
            for(auto it = keypoints.begin(); it != keypoints.end(); ++it)
            {
                if(!vehicleRect.contains(it->pt))
                {
                    keypoints.erase(it);
                    it--;
                }
            }
            cout << "after  : keypoints size = " << keypoints.size() << endl;
        }
        cout << "End of Task MP. 3" << endl;
        //// EOF STUDENT ASSIGNMENT -----------------------------------------------------------------------------------

        // ------------------------------------------------------------------------------------------------------------
        // optional : limit number of keypoints (helpful for debugging and learning)
        bool bLimitKpts = false;
        if (bLimitKpts)
        {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") == 0)
            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }
        // ------------------------------------------------------------------------------------------------------------

        // push keypoints and descriptor for current frame to end of data buffer
        dataBuffer.ReadLast()->keypoints = keypoints;
        cout << "#2 : DETECT KEYPOINTS done" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        //// STUDENT ASSIGNMENT // ------------------------------------------------------------------------------------
        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection
        //// based on descriptorType -> BRIEF, ORB, FREAK, AKAZE, SIFT
        cv::Mat descriptors;


        // Error Check : Some descriptors like KAZE and AKAZE only work with their own keypoints.
        // Probably, SIFT should only be used as both extractor/descriptor at the same time.
        // https://github.com/kyamagu/mexopencv/issues/351
        if ((detectorType != "AKAZE") && (descriptorType == "AKAZE"))
        {
            return 0;
        }
        CvTimeCount(cpuTimeDescript, true);
        descKeypoints(dataBuffer.ReadLast()->keypoints, dataBuffer.ReadLast()->cameraImg, descriptors, descriptorType);
        CvTimeCount(cpuTimeDescript, false);
        //// EOF STUDENT ASSIGNMENT // --------------------------------------------------------------------------------

        // push descriptors for current frame to end of data buffer
        dataBuffer.ReadLast()->descriptors = descriptors;
        cout << "#3 : EXTRACT DESCRIPTORS done" << endl;
        cout << "dataBuffer.size() = " << dataBuffer.size() << endl;



        //// -----------------------------------------------------------------------------------------------------------
        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;
            string matcherType     = "MAT_FLANN";  // MAT_BF, MAT_FLANN
            string descriptorType2 = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType    = "SEL_NN";     // SEL_NN, SEL_KNN

            //// STUDENT ASSIGNMENT -----------------------------------------------------------------------------------
            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
            matchDescriptors(dataBuffer.Read2ndLast()->keypoints, dataBuffer.ReadLast()->keypoints,
                             dataBuffer.Read2ndLast()->descriptors, dataBuffer.ReadLast()->descriptors,
                             matches, descriptorType2, matcherType, selectorType);
            //// EOF STUDENT ASSIGNMENT  ------------------------------------------------------------------------------

            // store matches in current data frame
            dataBuffer.ReadLast()->kptMatches = matches;

            //// -----------------------------------------------------------------------------------------------------------
            Report.nameDetector    = detectorType;
            Report.nameDescriptor  = descriptorType;
            Report.numImage            += 1;
            Report.numKeypoints        += keypoints.size();
            Report.numMatchedKeypoints += matches.size();
            Report.cpuTimeDetect       += cpuTimeDetect;
            Report.cpuTimeDescript     += cpuTimeDescript;

            cout << "#4 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            //  -------------------------------------------------------------------------------------------------------
            // visualize matches between current and previous image
            bVis = true;
            if (bVis)
            {
                cv::Mat matchImg = (dataBuffer.ReadLast()->cameraImg).clone();
                cv::drawMatches(dataBuffer.Read2ndLast()->cameraImg, dataBuffer.Read2ndLast()->keypoints,
                                dataBuffer.ReadLast()->cameraImg, dataBuffer.ReadLast()->keypoints,
                                matches, matchImg,
                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                string windowName = "Matching keypoints between two camera images";
                cv::namedWindow(windowName, 7);
                cv::resize(matchImg, matchImg, cv::Size(), 0.7,0.7);
                cv::imshow(windowName, matchImg);
                //
                bool bSaveImg = true;
                if(bSaveImg)
                {
                    string folderPath  = "../results/" + detectorType + "_" + descriptorType;
                    mkdir(folderPath.c_str(), S_IRWXU);
                    string filePathOut =  folderPath + "/" + imgNumber.str() + ".png";
                    cv::imwrite(filePathOut, matchImg);
                }
                //
                cout << "Press key to continue to next image" << endl;
                //cv::waitKey(0); // wait for key to be pressed
            }
            //  -------------------------------------------------------------------------------------------------------
        } // End of Matching
        //// -----------------------------------------------------------------------------------------------------------

    } // eof loop over all images


    Report.exportReport(bAppendToFile);

    cout << "----< Result Check >----" << endl;
    cout << "dataBuffer.size() = " << dataBuffer.size() << endl;

    return 0;
}
