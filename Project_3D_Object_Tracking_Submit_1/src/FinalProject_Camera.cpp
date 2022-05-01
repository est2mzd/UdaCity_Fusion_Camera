
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
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

#include <sys/stat.h>
#include "MyUtility.h"

using namespace std;

//----------------------------------------------------------------------//
int parameter_study(string detectorType, string descriptorType, string folderPathResultTop, string folderPathResultEach, bool bAppendToFile);

////---------------------------------------------------------------------------------------------------------
/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    string folderPathResultTop  = "../results";
    mkdir(folderPathResultTop.c_str(), S_IRWXU);

#if SIMULATION_TYPE == 1 // FP.5
    std::vector<string> detectorTypes   = { "SHITOMASI" };
    std::vector<string> descriptorTypes = { "BRIEF" };
#endif

#if SIMULATION_TYPE == 2 // FP.6
    std::vector<string> detectorTypes   = { "HARRIS", "SHITOMASI", "ORB", "SIFT", "AKAZE", "BRISK", "FAST"};
    std::vector<string> descriptorTypes = {"SURF", "BRIEF", "ORB", "SIFT", "BRISK", "FREAK", "AKAZE"};
#endif


    bool bAppendToFile = false;
    for(string detectorType : detectorTypes)
    {
        for(string descriptorType : descriptorTypes)
        {
            if((detectorType=="SIFT") && (descriptorType == "ORB"))
            {
                continue;
            }

            // Error Check : Some descriptors like KAZE and AKAZE only work with their own keypoints.
            // Probably, SIFT should only be used as both extractor/descriptor at the same time.
            // https://github.com/kyamagu/mexopencv/issues/351
            if ((detectorType != "AKAZE") && (descriptorType == "AKAZE"))
            {
                continue;
            }
            //
            string folderPathResultEach = folderPathResultTop + "/" + detectorType + "_" + descriptorType;
            mkdir(folderPathResultEach.c_str(), S_IRWXU);
            //
            parameter_study(detectorType, descriptorType, folderPathResultTop, folderPathResultEach, bAppendToFile);
            bAppendToFile = true;
            cout << "OK : " << detectorType << " / " << descriptorType << endl;
        }
    }
}



////---------------------------------------------------------------------------------------------------------
int parameter_study(string detectorType, string descriptorType, string folderPathResultTop, string folderPathResultEach, bool bAppendToFile)
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
#if SIMULATION_TYPE == 1
    int imgEndIndex   = 77;//18;   // last file index to load
#endif
#if SIMULATION_TYPE == 2
    int imgEndIndex   = 33;//18;   // last file index to load
#endif
    int imgStepWidth  = 1;
    int imgFillWidth  = 4; // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector
    
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

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = true;            // visualize results

    ////---------------------------------------------------------------------------------------------------------------
    // For Reporting
    ReportData Report;
    Report.fileNameLog_Type01 = "FinalReport_Type01.txt";
    Report.fileNameLog_Type02 = "FinalReport_Type02.txt";
    Report.exportDirectory = "../results";
    Report.nameDetector    = detectorType;
    Report.nameDescriptor  = descriptorType;
    Report.FrameRate       = sensorFrameRate;
    string folderPathSave;
    ////---------------------------------------------------------------------------------------------------------------

    /* MAIN LOOP OVER ALL IMAGES */

    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
    {
        /* LOAD IMAGE INTO BUFFER */
        // assemble filenames for current index
        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file 
        cv::Mat img = cv::imread(imgFullFilename);

        // push image into data frame buffer
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        cout << "==================== Image Index = " << imgIndex << " ====================" << endl;
        cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold  = 0.4;

        // Fig-1
        folderPathSave = folderPathResultTop + "/" + "Fig_1_Object_Classification";
        mkdir(folderPathSave.c_str(), S_IRWXU);
        //
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights,
                      (int)imgIndex, folderPathSave, bVis);

        cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


        /* CROP LIDAR POINTS */

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

        // remove Lidar points based on distance properties
        float minX =  2.0, maxX = 20.0;
        float maxY =  2.0, minR =  0.1;
        float minZ = -1.5, maxZ = -0.9;
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

        cout << "#3 : CROP LIDAR POINTS done" << endl;

        /* CLUSTER LIDAR POINT CLOUD */

        // associate Lidar points with camera-based ROI
        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end()-1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

        // Visualize 3D objects
        // Fig-2
        folderPathSave = folderPathResultTop + "/" + "Fig_2_Object_3D";
        mkdir(folderPathSave.c_str(), S_IRWXU);
        //
        double xMinLidarTopView = show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000),
                      imgIndex, folderPathSave);

        cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
        cout << "---------------" << endl;

        /* DETECT IMAGE KEYPOINTS */

        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
        //string detectorType = "SHITOMASI"; // based on detectorType -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT

        if (detectorType.compare("SHITOMASI") == 0)
        {
            detKeypointsShiTomasi(keypoints, imgGray, bVis);
        }
        else
        {
            detKeypointsModern(keypoints, imgGray, detectorType, bVis);
        }

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
        (dataBuffer.end() - 1)->keypoints = keypoints;

        cout << "Keypoint Num = " << keypoints.size() << endl;
        cout << "#5 : DETECT KEYPOINTS done" << endl;
        cout << "---------------" << endl;

        /* EXTRACT KEYPOINT DESCRIPTORS */

        cv::Mat descriptors;
        //string descriptorType = "BRISK"; // BRISK, BRIEF, ORB, FREAK, AKAZE, SIFT
        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descriptorType);

        // push descriptors for current frame to end of data buffer
        (dataBuffer.end() - 1)->descriptors = descriptors;

        cout << "descriptors size = " << descriptors.size().height << " / " << descriptors.size().width << endl;
        cout << "#6 : EXTRACT DESCRIPTORS done" << endl;
        cout << "---------------" << endl;


        //// -----------------------------------------------------------------------------------------------------------
        if (dataBuffer.size() > 1) // wait until at least two images have been processed
        {
            /* MATCH KEYPOINT DESCRIPTORS */
            vector<cv::DMatch> matches;
            string matcherType     = "MAT_FLANN";  // MAT_BF, MAT_FLANN
            string descriptorType2 = "DES_BINARY"; // DES_BINARY, DES_HOG
            string selectorType    = "SEL_NN";     // SEL_NN, SEL_KNN

            matchDescriptors((dataBuffer.end() - 2)->keypoints,   (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorType2, matcherType, selectorType);

            // store matches in current data frame
            (dataBuffer.end() - 1)->kptMatches = matches;

            cout << "Num matches (Descriptor of Prev and Curr) = " << matches.size() << endl;
            cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

            /* TRACK 3D OBJECT BOUNDING BOXES */
            //// STUDENT ASSIGNMENT
            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
            map<int, int> bbBestMatches;
            // associate bounding boxes between current and previous frame using keypoint matches
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1));
            //// EOF STUDENT ASSIGNMENT

            // store matches in current data frame
            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

            cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


            /* COMPUTE TTC ON OBJECT IN FRONT */

            // loop over all BB match pairs
            for (auto itBBMatch = (dataBuffer.end() - 1)->bbMatches.begin(); itBBMatch != (dataBuffer.end() - 1)->bbMatches.end(); ++itBBMatch)
            {
                // find bounding boxes associates with current match
                BoundingBox *prevBB, *currBB;
                for (auto itBBoxCurr = (dataBuffer.end() - 1)->boundingBoxes.begin(); itBBoxCurr != (dataBuffer.end() - 1)->boundingBoxes.end(); ++itBBoxCurr)
                {
                    if (itBBMatch->second == itBBoxCurr->boxID) // check wether current match partner corresponds to this BB
                    {
                        currBB = &(*itBBoxCurr);
                    }
                }

                for (auto itBBPrev = (dataBuffer.end() - 2)->boundingBoxes.begin(); itBBPrev != (dataBuffer.end() - 2)->boundingBoxes.end(); ++itBBPrev)
                {
                    if (itBBMatch->first == itBBPrev->boxID) // check wether current match partner corresponds to this BB
                    {
                        prevBB = &(*itBBPrev);
                    }
                }

                // compute TTC for current match
                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                {
                    cout << "------------------- Task-2 <start> ----------------------------" << endl;
                    cout << "computeTTCLidar()" << endl;
                    cout << "  Bounding Box ID [Prev , Curr]  = [ " << prevBB->boxID << " , " << currBB->boxID << " ]"<< endl;

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                    double ttcLidar;
                    double xMinLidar, deltaVLidar;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar,xMinLidar, deltaVLidar);
                    //
                    Report.Lidar_TopView_XMin .push_back(xMinLidarTopView);
                    Report.Lidar_TTC.push_back(ttcLidar);
                    Report.Lidar_BBox_XMin.push_back(xMinLidar);
                    Report.Lidar_BBox_DeltaV.push_back(deltaVLidar);
                    //
                    //// EOF STUDENT ASSIGNMENT

                    //// STUDENT ASSIGNMENT
                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                    double ttcCamera;

                    if ( ((dataBuffer.end() - 1)->kptMatches.size() > 0) &&
                         ((dataBuffer.end() - 2)->keypoints.size()  > 0) &&
                         ((dataBuffer.end() - 1)->keypoints.size()  > 0)  )
                    {
                        // <Important Explanation>
                        // (dataBuffer.end() - 1)->kptMatches) uses 2 keypoints
                        //    Query : Previous = (dataBuffer.end() - 2)->keypoints
                        //    Train : Current  = (dataBuffer.end() - 1)->keypoints
                        //  So, If we watn previous and current xy of keypoints, we can do like below.
                        //    auto pointPrev = kptsPrev[kptMatch->queryIdx].pt;
                        //    auto pointCurr = kptsCurr[kptMatch->trainIdx].pt;
                        clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
                        computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                    }
                    else
                    {
                        cout << "There is no camera matching";
                        ttcCamera = TTC_OUTLIER;
                    }
                    double ttcDiffPer = abs((ttcLidar-ttcCamera) / ttcLidar * 100.0);
                    cout << "TTC Lidar / TTC Cemera / TTC Diff % = " << ttcLidar << " / " << ttcCamera << " / " << ttcDiffPer << endl;
                    //
                    Report.Camera_TTC.push_back(ttcCamera);
                    Report.Diff_Percent_TTC.push_back(ttcDiffPer);
                    //// EOF STUDENT ASSIGNMENT

                    //bVis = true;
                    if (bVis)
                    {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                        
                        char str[200];
                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::resize(visImg, visImg, cv::Size(), 0.3,0.3);
                        cv::imshow(windowName, visImg);
                        if(VIEW_IMAGE_MODE)
                        {
                            cout << "Press key to continue to next frame" << endl;
                            cv::waitKey(0);
                        }
                        if(FIG_SAVE_MODE)
                        {
                            ostringstream imageNumber;
                            imageNumber << setfill('0') << setw(2) << imgIndex;
                            string filePathOut = folderPathResultEach + "/" + "Final_Results_TTC_" + imageNumber.str() + ".png";
                            cv::imwrite(filePathOut, visImg);
                        }
                    }
                    //bVis = false;

                } // eof TTC computation
            } // eof loop over all BB matches
        }
    cout << endl << endl << endl;
    } // eof loop over all images

#if SIMULATION_TYPE == 1 // FP.5
    Report.exportReport_Type01(bAppendToFile);
#endif

#if SIMULATION_TYPE == 2 // FP.6
    Report.exportReport_Type02(bAppendToFile);
#endif

    return 0;
}
