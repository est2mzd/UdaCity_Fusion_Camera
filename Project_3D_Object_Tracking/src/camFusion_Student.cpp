
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

#include <iomanip>
#include "MyUtility.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
                         cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat Coord3D(4, 1, cv::DataType<double>::type);
    cv::Mat Coord2D(3, 1, cv::DataType<double>::type);

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
        // pixel coordinates
        pointImagePlane.x = Coord2D.at<double>(0, 0) / Coord2D.at<double>(2, 0);
        pointImagePlane.y = Coord2D.at<double>(1, 0) / Coord2D.at<double>(2, 0);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator itBBox = boundingBoxes.begin(); itBBox != boundingBoxes.end(); ++itBBox)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBBox;
            smallerBBox.x = (*itBBox).roi.x + shrinkFactor * (*itBBox).roi.width / 2.0;
            smallerBBox.y = (*itBBox).roi.y + shrinkFactor * (*itBBox).roi.height / 2.0;
            smallerBBox.width = (*itBBox).roi.width * (1 - shrinkFactor);
            smallerBBox.height = (*itBBox).roi.height * (1 - shrinkFactor);

            // check wether a point is within current bounding box
            if (smallerBBox.contains(pointImagePlane))
            {
                // <Modification-1/3> : avoid duplicated inclusion of lidar points
                itBBox->lidarPoints.push_back(*itLidar);
                lidarPoints.erase(itLidar);
                itLidar--;
                break;
            }
        } // eof loop over all bounding boxes
    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
double show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize,
                     int imageID, std::string folderPathSave)
{
    cout << "------------------ show3DObjects() <start> ------------------" << endl;
    cout << "  Num of bounding box = " << boundingBoxes.size() << endl;
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    double xMinOut;
    int    targetBBoxId;

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // Error Check
        if(it1->lidarPoints.size() == 0)
        {
            continue;
        }

        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top2D view image
        int   top2D=1e8, left2D=1e8, bottom2D=0.0, right2D=0.0;
        float xMin3D=1e8, xMax3D=-1e8, yMin3D=1e8, yMax3D=-1e8;
        xMinOut = 0.0;

        int numLidarPoints = it1->lidarPoints.size();
        cout << "  Num of Lidar Points = " << numLidarPoints << endl;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float x3D = (*it2).x; // world position in m with x2D facing forward from sensor
            float y3D = (*it2).y; // world position in m with y2D facing left2D from sensor
            xMin3D = xMin3D < x3D ? xMin3D : x3D;
            xMax3D = xMax3D > x3D ? xMax3D : x3D;
            yMin3D = yMin3D < y3D ? yMin3D : y3D;
            yMax3D = yMax3D > y3D ? yMax3D : y3D;

            // top2D-view coordinates
            int y2D = (-x3D * imageSize.height / worldSize.height) + imageSize.height;
            int x2D = (-y3D * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top2D    = (top2D    < y2D) ? top2D    : y2D;
            left2D   = (left2D   < x2D) ? left2D   : x2D;
            bottom2D = (bottom2D > y2D) ? bottom2D : y2D;
            right2D  = (right2D  > x2D) ? right2D  : x2D;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x2D, y2D), 4, currColor, -1);
            //
            xMinOut += xMin3D;
        }
        xMinOut /= numLidarPoints;

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left2D, top2D), cv::Point(right2D, bottom2D), cv::Scalar(0, 0, 0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        int thickness = 4;
        int lineType  = cv::LINE_8;
        targetBBoxId  = it1->boxID;
        sprintf(str1, "Box_Id=%d, Lidar Point Num=%d", targetBBoxId, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left2D - 250, bottom2D + 50), cv::FONT_ITALIC, 2, currColor, thickness, lineType);
        //sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xMin3D, yMax3D - yMin3D);
        sprintf(str2, "x_min=%2.2f m, x_max=%2.2f m", xMin3D, xMax3D);
        putText(topviewImg, str2, cv::Point2f(left2D - 250, bottom2D + 125), cv::FONT_ITALIC, 2, currColor, thickness, lineType);
        sprintf(str2, "y_min=%2.2f m, y_max=%2.2f m", yMin3D, yMax3D);
        putText(topviewImg, str2, cv::Point2f(left2D - 250, bottom2D + 195), cv::FONT_ITALIC, 2, currColor, thickness, lineType);
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
    string windowName = "3D Objects";
    if(VIEW_IMAGE_MODE)
    {
        cv::namedWindow(windowName, 7);
        cv::resize(topviewImg, topviewImg, cv::Size(), 0.3,0.3);
        cv::imshow(windowName, topviewImg);
    }
    else
    {
        cv::namedWindow(windowName, 1);
        cv::imshow(windowName, topviewImg);
    }
    if(FIG_SAVE_MODE)
    {
        ostringstream imageNumber;
        imageNumber << setfill('0') << setw(2) << imageID;
        string filePathOut = folderPathSave + "/" + "image_" + imageNumber.str() + ".png";
        cv::imwrite(filePathOut, topviewImg);
    }
    cout << "  Target Bounding Box ID = " << targetBBoxId << endl;
    cout << "  X_Min_Lidar_Top_View   = " << xMinOut << "[m]" << endl;
    cout << "------------------ show3DObjects() <end> ------------------" << endl;

    return xMinOut;
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &bBoxCurr, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    cout << "------------------- Task-3 <start> ----------------------------" << endl;
    cout << "clusterKptMatchesWithROI()" << endl;
    //----------------------------------------------------//
    // Calculate the average distance
    std::vector<double> eucliDists;

    for(auto itMatchPair = kptMatches.begin(); itMatchPair != kptMatches.end(); ++itMatchPair)
    {
        // calc distance
        auto pointPrev = kptsPrev[itMatchPair->queryIdx].pt;
        auto pointCurr = kptsCurr[itMatchPair->trainIdx].pt;
        double distance = cv::norm(pointPrev - pointCurr);

        // store date
        eucliDists.push_back(distance);
    }

    //----------------------------------------------------//
    // Calculate Average and Variance(=sigma)
    double eucliDistAve = 0.0;
    double sigma = 0.0;
    CalcAverageSigma(eucliDists, eucliDistAve, sigma);

    //----------------------------------------------------//
    // Update
    double sigmaThresh = 1.0; // hyper parameter
    int numFoundKey = 0;
    for(auto itMatchPair = kptMatches.begin(); itMatchPair != kptMatches.end(); ++itMatchPair)
    {
        // calc distance
        double distance = cv::norm(kptsPrev[itMatchPair->queryIdx].pt - kptsCurr[itMatchPair->trainIdx].pt);
        double distanceDiff = abs(distance - eucliDistAve);

        // check the distance and Update keypoints and matches
        if( distanceDiff < sigma * sigmaThresh)
        {
            numFoundKey +=1;
            bBoxCurr.keypoints.push_back(kptsPrev[itMatchPair->queryIdx]);
            bBoxCurr.kptMatches.push_back(*itMatchPair);
        }
    }

    cout << "Num Found Keypoints = " << numFoundKey << endl;
    cout << "------------------- Task-3 <end> ----------------------------" << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    cout << "------------------- Task-4 <start> ----------------------------" << endl;
    cout << "computeTTCCamera()" << endl;

    // Error Check
    if( (kptsPrev.size()==0) || (kptsCurr.size()==0) || kptMatches.size()==0)
    {
        cout << ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Task-4 : There is no matching";
        TTC = TTC_OUTLIER;
        return;
    }

    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame

    for (auto itKptMatchOut = kptMatches.begin(); itKptMatchOut != kptMatches.end() - 1; ++itKptMatchOut)
    { // outer keypoint loop : (start) to (end-1)
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterPrev = kptsPrev.at(itKptMatchOut->queryIdx);
        cv::KeyPoint kpOuterCurr = kptsCurr.at(itKptMatchOut->trainIdx);

        for (auto itKptMatchIn = kptMatches.begin() + 1; itKptMatchIn != kptMatches.end(); ++itKptMatchIn)
        { // inner keypoint loop : (start+1) to (end)
            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerPrev = kptsPrev.at(itKptMatchIn->queryIdx);
            cv::KeyPoint kpInnerCurr = kptsCurr.at(itKptMatchIn->trainIdx);

            // compute distances and distance ratios
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    //-----------------------------------------------------------------------------
    // TODO: STUDENT TASK (replacement for meanDistRatio)
    double medianDistRatio = GetMedian(distRatios);
    double dT = 1 / frameRate;
    double denominator = (1 - medianDistRatio);

    if( abs(denominator) < 0.001 )
    {
        TTC = TTC_OUTLIER;
    }
    else
    {
        TTC = -dT / denominator;

        // TTC Limit Filter
        /*
        if( abs(TTC) > 100)
        {
            TTC = TTC / abs(TTC) * 100.0;
        }
        */
    }

    cout << "  Distance Ratio (h1/h0) = " << medianDistRatio << "[ Camera can not calculate dV and dX]"<< endl;
    cout << "  TTC of Camera = " << TTC << std::endl;
    cout << "------------------- Task-4 <end> ----------------------------" << endl;
}




void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    // Get num of bounding boxes
    int numPrevBBoxes = prevFrame.boundingBoxes.size();
    int numCurrBBoxes = currFrame.boundingBoxes.size();
    cv::Mat matchCountTable = cv::Mat::zeros(numPrevBBoxes, numCurrBBoxes, CV_32S);

    cout << "------------------- Task-1 <start> ----------------------------" << endl;
    cout << "matchBoundingBoxes()" << endl;
    // Loop-1 for all matching pairs
    for (auto itMatch = matches.begin(); itMatch != matches.end(); ++itMatch) {
        // Get keypoints
        int prevKeyID = itMatch->queryIdx;
        int currKeyID = itMatch->trainIdx;
        auto prevKeyPoint = prevFrame.keypoints[prevKeyID].pt;
        auto currKeyPoint = currFrame.keypoints[currKeyID].pt;

        // Loop for all current bounding boxes
        for (int currBBoxID = 0; currBBoxID < numCurrBBoxes; ++currBBoxID) {
            // get a bounding box which contain keypoint
            BoundingBox currBBox = currFrame.boundingBoxes[currBBoxID];

            if (currBBox.roi.contains(currKeyPoint)) {
                for (int prevBBoxID = 0; prevBBoxID < numPrevBBoxes; ++prevBBoxID) {
                    // get a bounding box which contain keypoint
                    BoundingBox prevBBox = prevFrame.boundingBoxes[prevBBoxID];

                    if (prevBBox.roi.contains(prevKeyPoint)) {
                        // Update Counter
                        matchCountTable.at<int>(prevBBoxID, currBBoxID) += 1;
                    }
                }
            }
        }
    }// end : Loop-1 for all matching pairs

    // View Matching Table
    if (DEBUG_MODE)
    {
        cout << "Mathcing Counter Table : Row = Previous BBox ID / Col = Current BBox ID" << endl;
        for (int prevBBoxID = 0; prevBBoxID < numPrevBBoxes; ++prevBBoxID) {
            for (int currBBoxID = 0; currBBoxID < numCurrBBoxes; ++currBBoxID) {
                ostringstream count;
                count << setfill(' ') << setw(3) << matchCountTable.at<int>(prevBBoxID, currBBoxID) ;
                cout << count.str() << " , ";
            }
            cout << endl;
        }
    }

    // Create Map
    // key   = previous bounding box id
    // value = current bounding box id
    for(int prevBBoxID = 0; prevBBoxID < numPrevBBoxes; ++prevBBoxID)
    {
        int targetCurrBoxID = 0;
        int maxCounter = -1;
        //
        for(int currBBoxID = 0; currBBoxID < numCurrBBoxes; ++currBBoxID)
        {
            int tmpCounter = matchCountTable.at<int>(prevBBoxID, currBBoxID);

            if( tmpCounter > maxCounter)
            {
                maxCounter = tmpCounter;
                targetCurrBoxID = currBBoxID;
            }
        }
        //
        if (maxCounter > 0)
        {
            bbBestMatches.insert({prevBBoxID, targetCurrBoxID});
        }
    }

    // View Map
    if(DEBUG_MODE)
    {
        cout << "####### bbMatches {Prev , Curr} ##########" << endl;
        for(auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
        {
            cout << it->first << ", " << it->second << endl;
        }
    }
    cout << "Create : map<int, int> bbBestMatches;" << endl;
    cout << "------------------- Task-1 <end> ----------------------------" << endl;
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC,
                     double& xMinLidar, double& deltaVLidar)
{
    cout << "  Lidar point Num [Prev , Curr] = [ " << lidarPointsPrev.size() << " , " << lidarPointsCurr.size() << " ]" << endl;
    // time between two measurements in seconds
    double deltaTime = 1.0 / frameRate;

    double averageXPrev = 0.0;
    double averageXCurr = 0.0;
    int numXPrev = lidarPointsPrev.size();
    int numXCurr = lidarPointsCurr.size();
    vector<double> PrevXs, CurrXs;

    for(auto itPrev = lidarPointsPrev.begin(); itPrev != lidarPointsPrev.end(); ++itPrev)
    {
        PrevXs.push_back(itPrev->x);
    }
    //
    for(auto itCurr = lidarPointsCurr.begin(); itCurr != lidarPointsCurr.end(); ++itCurr)
    {
        CurrXs.push_back(itCurr->x);
    }
    double PrevXMedian = GetMedian(PrevXs);
    double CurrXMedian = GetMedian(CurrXs);

    // Compute TTC from both measurements
    double deltaX = abs(PrevXMedian - CurrXMedian);
    double deltaV = deltaX / deltaTime;

    if (deltaX > 0.001)
    {
        TTC = CurrXMedian / deltaV;
    }
    else
    {
        // avoid "divide by zero"
        cout << "   <TTC Lidar> : Avoid division by zero " <<  endl;
        TTC = TTC_OUTLIER;
    }

    // TTC Limit Filter
    /*
    if( abs(TTC) > 100)
    {
        TTC = TTC / abs(TTC) * 100.0;
    }
     */

    xMinLidar   = CurrXMedian;
    deltaVLidar = deltaV;
    cout << "AverageX = " << CurrXMedian << " , deltaV = " << deltaV <<  endl;
    cout << "  TTC of Lidar = " << TTC << endl;
    cout << "------------------- Task-2 <end> ----------------------------" << endl;
}
