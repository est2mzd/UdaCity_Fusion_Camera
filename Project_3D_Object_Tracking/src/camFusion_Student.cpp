
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
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
                itBBox->lidarPoints.push_back(*itLidar);
                lidarPoints.erase(itLidar);
                itLidar--;
                break;
            }


            /*
            // check wether point is within current bounding box
            if (smallerBBox.contains(pointImagePlane))
            {
                enclosingBoxes.push_back(itBBox);
            }
             */

        } // eof loop over all bounding boxes

        /*
        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*itLidar);
        }
         */

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
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
    cv::namedWindow(windowName, 7);
    cv::resize(topviewImg, topviewImg, cv::Size(), 0.3,0.3);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &bBoxCurr, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    cout << "------------------- Task-3 <start> ----------------------------" << endl;
    /*
    // Type-1 : Calculate the average distance
    // this calculate the distance which use unmatched keypoints also. 
    cout << "bBoxCurr->keypoints.size() = "  << bBoxCurr.keypoints.size() << endl;  // 0
    cout << "bBoxCurr->kptMatches.size() = " << bBoxCurr.kptMatches.size() << endl; // 0
    cout << "kptsPrev.size() = " << kptsPrev.size() << endl; // 50
    cout << "kptsCurr.size() = " << kptsCurr.size() << endl; // 48
    cout << "kptMatches.size() = " << kptMatches.size() << endl; // 50

    vector<double> distances;
    double distance_sum = 0;
    int counter = 0;
    for (auto itKeyPrev = kptsPrev.begin(); itKeyPrev != kptsPrev.end(); ++itKeyPrev)
    {
        for (auto itKeyCurr = kptsCurr.begin(); itKeyCurr != kptsCurr.end(); ++itKeyCurr)
        {
            if(bBoxCurr.roi.contains(itKeyPrev->pt) && bBoxCurr.roi.contains(itKeyCurr->pt))
            {
                double dX = (itKeyPrev->pt.x - itKeyCurr->pt.x);
                double dY = (itKeyPrev->pt.y - itKeyCurr->pt.y);
                double distance = pow((dX*dX + dY*dY), 0.5);
                distance_sum += distance;
                counter +=1;
                cout << counter << ": distance = " << distance << endl;
                cout << "    " << "roi.width  = " << bBoxCurr.roi.width << endl;
                cout << "    " << "roi.height = " << bBoxCurr.roi.height << endl;
            }
        }
    }

    double distance_ave = distance_sum / counter;
    cout << "distance_ave = " << distance_ave << endl;
    // End of Type-1
    */

    //----------------------------------------------------//
    // Type-2 :

    //----------------------------------------------------//
    // Calculate the average distance
    std::vector<double> eucliDists;
    double eucliDistAve = 0;
    for(auto itMatchPair = kptMatches.begin(); itMatchPair != kptMatches.end(); ++itMatchPair)
    {
        // calc distance
        auto pointPrev = kptsPrev[itMatchPair->queryIdx].pt;
        auto pointCurr = kptsPrev[itMatchPair->trainIdx].pt;
        double distance = cv::norm(pointPrev - pointCurr);

        // store date
        eucliDistAve += distance;
        eucliDists.push_back(distance);
    }
    int numDist = eucliDists.size();
    eucliDistAve /= numDist;

    std::sort(eucliDists.begin(), eucliDists.end()); // ascending sort
    cout << "eucliDists Min/ Ave /Max = " << eucliDists[0] << " / " << eucliDistAve << " / " << eucliDists[numDist-1] << endl;

    //----------------------------------------------------//
    // Calculate Variance(=sigma)
    double sigma = 0.0;
    for(int i=0; i<numDist; ++i)
    {
        sigma += pow((eucliDists[i] - eucliDistAve),2);
    }
    sigma = pow( (sigma/numDist), 0.5);

    cout << "sigmaDistance = " << sigma << endl;

    //----------------------------------------------------//
    // Update
    double sigmaThresh = 1.0; // hyper parameter
    int numFoundKey = 0;
    for(auto itMatchPair = kptMatches.begin(); itMatchPair != kptMatches.end(); ++itMatchPair)
    {
        // calc distance
        double distance = cv::norm(kptsPrev[itMatchPair->queryIdx].pt - kptsPrev[itMatchPair->trainIdx].pt);
        double distanceDiff = abs(distance - eucliDistAve);

        // check the distance and Update keypoints and matches
        if( distanceDiff < sigma * sigmaThresh)
        {
            numFoundKey +=1;
            bBoxCurr.keypoints.push_back(kptsPrev[itMatchPair->queryIdx]);
            bBoxCurr.kptMatches.push_back(*itMatchPair);
        }
    }

    cout << "numFoundKey = " << numFoundKey << endl;
    cout << "------------------- Task-3 <end> ----------------------------" << endl;
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // time between two measurements in seconds
    double deltaTime = 1.0 / frameRate;

    int calcType = 2;

    if(calcType == 1)
    {
        // find closest distance to Lidar points
        double minXPrev = 1.0e9;
        double minXCurr = 1.0e9;
        //
        cout << "lidar point num <prev> = " << lidarPointsPrev.size() << endl;
        //
        for(auto itPrev = lidarPointsPrev.begin(); itPrev != lidarPointsPrev.end(); ++itPrev)
        {
            minXPrev = (minXPrev > itPrev->x) ? itPrev->x : minXPrev;
        }
        //
        for(auto itCurr = lidarPointsCurr.begin(); itCurr != lidarPointsCurr.end(); ++itCurr)
        {
            minXCurr = (minXCurr > itCurr->x) ? itCurr->x : minXCurr;
        }

        // Compute TTC from both measurements
        double deltaX = minXPrev - minXCurr;
        double deltaV = deltaX / deltaTime;

        if (abs(deltaX) > 0.001)
        {
            TTC = minXCurr / deltaV;
        }
        else
        {
            // avoid "divide by zero"
            TTC = 100.0;
        }
    }
    else
    {
        double aveXPrev = 0.0;
        double aveXCurr = 0.0;
        int numXPrev = lidarPointsPrev.size();
        int numXCurr = lidarPointsCurr.size();

        for(auto itPrev = lidarPointsPrev.begin(); itPrev != lidarPointsPrev.end(); ++itPrev)
        {
            aveXPrev += itPrev->x;
        }
        //
        for(auto itCurr = lidarPointsCurr.begin(); itCurr != lidarPointsCurr.end(); ++itCurr)
        {
            aveXCurr += itCurr->x;
        }

        aveXPrev /= (double)numXPrev;
        aveXCurr /= (double)numXCurr;

        // Compute TTC from both measurements
        double deltaX = abs(aveXPrev - aveXCurr);
        double deltaV = deltaX / deltaTime;

        if (abs(deltaX) > 0.001)
        {
            TTC = aveXPrev / deltaV;
        }
        else
        {
            // avoid "divide by zero"
            TTC = 100.0;
        }
    }
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // Get num of bounding boxes
    int numPrevBBoxes = prevFrame.boundingBoxes.size();
    int numCurrBBoxes = currFrame.boundingBoxes.size();
    cv::Mat matchCountTable = cv::Mat::zeros(numPrevBBoxes, numCurrBBoxes, CV_32S);

    cout << "------------------- Task-1 <start> ----------------------------" << endl;
    // Loop-1 for all matching pairs
    for (auto itMatch = matches.begin(); itMatch != matches.end(); ++itMatch)
    {
        // Get keypoints
        int prevKeyID = itMatch->queryIdx;
        int currKeyID = itMatch->trainIdx;
        auto prevKeyPoint = prevFrame.keypoints[prevKeyID].pt;
        auto currKeyPoint = currFrame.keypoints[currKeyID].pt;

        // Loop for all current bounding boxes
        for(int currBBoxID = 0; currBBoxID < numCurrBBoxes; ++currBBoxID)
        {
            // get a bounding box which contain keypoint
            BoundingBox currBBox = currFrame.boundingBoxes[currBBoxID];

            if(currBBox.roi.contains(currKeyPoint))
            {
                for(int prevBBoxID = 0; prevBBoxID < numPrevBBoxes; ++prevBBoxID)
                {
                    // get a bounding box which contain keypoint
                    BoundingBox prevBBox = prevFrame.boundingBoxes[prevBBoxID];

                    if(prevBBox.roi.contains(prevKeyPoint))
                    {
                        // Update Counter
                        matchCountTable.at<int>(prevBBoxID, currBBoxID) += 1;
                    }
                }
            }
        }
    }// end : Loop-1 for all matching pairs

    // View Matching Table
    cout << "Mathcing Table : Row = Previous BBox ID / Col = Current BBox ID" << endl;
    for(int prevBBoxID = 0; prevBBoxID < numPrevBBoxes; ++prevBBoxID)
    {
        for(int currBBoxID = 0; currBBoxID < numCurrBBoxes; ++currBBoxID)
        {
            cout << matchCountTable.at<int>(prevBBoxID, currBBoxID) << " , ";
        }
        cout << endl;
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
    cout << "####### Map {Prev , Curr} ##########" << endl;
    for(auto it = bbBestMatches.begin(); it != bbBestMatches.end(); ++it)
    {
        cout << it->first << ", " << it->second << endl;
    }

    cout << "------------------- Task-1 <end> ----------------------------" << endl;
}
