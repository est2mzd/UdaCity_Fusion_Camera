//
// Created by takuya on 22/04/26.
//
#ifndef CAMERA_FUSION_MYUTILITY_H
#define CAMERA_FUSION_MYUTILITY_H

#define DEBUG_MODE 0
#define VIEW_IMAGE_MODE 0
#define FIG_SAVE_MODE 0
#define TTC_OUTLIER 10000
#define SIMULATION_TYPE 2 // 1=FP.5 , 2=FP.6

#include <vector>
#include <stdio.h>
#include <iostream>
#include <cmath>
#include <algorithm>

////------------------------------------------------------------------------------------------------------------------
void CvTimeCount(double& t, bool FlagStart);

////------------------------------------------------------------------------------------------------------------------
template<class T>
T GetMedian(std::vector<T>& inputData)
{
    std::sort(inputData.begin(), inputData.end());
    int  dataSize   = inputData.size();
    int  middleId   = floor(dataSize / 2.0);
    bool isEvenSize = (dataSize % 2 == 0) ? true : false;
    double median   = (isEvenSize) ? 0.5*(inputData[middleId - 1] + inputData[middleId]) : inputData[middleId];

    if (inputData.size()==0)
    {
        median = 1.0;
    }

    return median;
}

////------------------------------------------------------------------------------------------------------------------
template<class T>
void CalcAverageSigma(std::vector<T>& inputData, T& average, T& sigma)
{
    int dataNum = inputData.size();
    average = 0.0;
    sigma   = 0.0;

    //----------------- Calc Average --------------------//
    for(int i=0; i<dataNum; ++i)
    {
        average += inputData[i];
    }

    if(dataNum > 0)
    {
        average = average / dataNum;
    }

    //----------------- Calc Sigma --------------------//
    // Calculate Variance(=sigma)
    for(int i=0; i<dataNum; ++i)
    {
        sigma += pow((inputData[i] - average),2);
    }
    sigma = pow( (sigma/dataNum), 0.5);
}

////------------------------------------------------------------------------------------------------------------------
template<class T>
void GetVectorMinMax(std::vector<T>& inputData, T& min, T& max)
{
    std::sort(inputData, inputData.begin(), inputData.end());
    if(inputData.size() > 0)
    {
        min = inputData[0];
        max = inputData[inputData.size()-1];
    }
}

////------------------------------------------------------------------------------------------------------------------
class ReportData
{
public:
    std::vector<double> Lidar_TopView_XMin;
    std::vector<double> Lidar_BBox_XMin;
    std::vector<double> Lidar_BBox_DeltaV;
    std::vector<double> Lidar_TTC;
    std::vector<double> Camera_TTC;
    std::vector<double> Diff_Percent_TTC;
    //
    std::string nameDetector;
    std::string nameDescriptor;
    std::string exportDirectory;
    std::string fileNameLog_Type01;
    std::string fileNameLog_Type02;
    //
    double FrameRate;
    ReportData();
    void exportReport_Type01(bool bAppendToFile);
    void exportReport_Type02(bool bAppendToFile);

private:
    std::string filePathLog_Type01;
    std::string filePathLog_Type02;
    void CreateFileNameLog();
    void CreateExportDir();
};

////------------------------------------------------------------------------------------------------------------------


















#endif //CAMERA_FUSION_MYUTILITY_H
