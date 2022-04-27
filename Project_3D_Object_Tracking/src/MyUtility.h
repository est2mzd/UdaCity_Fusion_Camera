//
// Created by takuya on 22/04/26.
//
#ifndef CAMERA_FUSION_MYUTILITY_H
#define CAMERA_FUSION_MYUTILITY_H

#define DEBUG_MODE 0

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
    return isEvenSize ? 0.5*(inputData[middleId - 1] + inputData[middleId]) : inputData[middleId];
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
    int numImage;
    int numKeypoints;
    int numMatchedKeypoints;
    //
    double cpuTimeDetect;
    double cpuTimeDescript;
    double cpuTimeCalcTTCCamera;
    double cpuTimeCalcTTCLidar;
    //
    double TTCLidar;
    double TTCCamera;
    //
    std::string nameDetector;
    std::string nameDescriptor;
    std::string exportDirectory;
    std::string fileNameLog;
    ReportData();
    void exportReport(bool bAppendToFile);

private:
    std::string filePathLog;
    void CreateFileNameLog();
    void CreateExportDir();
    void CalcAverage();
};

////------------------------------------------------------------------------------------------------------------------


















#endif //CAMERA_FUSION_MYUTILITY_H
