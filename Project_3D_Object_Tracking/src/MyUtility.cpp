//
// Created by takuya on 22/04/26.
//
#include "MyUtility.h"
#include <vector>
#include <opencv2/core.hpp>
#include <sys/stat.h>
#include <fstream>

using namespace std;

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
    //
    cpuTimeDetect       = 0.0;
    cpuTimeDescript     = 0.0;
    cpuTimeCalcTTCCamera = 0.0;
    cpuTimeCalcTTCLidar  = 0.0;
    //
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
