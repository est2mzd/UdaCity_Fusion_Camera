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
    nameDetector        = "no_name";
    nameDescriptor      = "no_name";
    fileNameLog_Type01  = "no_name";
    fileNameLog_Type02  = "no_name";
    exportDirectory     = "../results";
};

void ReportData::CreateFileNameLog()
{
    filePathLog_Type01 = exportDirectory + "/" + fileNameLog_Type01;
    filePathLog_Type02 = exportDirectory + "/" + fileNameLog_Type02;
}

void ReportData::CreateExportDir()
{
    const char* dir = exportDirectory.c_str();
    mkdir(dir, S_IRWXU); // mode = read/write is OK
}

void ReportData::exportReport_Type01(bool bAppendToFile)
{
    CreateExportDir();
    CreateFileNameLog();
    string spacer = ",";
    std::ofstream file;
    //
    if(!bAppendToFile)
    {
        file.open(filePathLog_Type01, std::ios::out);
        file << "Detector" << spacer << "Descriptor" << spacer << "ID"
             << spacer << "Lidar_TopView_Xmin"   << spacer << "Lidar_BBox_Xmin"
             << spacer << "Lidar_TopView_DeltaV" << spacer << "Lidar_BBox_DeltaV"
             << spacer << "TTC_Lidar_TopView" << spacer << "TTC_Lidar_BBox" << spacer << "TTC_Camera"
             << spacer << "TTC_Diff1[%]" << spacer << "TTC_Diff2[%]"
             << std::endl;
    }
    else
    {
        file.open(filePathLog_Type01, std::ios::app);
    }
    //
    for(int i=0; i< Lidar_TTC.size(); ++i)
    {
        double Lidar_TopView_DeltaV;
        double deltaT = 1 / FrameRate;
        if(i==0)
        {
            Lidar_TopView_DeltaV = abs(Lidar_TopView_XMin[i] - Lidar_TopView_XMin[i+1])  / deltaT;
        }
        else
        {
            Lidar_TopView_DeltaV = abs(Lidar_TopView_XMin[i] - Lidar_TopView_XMin[i-1])  / deltaT;
        }
        double Lidar_TTC_TopView = Lidar_TopView_XMin[i] / Lidar_TopView_DeltaV;
        //
        if(abs(Lidar_TTC_TopView) > 100)
        {
            Lidar_TTC_TopView = Lidar_TTC_TopView / abs(Lidar_TTC_TopView) * 100.0;
        }


        double Diff_TTC_Lidars = abs(Lidar_TTC_TopView - Lidar_TTC[i]) / Lidar_TTC[i] * 100;
        //
        char ToFile[200];
        sprintf(ToFile, "%s,%s,%02d,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f",
                nameDetector.c_str(), nameDescriptor.c_str(), i,
                Lidar_TopView_XMin[i], Lidar_BBox_XMin[i],
                Lidar_TopView_DeltaV,  Lidar_BBox_DeltaV[i],
                Lidar_TTC_TopView,     Lidar_TTC[i],  Camera_TTC[i],
                Diff_TTC_Lidars, Diff_Percent_TTC[i]);
        //
        file << ToFile << std::endl;
    }
    //
    file.close();
}

void ReportData::exportReport_Type02(bool bAppendToFile)
{
    CreateExportDir();
    CreateFileNameLog();
    string spacer = ",";
    std::ofstream file;
    //
    if(!bAppendToFile)
    {
        file.open(filePathLog_Type02, std::ios::out);
        file << "Detector" << "_" << "Descriptor"
             << spacer << "Num_Lidar_Res"     << spacer << "Num_Camera_Res"
             << spacer << "Num_Outlier_Lidar" << spacer << "Num_Outlier_Camera"
             << spacer << "TTC_Ave_Lidar"     << spacer << "TTC_Ave_Camera" << spacer << "TTC_Ave_Diff"
             << spacer << "TTC_Sigma_Lidar"   << spacer << "TTC_Sigma_Camera"
             << std::endl;
    }
    else
    {
        file.open(filePathLog_Type02, std::ios::app);
    }
    //
    int Num_Lidar_Res  = Lidar_TTC.size();
    int Num_Camera_Res = Camera_TTC.size();
    //
    double TTC_Ave_Lidar=0.0, TTC_Ave_Camera=0.0;
    double TTC_Sigma_Lidar = 0.0, TTC_Sigma_Camera = 0.0;
    //
    vector<double> TTCs_Lidar_wo_Outlier;
    vector<double> TTCs_Camera_wo_Outlier;
    int    Num_Outlier_Lidar = 0, Num_Outlier_Camera = 0;
    // ------------------------------------------------//
    // Calculation for Lidar
    // ------------------------------------------------//
    for(int i=0; i< Num_Lidar_Res; ++i)
    {
        if(Lidar_TTC[i] < TTC_OUTLIER )
        {
            TTCs_Lidar_wo_Outlier.push_back(Lidar_TTC[i]);
        }
    }
    Num_Outlier_Lidar = Num_Lidar_Res - TTCs_Lidar_wo_Outlier.size();
    // ------------------------------------------------//
    for(int i=0; i< Num_Camera_Res; ++i)
    {
        if(Camera_TTC[i] < TTC_OUTLIER )
        {
            TTCs_Camera_wo_Outlier.push_back(Camera_TTC[i]);
        }
    }
    Num_Outlier_Camera = Num_Camera_Res - TTCs_Camera_wo_Outlier.size();
    // ------------------------------------------------//

    CalcAverageSigma(TTCs_Lidar_wo_Outlier, TTC_Ave_Lidar, TTC_Sigma_Lidar);
    CalcAverageSigma(TTCs_Camera_wo_Outlier, TTC_Ave_Camera, TTC_Sigma_Camera);


    // ------------------------------------------------//
    char ToFile[200];
    sprintf(ToFile, "%s,%02d,%02d,%02d,%02d,%6.2f,%6.2f,%6.2f,%6.2f,%6.2f",
            (nameDetector + "_" + nameDescriptor).c_str(),
            Num_Lidar_Res, Num_Camera_Res,
            Num_Outlier_Lidar, Num_Outlier_Camera,
            TTC_Ave_Lidar, TTC_Ave_Camera, (TTC_Ave_Camera- TTC_Ave_Lidar)/TTC_Ave_Lidar*100,
            TTC_Sigma_Lidar, TTC_Sigma_Camera);
    file << ToFile << std::endl;
    file.close();
}
