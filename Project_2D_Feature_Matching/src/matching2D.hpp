#ifndef matching2D_hpp
#define matching2D_hpp

#include <stdio.h>
#include <iostream>
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


void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsShiTomasi(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis=false);
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis=false);
void descKeypoints(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, std::string descriptorType);
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType);
void visualizeResults(cv::Mat& img, std::string window_name, std::vector<cv::KeyPoint> keypoints);

//----------------------------------------------------------------------//
// https://www.modernmetalproduction.com/simple-circular-buffer-in-c/
template <class T>
class RingBuffer
{
public:
    RingBuffer(unsigned int buffer_size); // Constructor
    T* ReadLast();
    T* Read2ndLast();
    void Write(T input);
    int size();
    //void Print();
private:
    std::vector<T> Buffer;
    unsigned int ReadIndex;
    unsigned int WriteIndex;
    unsigned int WriteDataNum;
    void UpdateReadIndex();
};

template <class T>
RingBuffer<T>::RingBuffer(unsigned int buffer_size): Buffer(buffer_size) // Constructor
{
    ReadIndex    = buffer_size-1;
    WriteIndex   = 0;
    WriteDataNum = 0;
}

template <class T>
void RingBuffer<T>::Write(T input)
{
    Buffer[WriteIndex] = input;
    WriteIndex   +=1;
    WriteDataNum +=1;
    if(WriteIndex >= Buffer.size())
    {
        WriteIndex = 0;
    }
    UpdateReadIndex();
}

template <class T>
T* RingBuffer<T>::ReadLast()
{
    T* output = &Buffer[ReadIndex];
    return output;
}

template <class T>
T* RingBuffer<T>::Read2ndLast()
{
    T* output;

    if(ReadIndex == 0)
    {
        output = &Buffer[Buffer.size()-1];
    }
    else
    {
        output = &Buffer[ReadIndex-1];
    }

    return output;
}

template <class T>
void RingBuffer<T>::UpdateReadIndex()
{
    ReadIndex +=1;
    if(ReadIndex >= Buffer.size())
    {
        ReadIndex = 0;
    }
}

template <class T>
int RingBuffer<T>::size()
{
    if(WriteDataNum >= Buffer.size())
    {
        return Buffer.size();
    }
    else
    {
        return WriteDataNum;
    }
}

/*
template <class T>
void RingBuffer<T>::Print()
{
    std::cout << "---- RingBuffer ----" << std::endl;
    for(int i=0; i < Buffer.size(); ++i)
    {
        std::cout << "Buffer[" << i << "] = " << Buffer[i] << std::endl;
    }
    std::cout << "ReadIndex   = " << ReadIndex      << std::endl;
    std::cout << "Read2ndLast = " << *Read2ndLast() << std::endl;
    std::cout << "ReadLast    = " << *ReadLast()    << std::endl;
}
*/

//----------------------------------------------------------------------//

#endif /* matching2D_hpp */
