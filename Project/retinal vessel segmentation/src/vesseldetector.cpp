#include "vesseldetector.h"

VesselDetector::VesselDetector(std::vector<cv::Mat> images, std::vector<cv::Mat> masks, std::vector<cv::Mat> truths)
{
    this->images = images;
    this->masks = masks;
    this->truths = truths;
}

void VesselDetector::operator()()
{
    process();
}

std::vector<cv::Mat> VesselDetector::getSegmentedList()
{
    return segmented;
}

void VesselDetector::process()
{
    for(auto & image : images)
    {
        cv::Mat result;
        segment(image, result);
        segmented.push_back(result);
    }
}

void VesselDetector::segment(const cv::Mat &image, cv::Mat &result)
{
    cv::cvtColor(image, result, CV_BGR2GRAY);

    // then invert so that vessels are bright
    result = 255 - result;

    // apply thresholding
    cv::threshold(result, result, 160, 255, CV_THRESH_BINARY);
}
