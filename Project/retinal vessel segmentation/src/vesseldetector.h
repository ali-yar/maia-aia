#ifndef VESSELDETECTOR_H
#define VESSELDETECTOR_H

#include <utils.h>

class VesselDetector
{
private:
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> masks;
    std::vector<cv::Mat> truths;
    std::vector<cv::Mat> segmented;

    void segment(const cv::Mat &image, cv::Mat &result);

public:
    VesselDetector(std::vector <cv::Mat> , std::vector <cv::Mat> , std::vector <cv::Mat> );
    void process();
    void operator ()();
    std::vector<cv::Mat> getSegmentedList();
};

#endif // VESSELDETECTOR_H
