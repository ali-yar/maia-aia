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

    int numKernels;
    int medianSize;
    int threshOffset;
    float gamma;
    float nlMeansDnoising_h;
    int nlMeansDnoising_templateWinSize;
    int nlMeansDnoising_searchWinSize;
    double clahe_limit;
    cv::Size clahe_size;
    cv::Size contourTrimSize;

public:
    VesselDetector(std::vector <cv::Mat> , std::vector <cv::Mat> , std::vector <cv::Mat> );
    void setParams(int numKernels, int medianSize, int threshOffset, float gamma, float nlMeansDnoising_h,
                   int nlMeansDnoising_templateWinSize, int nlMeansDnoising_searchWinSize,
                   double clahe_limit, cv::Size clahe_size, cv::Size contourTrimSize);
    void process();
    void operator ()();
    void segment(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result);
    std::vector<cv::Mat> getSegmentedList();
    cv::Point getSeed(const cv::Mat &image, const cv::Mat &mask = cv::Mat());
    cv::Rect getEnclosedBox(const cv::Mat &image);

};

#endif // VESSELDETECTOR_H
