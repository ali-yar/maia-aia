#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

class Utils
{
public:
    static std::string dataset_path;

    Utils();
    static std::vector<cv::Mat> getImagesInFolder(std::string , std::string = ".tif", bool = false);
    static bool isDirectory(std::string );
    static double accuracy(std::vector<cv::Mat> & , std::vector<cv::Mat> & , std::vector<cv::Mat> & , std::vector<cv::Mat> * = 0);
    static int imdepth(int );
};

#endif // UTILS_H
