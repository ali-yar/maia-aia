#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/photo/photo.hpp>

class Utils
{
public:
    static std::string dataset_path1;
    static std::string dataset_path2;
    static std::string dataset_path3;

    Utils();
    static std::vector<cv::Mat> getImagesInFolder(std::string , std::string = ".tif", bool = false);
    static bool isDirectory(std::string );
    static double accuracy(std::vector<cv::Mat> & , std::vector<cv::Mat> & , std::vector<cv::Mat> & , std::vector<cv::Mat> * = 0);
    static int imdepth(int );
    static cv::vector<cv::Mat> createTiltedStructuringElements(int , int , int );
    static void imshow(std::string title, const cv::Mat& im, int waitKey = -1);
    static void grow(const cv::Mat& image, cv::Mat dst, int x, int y, int minI, int maxI);
    static int getTriangleAutoThreshold(const cv::Mat& image);
    static std::vector<int> histogram(const cv::Mat & image, int bins = -1);
    static void skeletonize(const cv::Mat & image, cv::Mat& skeleton);
    static void gammaCorrect(const cv::Mat & image, cv::Mat& dst, float gamma = 1.);
    static void contrastCorrect(const cv::Mat & image, cv::Mat& dst, double alpha, int beta);
    static void contrastAutoAdjust(const cv::Mat &src, cv::Mat &dst, float clipHistPercent=0);
    static cv::Point2i findOpticDisk(const cv::Mat &image, int searchSize = 80);
};

#endif // UTILS_H
