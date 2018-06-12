#include "vesseldetector.h"

VesselDetector::VesselDetector(std::vector<cv::Mat> images, std::vector<cv::Mat> masks, std::vector<cv::Mat> truths)
{
    this->images = images;
    this->masks = masks;
    this->truths = truths;

    // give default values
    numKernels = 12;
    medianSize = 21;
    threshOffset = 15;
    gamma = 1.0;
    nlMeansDnoising_h = 3;
    nlMeansDnoising_templateWinSize = 3;
    nlMeansDnoising_searchWinSize = 13;
    clahe_limit = 2.0;
    clahe_size = cv::Size(5,5);
    contourTrimSize = cv::Size(5,5);
}

void VesselDetector::setParams(int numKernels, int medianSize, int threshOffset, float gamma, float nlMeansDnoising_h,
                               int nlMeansDnoising_templateWinSize, int nlMeansDnoising_searchWinSize,
                               double clahe_limit, cv::Size clahe_size, cv::Size contourTrimSize)
{
    this->numKernels = numKernels;
    this->medianSize = medianSize;
    this->threshOffset = threshOffset;
    this->gamma = gamma;
    this->nlMeansDnoising_h = nlMeansDnoising_h;
    this->nlMeansDnoising_templateWinSize = nlMeansDnoising_templateWinSize;
    this->nlMeansDnoising_searchWinSize = nlMeansDnoising_searchWinSize;
    this->clahe_limit = clahe_limit;
    this->clahe_size = clahe_size;
    this->contourTrimSize = contourTrimSize;
}

void VesselDetector::operator()()
{
    process();
}

std::vector<cv::Mat> VesselDetector::getSegmentedList()
{
    return segmented;
}

cv::Point VesselDetector::getSeed(const cv::Mat &image, const cv::Mat &mask)
{
    cv::Mat im;
    cv::Mat mask2(im.size(),CV_8U,cv::Scalar(255));
    cv::Point pt;

    if (!mask.empty()) {
        cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(50,50));
        cv::erode(mask,mask2,se);
        image.copyTo(im,mask2);
    }

    cv::erode(im,im,cv::Mat(),cv::Point(-1,-1),2);

    cv::minMaxLoc(im,0,0,0,&pt,mask2);

    return pt;
}

cv::Rect VesselDetector::getEnclosedBox(const cv::Mat &image)
{
    cv::Mat im;
    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(170,170));
    cv::erode(image,im,se,cv::Point(-1,-1),1);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(im, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE, cv::Point(0,0));
    cv::Rect rect = cv::boundingRect(cv::Mat(contours[0]));

    return rect;
}

void VesselDetector::process()
{
    segmented.clear();

    for(int i=0; i<images.size(); i++)
    {
        cv::Mat result;
        segment(images[i], masks[i], result);
        segmented.push_back(result);
    }
}


void VesselDetector::segment(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
{
    int key = -1;

    // vars & objects declaration
    int triangleThresh;
    cv::Mat im, grey, bg, bgSubstracted, nlMeansDenoised, equalized, tophat, tophatSum, thresholded, structElem;
    std::vector<cv::Mat> imrgb, kernel;

    // apply mask on the image to ensure non FOV pixels are set to 0
    image.copyTo(im, mask);
    Utils::imshow("original", im, key);

    // apply gamma correction
    Utils::gammaCorrect(im,im,gamma);
    Utils::imshow("gamma correct", im, key);

    // get greyscale image from green channel
    cv::split(im,imrgb);
    grey = imrgb[1];
    grey = 255 - grey;
    Utils::imshow("grayscale with inverted intensities", grey, key);

    // substract background (median blur with big kernel) from image
    cv::medianBlur(grey,bg,medianSize);
    bgSubstracted = grey - bg;
    Utils::imshow("background substaction", bgSubstracted, key);

    // denoise with non-local means filtering
    cv::fastNlMeansDenoising(bgSubstracted,nlMeansDenoised,nlMeansDnoising_h,nlMeansDnoising_templateWinSize,nlMeansDnoising_searchWinSize);
    Utils::imshow("non-local means denoising", nlMeansDenoised, key);

    // enhance contrast with CLAHE
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_limit, clahe_size);
    clahe->apply(nlMeansDenoised, equalized);
    Utils::imshow("hist equalized", equalized, key);

    // apply multiple tophats with rotating structuring element
    tophatSum.create(image.size(), CV_32F);
    tophatSum.setTo(cv::Scalar(0));
    kernel = Utils::createTiltedStructuringElements(17,1,numKernels);
    for (int i=0; i<numKernels; i++) {
        cv::morphologyEx(equalized,tophat,CV_MOP_TOPHAT,kernel[i]);
        Utils::imshow("tophat", tophat, key);
        tophat.convertTo(tophat,CV_32F);
        tophatSum += tophat;
    }
    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
    Utils::imshow("total tophats", tophatSum, key);

    // threshold
    if (threshOffset > 0) {
        cv::threshold(tophatSum, thresholded, threshOffset, 255, CV_THRESH_BINARY);
    } else { // apply OTSU thresholding
        cv::threshold(tophatSum, thresholded, threshOffset, 255, CV_THRESH_BINARY | CV_THRESH_OTSU);
    }

    Utils::imshow("total tophats threholded", thresholded, key);

    // remove round edge
    structElem = cv::getStructuringElement(cv::MORPH_ELLIPSE,contourTrimSize);
    cv::erode(mask,mask,structElem);
    thresholded.copyTo(result,mask);
    Utils::imshow("edge trim", result, key);

    cv::destroyAllWindows();
}



























//void VesselDetector::segment(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
//{
//    cv::Mat im, grey, nlMeansDenoised, tophat;
//    cv::Mat tophatSum(image.size(), CV_32F);

//    std::vector<cv::Mat> imrgb, kernel;
//    int numKernels = 12;

//    // apply mask on the image
//    image.copyTo(im, mask);
//    cv::imshow("original", im); cv::waitKey(1);

//    // get greyscale image from green channel
//    cv::split(im,imrgb);
//    grey = imrgb[1];
//    grey = 255 - grey;
//    cv::imshow("grayscale with inverted intensities", grey); cv::waitKey(1);

//    // denoise with non-local means
//    cv::fastNlMeansDenoising(grey,nlMeansDenoised,2,7,21);
//    cv::imshow("non-local means denoising", nlMeansDenoised); cv::waitKey(1);

//    // apply multiple tophats with rotating structuring elements
//    tophatSum.setTo(cv::Scalar(0));
//    kernel = Utils::createTiltedStructuringElements(17,1,numKernels);
//    for (int i=0; i<numKernels; i++) {
//        cv::morphologyEx(nlMeansDenoised,tophat,CV_MOP_TOPHAT,kernel[i]);
//        cv::imshow("tophat", tophat);
//        cv::waitKey(20);
//        tophat.convertTo(tophat,CV_32F);
//        tophatSum += tophat;
//    }
//    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("total tophats", tophatSum); cv::waitKey(1);

//    cv::Mat thresh;
//    cv::threshold(tophatSum,thresh,20,255,CV_THRESH_BINARY);
//    cv::imshow("thresh direct", thresh); cv::waitKey(1);


//    // region growing
//    cv::Point seed = getSeed(tophatSum,mask);
//    cv::Mat segmented(tophatSum.size(), CV_8U, cv::Scalar(0));
//    Utils::grow(tophatSum,segmented,seed.y,seed.x,11,255);
//    cv::imshow("segmented", segmented); cv::waitKey(1);


//    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5));
//    cv::erode(mask,mask,se);
//    segmented.copyTo(result,mask);
//    cv::imshow("edge trim", result); cv::waitKey(0);

//    cv::destroyAllWindows();

//    //  result = tophatSum;

//}

//void VesselDetector::segment2(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
//{
//    cv::Mat im, grey, nlMeansDenoised, tophat;
//    cv::Mat tophatSum(image.size(), CV_32F);

//    std::vector<cv::Mat> imrgb, kernel;
//    int numKernels = 19;

//    // apply mask on the image
//    image.copyTo(im, mask);
//    cv::imshow("original", im); cv::waitKey(1);

//    // get greyscale image from green channel
//    cv::split(im,imrgb);
//    grey = imrgb[1];
//    grey = 255 - grey;
//    cv::imshow("grayscale with inverted intensities", grey); cv::waitKey(1);

//    // denoise with non-local means
//    cv::fastNlMeansDenoising(grey,nlMeansDenoised,2,7,21);
//    cv::imshow("non-local means denoising", nlMeansDenoised); cv::waitKey(1);

//    cv::Mat equalized;
//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(5,5));
//    clahe->apply(nlMeansDenoised, equalized);
//    cv::imshow("hist equalized", equalized);
//    cv::waitKey(1);

//    // apply multiple tophats with rotating structuring elements
//    tophatSum.setTo(cv::Scalar(0));
//    kernel = Utils::createTiltedStructuringElements(19,1,numKernels);
//    for (int i=0; i<numKernels; i++) {
//        cv::morphologyEx(equalized,tophat,CV_MOP_TOPHAT,kernel[i]);
//        cv::imshow("tophat", tophat);
//        cv::waitKey(20);
//        tophat.convertTo(tophat,CV_32F);
//        tophatSum += tophat;
//    }
//    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("total tophats", tophatSum); cv::waitKey(1);



//    // region growing
//    cv::Point seed = getSeed(tophatSum,mask);
//    cv::Mat segmented(tophatSum.size(), CV_8U, cv::Scalar(0));
//    Utils::grow(tophatSum,segmented,seed.y,seed.x,30,255);
//    cv::imshow("segmented", segmented); cv::waitKey(1);

//    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(5,5));
//    cv::erode(mask,mask,se);
//    segmented.copyTo(result,mask);
//    cv::imshow("edge trim", result); cv::waitKey(1);


//    cv::destroyAllWindows();

//}

//// 93.63%
//void VesselDetector::segment3(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
//{
//    cv::Mat im, grey, nlMeansDenoised, tophat;
//    cv::Mat tophatSum(image.size(), CV_32F);

//    std::vector<cv::Mat> imrgb, kernel;
//    int numKernels = 12;

//    // apply mask on the image
//    image.copyTo(im, mask);
//    cv::imshow("original", im); cv::waitKey(1);

//    // get greyscale image from green channel
//    cv::split(im,imrgb);
//    grey = imrgb[1];
//    grey = 255 - grey;
//    cv::imshow("grayscale with inverted intensities", grey); cv::waitKey(1);

//    // denoise with non-local means
//    cv::fastNlMeansDenoising(grey,nlMeansDenoised,2,7,21);
//    cv::imshow("non-local means denoising", nlMeansDenoised); cv::waitKey(1);

//    cv::Mat equalized;
//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(5,5));
//    clahe->apply(nlMeansDenoised, equalized);
//    cv::imshow("hist equalized", equalized);
//    cv::waitKey(1);

//    // apply multiple tophats with rotating structuring elements
//    tophatSum.setTo(cv::Scalar(0));
//    kernel = Utils::createTiltedStructuringElements(17,1,numKernels);
//    for (int i=0; i<numKernels; i++) {
//        cv::morphologyEx(equalized,tophat,CV_MOP_TOPHAT,kernel[i]);
//        cv::imshow("tophat", tophat);
//        cv::waitKey(20);
//        tophat.convertTo(tophat,CV_32F);
//        tophatSum += tophat;
//    }
//    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("total tophats", tophatSum); cv::waitKey(1);



//    // region growing
//    cv::Mat roi;

//    roi = tophatSum.rowRange(0,tophat.rows/2);
//    cv::Point seed = getSeed(roi,mask.rowRange(0,tophat.rows/2));
//    cv::Mat segmented1(tophatSum.size(), CV_8U, cv::Scalar(0));
//    Utils::grow(tophatSum,segmented1,seed.y,seed.x,30,255);
//    //cv::imshow("segmented roi 1", segmented1); cv::waitKey(1);

//    cv::Mat roi2 = tophatSum.rowRange(tophat.rows/2+70, tophat.rows-1);
//    seed = getSeed(roi2,mask.rowRange(tophat.rows/2+70, tophat.rows-1));
//    cv::Mat segmented2(tophatSum.size(), CV_8U, cv::Scalar(0));
//    Utils::grow(tophatSum,segmented2,seed.y+tophat.rows/2+70,seed.x,30,255);
//    //cv::imshow("segmented roi 2", segmented2); cv::waitKey(1);

//    cv::Mat segmented;
//    cv::bitwise_or(segmented1,segmented2,segmented);
//    cv::imshow("segmented", segmented); cv::waitKey(1);

//    // trim edge
//    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7,7));
//    cv::erode(mask,mask,se);
//    segmented.copyTo(result,mask);
//    cv::imshow("edge trim", result); cv::waitKey(0);


//    cv::destroyAllWindows();

//}

// Accuracy = 93.43%
//void VesselDetector::segment312(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
//{
//    cv::Mat im, grey, nlMeansDenoised, tophat;
//    cv::Mat tophatSum(image.size(), CV_32F);

//    std::vector<cv::Mat> imrgb, kernel;
//    int numKernels = 12;

//    // apply mask on the image
//    image.copyTo(im, mask);
//    cv::imshow("original", im); cv::waitKey(1);

//    Utils::gammaCorrect(im,im,0.5);
//    cv::imshow("gamma correct", im); cv::waitKey(1);

//    // get greyscale image from green channel
//    cv::split(im,imrgb);
//    grey = imrgb[1];
//    grey = 255 - grey;
//    cv::imshow("grayscale with inverted intensities", grey); cv::waitKey(1);

//    // denoise with non-local means
//    cv::fastNlMeansDenoising(grey,nlMeansDenoised,2,7,21);
//    cv::imshow("non-local means denoising", nlMeansDenoised); cv::waitKey(1);

//    cv::Mat equalized;
//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(5,5));
//    clahe->apply(nlMeansDenoised, equalized);
//    cv::imshow("hist equalized", equalized);
//    cv::waitKey(1);

//    // equalized = nlMeansDenoised;

//    // apply multiple tophats with rotating structuring elements
//    tophatSum.setTo(cv::Scalar(0));
//    kernel = Utils::createTiltedStructuringElements(17,1,numKernels);
//    for (int i=0; i<numKernels; i++) {
//        cv::morphologyEx(equalized,tophat,CV_MOP_TOPHAT,kernel[i]);
//        cv::imshow("tophat", tophat);
//        cv::waitKey(20);
//        tophat.convertTo(tophat,CV_32F);
//        tophatSum += tophat;
//    }
//    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("total tophats", tophatSum); cv::waitKey(1);

//    int thresh = Utils::getTriangleAutoThreshold(tophatSum);

//    cv::Mat thresholded;
//    cv::threshold(tophatSum, thresholded, thresh+15, 255, CV_THRESH_BINARY);
//    cv::imshow("total tophats threholded", thresholded); cv::waitKey(1);


//    // trim edge
//    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(7,7));
//    cv::erode(mask,mask,se);
//    thresholded.copyTo(result,mask);
//    cv::imshow("edge trim", result); cv::waitKey(1);


//    cv::destroyAllWindows();

//}


// Accuracy = 93.76%
//void VesselDetector::segment31(const cv::Mat &image, const cv::Mat &mask, cv::Mat &result)
//{
//    int key = 0;

//    float gamma = 1.;
//    float fast = 3;
//    int offset = 17;

////    float gamma = 0.3;
////    float fast = 3;
////    int offset = 10;

////    float gamma = 0.3;
////    float fast = 3;
////    int offset = 10;

//    cv::Mat im, grey, nlMeansDenoised, tophat;
//    cv::Mat tophatSum(image.size(), CV_32F);

//    std::vector<cv::Mat> imrgb, kernel;
//    int numKernels = 12;

//    // apply mask on the image
//    image.copyTo(im, mask);
//    cv::imshow("original", im); cv::waitKey(key);

//    Utils::gammaCorrect(im,im,gamma);
//    cv::imshow("gamma correct", im); cv::waitKey(key);

//    // get greyscale image from green channel
//    cv::split(im,imrgb);
//    grey = imrgb[1];
//    grey = 255 - grey;
//    cv::imshow("grayscale with inverted intensities", grey); cv::waitKey(key);

//    // denoise with non-local means
//    cv::fastNlMeansDenoising(grey,nlMeansDenoised,fast,7,21);
//    cv::imshow("non-local means denoising", nlMeansDenoised); cv::waitKey(key);

//    cv::Mat equalized;
//    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(5,5));
//    clahe->apply(nlMeansDenoised, equalized);
//    cv::imshow("hist equalized", equalized);
//    cv::waitKey(key);

//    // equalized = nlMeansDenoised;

//    // apply multiple tophats with rotating structuring elements
//    tophatSum.setTo(cv::Scalar(0));
//    kernel = Utils::createTiltedStructuringElements(17,1,numKernels);
//    for (int i=0; i<numKernels; i++) {
//        cv::morphologyEx(equalized,tophat,CV_MOP_TOPHAT,kernel[i]);
//        cv::imshow("tophat", tophat);
//        cv::waitKey(20);
//        tophat.convertTo(tophat,CV_32F);
//        tophatSum += tophat;
//    }
//    cv::normalize(tophatSum, tophatSum, 0, 255, cv::NORM_MINMAX, CV_8U);
//    cv::imshow("total tophats", tophatSum); cv::waitKey(key);

//    int thresh = Utils::getTriangleAutoThreshold(tophatSum);

//    cv::Mat thresholded;
//    cv::threshold(tophatSum, thresholded, thresh+offset, 255, CV_THRESH_BINARY);
//    cv::imshow("total tophats threholded", thresholded); cv::waitKey(key);


//    // trim edge
//    cv::Mat se = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(11,11));
//    cv::erode(mask,mask,se);
//    thresholded.copyTo(result,mask);
//    cv::imshow("edge trim", result); cv::waitKey(key);


//    cv::destroyAllWindows();

//}


