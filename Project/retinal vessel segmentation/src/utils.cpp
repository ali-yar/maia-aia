#include "utils.h"

std::string Utils::dataset_path1 = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Advanced Image Analysis/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/";
std::string Utils::dataset_path2 = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Advanced Image Analysis/AIA-Retinal-Vessel-Segmentation/datasets/CHASEDB1/";
std::string Utils::dataset_path3 = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Advanced Image Analysis/AIA-Retinal-Vessel-Segmentation/datasets/STARE/";


Utils::Utils()
{

}

std::vector<cv::Mat> Utils::getImagesInFolder(std::string folder, std::string ext, bool force_gray)
{
    // check folders exist
    if(!isDirectory(folder))
        throw printf("in getImagesInFolder(): cannot open folder at \"%s\"", folder.c_str());

    // get all files within folder
    std::vector < std::string > files;
    cv::glob(folder, files);

    // open files that contains 'ext'
    std::vector < cv::Mat > images;
    for(auto & f : files)
    {
        if(f.find(ext) == std::string::npos)
            continue;

        images.push_back(cv::imread(f, force_gray ? CV_LOAD_IMAGE_GRAYSCALE : CV_LOAD_IMAGE_UNCHANGED));
    }

    return images;
}

bool Utils::isDirectory(std::string path) {
    struct stat s;
    if( stat(path.c_str(),&s) == 0 )
    {
        if( s.st_mode & S_IFDIR )
            return true;
        else if( s.st_mode & S_IFREG )
            return false;
        else return false;
    }
    else return false;
}


// Accuracy (ACC) is defined as:
// ACC = (True positives + True negatives)/(number of samples)
// i.e., as the ratio between the number of correctly classified samples (in our case, pixels)
// and the total number of samples (pixels)
double Utils::accuracy(
        std::vector <cv::Mat> & segmented_images,		// (INPUT)  segmentation results we want to evaluate (1 or more images, treated as binary)
        std::vector <cv::Mat> & groundtruth_images,     // (INPUT)  reference/manual/groundtruth segmentation images
        std::vector <cv::Mat> & mask_images,			// (INPUT)  mask images to restrict the performance evaluation within a certain region
        std::vector <cv::Mat> * visual_results		// (OUTPUT) (optional) false color images displaying the comparison between automated segmentation results and groundtruth
        //          True positives = blue, True negatives = gray, False positives = yellow, False negatives = red
        )
{
    // (a lot of) checks (to avoid undesired crashes of the application!)
    if(segmented_images.empty())
        throw ("in accuracy(): the set of segmented images is empty");
    if(groundtruth_images.size() != segmented_images.size())
        throw "in accuracy(): the number of groundtruth images (%d) is different than the number of segmented images (%d)", groundtruth_images.size(), segmented_images.size();
    if(mask_images.size() != segmented_images.size())
        throw "in accuracy(): the number of mask images (%d) is different than the number of segmented images (%d)", mask_images.size(), segmented_images.size();
    for(size_t i=0; i<segmented_images.size(); i++)
    {
        if(segmented_images[i].depth() != CV_8U || segmented_images[i].channels() != 1)
            throw printf("in accuracy(): segmented image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, imdepth(segmented_images[i].depth()), segmented_images[i].channels());
        if(!segmented_images[i].data)
            throw printf("in accuracy(): segmented image #%d has invalid data", i);
        if(groundtruth_images[i].depth() != CV_8U || groundtruth_images[i].channels() != 1)
            throw printf("in accuracy(): groundtruth image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, imdepth(groundtruth_images[i].depth()), groundtruth_images[i].channels());
        if(!groundtruth_images[i].data)
            throw printf("in accuracy(): groundtruth image #%d has invalid data", i);
        if(mask_images[i].depth() != CV_8U || mask_images[i].channels() != 1)
            throw printf("in accuracy(): mask image #%d is not a 8-bit single channel images (bitdepth = %d, nchannels = %d)", i, imdepth(mask_images[i].depth()), mask_images[i].channels());
        if(!mask_images[i].data)
            throw printf("in accuracy(): mask image #%d has invalid data", i);
        if(segmented_images[i].rows != groundtruth_images[i].rows || segmented_images[i].cols != groundtruth_images[i].cols)
            throw printf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and groundtruth (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, groundtruth_images[i].rows, groundtruth_images[i].cols);
        if(segmented_images[i].rows != mask_images[i].rows || segmented_images[i].cols != mask_images[i].cols)
            throw printf("in accuracy(): image size mismatch between %d-th segmented (%d x %d) and mask (%d x %d) images", i, segmented_images[i].rows, segmented_images[i].cols, mask_images[i].rows, mask_images[i].cols);
    }

    // clear previously computed visual results if any
    if(visual_results)
        visual_results->clear();

    // True positives (TP), True negatives (TN), and total number N of pixels are all we need
    double TP = 0, TN = 0, N = 0;

    // examine one image at the time
    for(size_t i=0; i<segmented_images.size(); i++)
    {
        // the caller did not ask to calculate visual results
        // accuracy calculation is easier...
        if(visual_results == 0)
        {
            for(int y=0; y<segmented_images[i].rows; y++)
            {
                uchar* segData = segmented_images[i].ptr<uchar>(y);
                uchar* gndData = groundtruth_images[i].ptr<uchar>(y);
                uchar* mskData = mask_images[i].ptr<uchar>(y);

                for(int x=0; x<segmented_images[i].cols; x++)
                {
                    if(mskData[x])
                    {
                        N++;		// found a new sample within the mask

                        if(segData[x] && gndData[x])
                            TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)
                        else if(!segData[x] && !gndData[x])
                            TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)
                    }
                }
            }
        }
        else
        {
            // prepare visual result (3-channel BGR image initialized to black = (0,0,0) )
            cv::Mat visualResult = cv::Mat(segmented_images[i].size(), CV_8UC3, cv::Scalar(0,0,0));

            for(int y=0; y<segmented_images[i].rows; y++)
            {
                uchar* segData = segmented_images[i].ptr<uchar>(y);
                uchar* gndData = groundtruth_images[i].ptr<uchar>(y);
                uchar* mskData = mask_images[i].ptr<uchar>(y);
                uchar* visData = visualResult.ptr<uchar>(y);

                for(int x=0; x<segmented_images[i].cols; x++)
                {
                    if(mskData[x])
                    {
                        N++;		// found a new sample within the mask

                        if(segData[x] && gndData[x])
                        {
                            TP++;	// found a true positive: segmentation result and groundtruth match (both are positive)

                            // mark with blue
                            visData[3*x + 0 ] = 255;
                            visData[3*x + 1 ] = 0;
                            visData[3*x + 2 ] = 0;
                        }
                        else if(!segData[x] && !gndData[x])
                        {
                            TN++;	// found a true negative: segmentation result and groundtruth match (both are negative)

                            // mark with gray
                            visData[3*x + 0 ] = 128;
                            visData[3*x + 1 ] = 128;
                            visData[3*x + 2 ] = 128;
                        }
                        else if(segData[x] && !gndData[x])
                        {
                            // found a false positive

                            // mark with yellow
                            visData[3*x + 0 ] = 0;
                            visData[3*x + 1 ] = 255;
                            visData[3*x + 2 ] = 255;
                        }
                        else
                        {
                            // found a false positive

                            // mark with red
                            visData[3*x + 0 ] = 0;
                            visData[3*x + 1 ] = 0;
                            visData[3*x + 2 ] = 255;
                        }
                    }
                }
            }

            visual_results->push_back(visualResult);
        }
    }

    return (TP + TN) / N;	// according to the definition of Accuracy
}

int Utils::imdepth(int ocv_depth)
{
    switch(ocv_depth)
    {
        case CV_8U:  return 8;
        case CV_8S:  return 8;
        case CV_16U: return 16;
        case CV_16S: return 16;
        case CV_32S: return 32;
        case CV_32F: return 32;
        case CV_64F: return 64;
        default:     return -1;
    }
}

// create 'n' rectangular Structuring Elements (SEs) at different orientations spanning the whole 360°
cv::vector<cv::Mat>						// vector of 'width' x 'width' uint8 binary images with non-black pixels being the SE
Utils::createTiltedStructuringElements(
        int width,							// SE width (must be odd)
        int height,							// SE height (must be odd)
        int n)								// number of SEs
{
    // check preconditions
    if( width%2 == 0 )
        throw printf("Structuring element width (%d) is not odd", width);
    if( height%2 == 0 )
        throw printf("Structuring element height (%d) is not odd", height);

    // draw base SE along x-axis
    cv::Mat base(width, width, CV_8U, cv::Scalar(0));
    // workaround: cv::line does not work properly when thickness > 1. So we draw line by line.
    for(int k=width/2-height/2; k<=width/2+height/2; k++)
        cv::line(base, cv::Point(0,k), cv::Point(width, k), cv::Scalar(255));

    // compute rotated SEs
    cv::vector <cv::Mat> SEs;
    SEs.push_back(base);
    double angle_step = 180.0/n;
    for(int k=1; k<n; k++)
    {
        cv::Mat SE;
        cv::warpAffine(base, SE, cv::getRotationMatrix2D(cv::Point2f(base.cols/2.0f, base.rows/2.0f), k*angle_step, 1.0), cv::Size(width, width), CV_INTER_NN);
        SEs.push_back(SE);
    }

    return SEs;
}

void Utils::imshow(std::string title, const cv::Mat &im, int waitKey)
{
    if (waitKey < 0)
        return;

    cv::imshow(title,im);
    cv::waitKey(waitKey);
}

// Region growing
void Utils::grow(const cv::Mat& image, cv::Mat dst, int y, int x, int minI, int maxI) {
    dst.at<uchar>(y,x) = 255; // mark as region
    //    cv::imshow("segmented", dst); cv::waitKey(1);

    int r, c;

    // parse 8 neighboors
    for (int i = -1; i<2; i++) {
        r = y - i;
        if (r<0 || r>image.rows - 1)
            continue;
        for (int j = -1; j<2; j++) {
            c = x - j;
            if (c<0 || c>image.cols - 1)
                continue;
            if (r == y && c == x)
                continue;
            if (dst.at<uchar>(r, c) == 0 && image.at<uchar>(r, c)>minI && image.at<uchar>(r, c)<maxI) { //pixel not visited and has close intensity
                grow(image, dst, r, c, minI, maxI);
            }
            else if (dst.at<uchar>(r, c) == 0) {
                dst.at<uchar>(r, c) = 100;
            }
        }
    }

    cv::threshold(dst,dst,101,255,CV_THRESH_BINARY);
}

std::vector<int> Utils::histogram(const cv::Mat & image, int bins)
{
    // checks
    if(!image.data)
        throw printf("in histogram(): invalid image");
    if(image.channels() != 1)
        throw printf("in histogram(): unsupported number of channels");

    // the number of gray levels
    int grayLevels  = static_cast<int>( std::pow(2,8) );

    // computing the number of bins
    bins = bins == -1 ? grayLevels : bins;

    // input-output parameters of cv::calcHist function
    int histSize[1]  = {bins};				// number of bins
    int channels[1]  = {0};					// only 1 channel used here
    float hranges[2] = {0.0f, static_cast<float>(grayLevels)};	// [min, max) pixel levels to take into account
    const float* ranges[1] = {hranges};		// [min, max) pixel levels for all the images (here only 1 image)
    cv::MatND histo;						// where the output histogram is stored

    // histogram computation
    cv::calcHist(&image, 1, channels, cv::Mat(), histo, 1, histSize, ranges);

    // conversion from MatND to vector<int>
    std::vector<int> hist;
    for(int i=0; i<bins; i++)
        hist.push_back(static_cast<int>(histo.at<float>(i)));

    return hist;
}

void Utils::skeletonize(const cv::Mat &image, cv::Mat& skeleton)
{
    skeleton.create(image.size(), CV_8U);
    skeleton.setTo(0);
    cv::Mat temp(image.size(), CV_8U);
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));

    bool done;
    do
    {
        cv::morphologyEx(image, temp, cv::MORPH_OPEN, element);
        cv::bitwise_not(temp, temp);
        cv::bitwise_and(image, temp, temp);
        cv::bitwise_or(skeleton, temp, skeleton);
        cv::erode(image, image, element);

        double max;
        cv::minMaxLoc(image, 0, &max);
        done = (max == 0);
    } while (!done);
}

void Utils::gammaCorrect(const cv::Mat &src, cv::Mat &dst, float gamma)
{
    unsigned char lut[256];
    for (int i = 0; i < 256; i++) {
        lut[i] = cv::saturate_cast<uchar>(pow((float)(i / 255.0), gamma) * 255.0f);
    }

    dst = src.clone();

    const int channels = dst.channels();
    if (channels == 1) {
        cv::MatIterator_<uchar> it = dst.begin<uchar>();
        cv::MatIterator_<uchar> end = dst.end<uchar>();
        for ( ; it != end; it++) {
            *it = lut[(*it)];
        }
    } else if (channels == 3) {
        cv::MatIterator_<cv::Vec3b> it = dst.begin<cv::Vec3b>();
        cv::MatIterator_<cv::Vec3b> end = dst.end<cv::Vec3b>();
        for ( ; it != end; it++) {
            (*it)[0] = lut[((*it)[0])];
            (*it)[1] = lut[((*it)[1])];
            (*it)[2] = lut[((*it)[2])];
        }
    }
}

void Utils::contrastCorrect(const cv::Mat &image, cv::Mat &dst, double alpha, int beta)
{
    // e.g. alpha = [1.0-3.0]    beta = [0-100]
    dst = cv::Mat::zeros(image.size(), image.type());
    for( int y = 0; y < image.rows; y++ ) {
        for( int x = 0; x < image.cols; x++ ) {
            for( int c = 0; c < 3; c++ ) {
                dst.at<cv::Vec3b>(y,x)[c] =
                        cv::saturate_cast<uchar>( alpha*( image.at<cv::Vec3b>(y,x)[c] ) + beta );
            }
        }
    }
}


int Utils::getTriangleAutoThreshold(const cv::Mat& image)
{
    std::vector<int> data = Utils::histogram(image);

    // find min and max
    int min = 0, dmax=0, max = 0, min2=0;
    for (int i = 0; i < data.size(); i++) {
        if (data[i]>0){
            min=i;
            break;
        }
    }
    if (min>0) min--; // line to the (p==0) point, not to data[min]

    // The Triangle algorithm cannot tell whether the data is skewed to one side or another.
    // This causes a problem as there are 2 possible thresholds between the max and the 2 extremes
    // of the histogram.
    // Here I propose to find out to which side of the max point the data is furthest, and use that as
    //  the other extreme.
    for (int i = int(data.size()) - 1; i >0; i-- ) {
        if (data[i]>0){
            min2=i;
            break;
        }
    }
    if (min2<data.size() - 1) min2++; // line to the (p==0) point, not to data[min]

    for (int i =0; i < data.size(); i++) {
        if (data[i] >dmax) {
            max=i;
            dmax=data[i];
        }
    }
    // find which is the furthest side
    //IJ.log(""+min+" "+max+" "+min2);
    bool inverted = false;
    if ((max-min)<(min2-max)){
        // reverse the histogram
        //IJ.log("Reversing histogram.");
        inverted = true;
        int left  = 0;          // index of leftmost element
        int right = int(data.size()) - 1; // index of rightmost element
        while (left < right) {
            // exchange the left and right elements
            int temp = data[left];
            data[left]  = data[right];
            data[right] = temp;
            // move the bounds toward the center
            left++;
            right--;
        }
        min=int(data.size()) - 1-min2;
        max=int(data.size()) - 1-max;
    }

    if (min == max){
        //IJ.log("Triangle:  min == max.");
        return min;
    }

    // describe line by nx * x + ny * y - d = 0
    double nx, ny, d;
    // nx is just the max frequency as the other point has freq=0
    nx = data[max];   //-min; // data[min]; //  lowest value bmin = (p=0)% in the image
    ny = min - max;
    d = std::sqrt(nx * nx + ny * ny);
    nx /= d;
    ny /= d;
    d = nx * min + ny * data[min];

    // find split point
    int split = min;
    double splitDistance = 0;
    for (int i = min + 1; i <= max; i++) {
        double newDistance = nx * i + ny * data[i] - d;
        if (newDistance > splitDistance) {
            split = i;
            splitDistance = newDistance;
        }
    }
    split--;

    if (inverted) {
        // The histogram might be used for something else, so let's reverse it back
        int left  = 0;
        int right = int(data.size()) - 1;
        while (left < right) {
            int temp = data[left];
            data[left]  = data[right];
            data[right] = temp;
            left++;
            right--;
        }
        return (int(data.size()) - 1-split);
    }
    else
        return split;
}

cv::Point2i Utils::findOpticDisk(const cv::Mat &image, int searchSize)
{
    cv::Point2i center;

    cv::Mat im = image;

    int max = 0;

    for (int y=0; y<im.rows; y++){
        for (int x=0; x<im.rows; x++){
            if (y+searchSize>im.rows || x+searchSize>im.cols)
                continue;

            cv::Mat win = im.rowRange(y,y+searchSize).colRange(x,x+searchSize);

            int sum = cv::sum(win)[0];
            if (sum > max) {
                max = sum;
                center.x = x + (searchSize/2);
                center.y = y + (searchSize/2);
            }
        }
    }
    return center;
}

void sharpen(cv::Mat& img, cv::Mat& dst) {
    int k = 2;
    cv::Mat LK = (cv::Mat_<float>(3, 3) <<
                  -k, -k, -k,
                  -k, 1 + 8 * k, -k,
                  -k, -k, -k);

    cv::filter2D(img, dst, CV_8U, LK);

}


void Utils::contrastAutoAdjust(const cv::Mat &src, cv::Mat &dst, float clipHistPercent)
{
    // http://answers.opencv.org/question/75510/how-to-make-auto-adjustmentsbrightness-and-contrast-for-image-android-opencv-image-correction/
    CV_Assert(clipHistPercent >= 0);
    CV_Assert((src.type() == CV_8UC1) || (src.type() == CV_8UC3) || (src.type() == CV_8UC4));

    int histSize = 256;
    float alpha, beta;
    double minGray = 0, maxGray = 0;

    //to calculate grayscale histogram
    cv::Mat gray;
    if (src.type() == CV_8UC1) gray = src;
    else if (src.type() == CV_8UC3) cvtColor(src, gray, CV_BGR2GRAY);
    else if (src.type() == CV_8UC4) cvtColor(src, gray, CV_BGRA2GRAY);
    if (clipHistPercent == 0)
    {
        // keep full available range
        cv::minMaxLoc(gray, &minGray, &maxGray);
    }
    else
    {
        cv::Mat hist; //the grayscale histogram

        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;
        cv::calcHist(&gray, 1, 0, cv::Mat (), hist, 1, &histSize, &histRange, uniform, accumulate);

        // calculate cumulative distribution from the histogram
        std::vector<float> accumulator(histSize);
        accumulator[0] = hist.at<float>(0);
        for (int i = 1; i < histSize; i++)
        {
            accumulator[i] = accumulator[i - 1] + hist.at<float>(i);
        }

        // locate points that cuts at required value
        float max = accumulator.back();
        clipHistPercent *= (max / 100.0); //make percent as absolute
        clipHistPercent /= 2.0; // left and right wings
        // locate left cut
        minGray = 0;
        while (accumulator[minGray] < clipHistPercent)
            minGray++;

        // locate right cut
        maxGray = histSize - 1;
        while (accumulator[maxGray] >= (max - clipHistPercent))
            maxGray--;
    }

    // current range
    float inputRange = maxGray - minGray;

    alpha = (histSize - 1) / inputRange;   // alpha expands current range to histsize range
    beta = -minGray * alpha;             // beta shifts current range so that minGray will go to 0

    // Apply brightness and contrast normalization
    // convertTo operates with saurate_cast
    src.convertTo(dst, -1, alpha, beta);

    // restore alpha channel from source
    if (dst.type() == CV_8UC4)
    {
        int from_to[] = { 3, 3};
        cv::mixChannels(&src, 4, &dst,1, from_to, 1);
    }
    return;
}
