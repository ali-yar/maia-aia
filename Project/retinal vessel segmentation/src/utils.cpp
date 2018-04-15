#include "utils.h"

std::string Utils::dataset_path = "C:/Users/hp4540/Documents/MAIA Courses/UNICAS/Advanced Image Analysis/AIA-Retinal-Vessel-Segmentation/datasets/DRIVE/";

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
