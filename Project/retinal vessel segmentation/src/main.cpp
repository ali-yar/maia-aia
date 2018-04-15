#include <iostream>

#include <utils.h>
#include <vesseldetector.h>

int main()
{
    // Load dataset images
    std::vector<cv::Mat> images = Utils::getImagesInFolder(Utils::dataset_path + "images", ".tif");
    std::vector<cv::Mat> truths = Utils::getImagesInFolder(Utils::dataset_path + "groundtruths", ".tif", true);
    std::vector<cv::Mat> masks  = Utils::getImagesInFolder(Utils::dataset_path + "masks", ".tif", true);

    // Run segmentation process
    VesselDetector vd(images, masks, truths);
    vd();

    // Obtain segmented images
    std::vector<cv::Mat> results = vd.getSegmentedList();

    // Compute accuracy
    std::vector <cv::Mat> visual_results;
    double ACC = Utils::accuracy(results, truths, masks, &visual_results);
    printf("Accuracy (dummy segmentation) = %.2f%%\n", ACC*100);

    // Display visual comparaison results
    for(auto & v : visual_results) {
        cv::imshow("Visual result", v);
        cv::waitKey(0);
    }

    return 0;
}
