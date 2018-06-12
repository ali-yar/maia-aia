#include <iostream>
#include <utils.h>
#include <vesseldetector.h>

int main()
{
    int dataset = 1;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> truths;
    std::vector<cv::Mat> masks;

    // Load dataset images
    if (dataset == 1) {         // DRIVE - manual: 94.07%  /  OTSU: 94.02%
        images = Utils::getImagesInFolder(Utils::dataset_path1 + "images", ".tif");
        truths = Utils::getImagesInFolder(Utils::dataset_path1 + "groundtruths", ".tif", true);
        masks  = Utils::getImagesInFolder(Utils::dataset_path1 + "masks", ".tif", true);
    } else if (dataset == 2) {  // CHASE - manual: 94.00%  /  OTSU: 94.03%
        images = Utils::getImagesInFolder(Utils::dataset_path2 + "images", ".jpg");
        truths = Utils::getImagesInFolder(Utils::dataset_path2 + "groundtruths", ".png", true);
        masks  = Utils::getImagesInFolder(Utils::dataset_path2 + "masks", ".png", true);
    } else if (dataset == 3) {  // STARE - manual: 94.71%  /  OTSU: 94.96%
        images = Utils::getImagesInFolder(Utils::dataset_path3 + "images", ".ppm");
        truths = Utils::getImagesInFolder(Utils::dataset_path3 + "groundtruths", ".ppm", true);
        masks = Utils::getImagesInFolder(Utils::dataset_path3 + "masks", ".png", true);
    }


    VesselDetector vd(images, masks, truths);

    // PARAMS with fixed thresholding
    //    vd.setParams(12, 21, 16, .8, 3, 3, 13, 2.0, cv::Size(11,11), cv::Size(1,1));
    //    vd.setParams(12, 49, 15, .3, 3, 5, 18, 2.0, cv::Size(11,11), cv::Size(1,1));
    //    vd.setParams(12, 49, 10, .2, 1, 2, 10, 2.0, cv::Size(5,5), cv::Size(11,11));

    // PARAMS with otsu thresholding
//      vd.setParams(12, 33, 0, .8, 3, 7, 7, 5.0, cv::Size(3,3), cv::Size(1,1));
//        vd.setParams(12, 49, 0, .3, 3, 5, 18, 2.0, cv::Size(11,11), cv::Size(1,1));
//    vd.setParams(12, 27, 0, .2, 1, 2, 12, 7.0, cv::Size(28,28), cv::Size(11,11));
        // 90



    // Run segmentation process
    vd();

    // Obtain segmented images
    std::vector<cv::Mat> results = vd.getSegmentedList();

    // Compute accuracy
    std::vector <cv::Mat> visual_results;
    double ACC = Utils::accuracy(results, truths, masks, &visual_results);
    printf("Accuracy = %.2f%%\n", ACC*100);

    //     Display visual comparaison results
    //        for(auto & v : visual_results) {
    //            cv::imshow("Visual result", v); cv::waitKey(0);
    //        }

    // Find optic disk
    //    for (int i=0; i<truths.size(); i++) {
    //        cv::Mat im = truths[i].clone();
    //        cv::Point2i center = Utils::findOpticDisk(im);
    //        cv::circle(im,center,40,150,3);
    //        cv::imshow("disk", im); cv::waitKey(0);
    //    }

    return 0;
}
