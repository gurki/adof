////////////////////////////////////////////////////////////////////////////////
//
//  Practical Course: GPU Programming in Computer Vision
//
//  Technical University Munich, Computer Vision Group
//  Winter Semester 2013/2014, March 3 - April 4
//
////////////////////////////////////////////////////////////////////////////////
//
//  Group 05 - Final Project
//
//  Realtime Depth of Focus Adaption using Global Solutions
//  of Variational Methods with Convex Regularization 
//  for Stereo Disparity Estimation of Death
//
////////////////////////////////////////////////////////////////////////////////
//
//  Zorah Lähner, mail@zorah-laehner.de, p063
//  Tobias Gurdan, tobias@gurdan.de, p064
//  Nicolai Oswald, mail@nicolai-oswald.de, p065
//
////////////////////////////////////////////////////////////////////////////////


#include "../../common/core/aux.h"
#include "../../common/core/Logger.h"

#include "DisparityEstimator.h"


////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    Logger logger("main");
    printDeviceMemoryUsage();


    //  load images
    cv::Mat img1, img2;
    loadImagePair(argc, argv, img1, img2);

    string animFile = "";
    getParam("save", animFile, argc, argv);

    int maxIter = 5000;
    getParam("iter", maxIter, argc, argv);


    // initialize algorithm
    DisparityEstimator de;
    de.setImagePair(img1, img2);
    de.setParameters(argc, argv);
    de.initialize();

    // de.showHostInputImages();
    // cv::waitKey();


    //  run stereo disparity estimation algorithm of death
    Logger::setMaxLevel(2);

    int key = 0;
    cv::Mat disp;
    cv::VideoWriter writer;

    for (int iter = 1; iter < maxIter; iter++)
    {
        stringstream str; str << "iter: " << iter;
        logger.push(str.str());

        de.update();
        
        logger.pop();

        if (animFile != "")
        {
            if (!writer.isOpened()) {
                const int fourcc = CV_FOURCC('M','J','P','G');
                writer.open(animFile, fourcc, 30.0f, img1.size(), false);
            }

            de.getDeviceDisparity(disp);
            writer.write(disp);
            cv::imshow("disparity", disp); 

        } 
        else {
            de.showDeviceDisparity();
        }

        key = cv::waitKey(10);

        if (key == 27) {
            break;
        }
    }

    return EXIT_SUCCESS;
}





