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

#include "LinearDiffusion.h"


int steps = 200;


void didPressMouseButton(int event, int x, int y, int, void* data)
{
    if (event == cv::EVENT_LBUTTONDOWN || event == cv::EVENT_MOUSEMOVE) 
    {
        Logger logger("didPressMouseButton");

        if (!data) {
            logger << "no lineardiffusion object received"; logger.eol();
            logger.pop(false);
        }

        LinearDiffusion* ld = (LinearDiffusion*)data;

        ld->initialize();
        ld->setFocus(x, y);

        cudaStartTimer();

        for (int i = 0; i < steps; ++i) {
            ld->adaptiveUpdate();
            // ld->intensityUpdate();
        }

        cout << "dt: " << cudaStopTimer() / steps << " ms" << endl;

        ld->showDiffusedImage("output");
    }
}


int main(int argc, char* argv[])
{
    Logger logger("main");
    Logger::setMaxLevel(1);

    printDeviceMemoryUsage();

    int nlayer = 141;
    getParam("steps", steps, argc, argv);
    getParam("nlayer", nlayer, argc, argv);

    //  load images
    cv::Mat img, disp;
    loadImage(argc, argv, img);
    loadDisparityMap(argc, argv, disp);


    // discretize depth map
    double min, max;
    cv::minMaxIdx(disp, &min, &max);

    disp.convertTo(disp, CV_32F);
    disp -= min;
    disp /= (max - min);
    disp *= nlayer;
    disp.convertTo(disp, CV_8U);


    // initialize algorithm
    LinearDiffusion ld;

    ld.setImage(img);
    ld.setDisparityMap(disp);
    // ld.showInputImages();

    ld.setParameters(argc, argv);
    ld.initialize();

    //  register mouse callback
    cv::imshow("output", img);
    cv::setMouseCallback("output", didPressMouseButton, &ld);

    //  run diffusion

    int key = 0;

    while (true)
    {
        key = cv::waitKey();

        if (key == 27) {
            break;
        }
        else 
        {
            // logger << "update"; logger.eol();
            
            for (int i = 0; i < steps; ++i) {
                ld.adaptiveUpdate();
                // ld.intensityUpdate();
            }

            ld.showDiffusedImage("output");
        }
    }

    return EXIT_SUCCESS;
}


