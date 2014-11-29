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
//  Nicolai Oswald, mail@nicolai-oswaip.de, p065
//
////////////////////////////////////////////////////////////////////////////////


#include "../../common/core/aux.h"
#include "../../common/core/Logger.h"

#include "Inpainting.h"


int steps = 100;


void didPressMouseButton(int event, int x, int y, int, void* data)
{
    if (event == cv::EVENT_LBUTTONDOWN) 
    {
        Logger logger("didPressMouseButton");

        if (!data) {
            logger << "no inpainting object received"; logger.eol();
            logger.pop(false);
        }

        Inpainting* ip = (Inpainting*)data;

        ip->setFocus(x, y);

        // cudaStartTimer();

        for (int i = 0; i < steps; ++i) {
            ip->update();
        }

        // cout << "dt: " << cudaStopTimer() / steps << " ms" << endl;

        ip->showInpaintedImage("output");
    }
}


int main(int argc, char* argv[])
{
    Logger logger("main");
    Logger::setMaxLevel(1);

    printDeviceMemoryUsage();

    getParam("steps", steps, argc, argv);


    //  load images
    cv::Mat img, disp;
    loadImage(argc, argv, img);
    loadDisparityMap(argc, argv, disp);

    const int d = 5;
    disp.convertTo(disp, CV_32F);
    disp /= (255.0f / d);
    disp.convertTo(disp, CV_8U);


    // initialize algorithm
    Inpainting ip;

    ip.setImage(img);
    ip.setMask(disp);
    ip.showMask();

    ip.setParameters(argc, argv);
    ip.initialize();

    //  register mouse callback
    cv::imshow("output", img);
    cv::setMouseCallback("output", didPressMouseButton, &ip);

    //  run inpainting
    int key = 0;

    while (true)
    {
        key = cv::waitKey();

        if (key == 27) {
            break;
        } else if (key == 'r') {
            ip.initialize();
            ip.showInpaintedImage("output");
        }
        else 
        {
            // logger << "update"; logger.eol();
            
            for (int i = 0; i < steps; ++i) {
                ip.update();
            }

            ip.showInpaintedImage("output");
        }
    }

    return EXIT_SUCCESS;
}


