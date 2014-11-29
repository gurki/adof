#ifndef INPAINTING_H
#define INPAINTING_H


#include "../../common/core/aux.h"


class Inpainting
{
    public:

        //  constructor
        Inpainting();
        ~Inpainting();

        //  setter
        void setImage(const cv::Mat& img);
        void setMask(const cv::Mat& mask);
        void setParameters(int argc, char* argv[]);

        void setFocus(const int depth);
        void setFocus(const int x, const int y);

        //  getter
        void getImage(cv::Mat& image) { img_.copyTo(image); };
        void getMask(cv::Mat& mask) { mask_.copyTo(mask); };
        void getInpaintedImage(cv::Mat& inpaint);
        
        bool isValid() { return !img_.empty() && !mask_.empty(); };

        //  algorithm
        void reset();
        void update();

        //  utility
        void showMask(const string& windowName = "mask");
        void showInputImages(const string& windowName = "inputImages");
        void showInpaintedImage(const string& windowName = "diffusedImage");


    protected:

        //  initialization
        void updateParameters();
        void invalidatePointer();
        void freeMemory();

    private:

        //  parameters
        float theta_, eps_;
        int maskFlag_;

        //  properties
        size_t w_, h_, nc_;
        size_t size2_, size2v_;
        size_t bytes2i_, bytes2f_, bytes2fv_;

        //  kernel
        dim3 dim_, block_, grid_;

        //  host variables
        cv::Mat img_, mask_;

        float* h_img_;
        int* h_mask_;

        //  device variables
        float* d_u_, * d_g_, * d_dx_, * d_dy_;
        int* d_mask_;
};


#endif