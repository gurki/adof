#ifndef LINEAR_DIFFUSION_H
#define LINEAR_DIFFUSION_H


#include "../../common/core/aux.h"


class LinearDiffusion
{
    public:

        //  constructor
        LinearDiffusion();
        ~LinearDiffusion();

        //  setter
        void setImage(const cv::Mat& img);
        void setDisparityMap(const cv::Mat& disp);
        void setParameters(int argc, char* argv[]);

        void setFocus(const int depth);
        void setFocus(const int x, const int y);

        void setEta(const float eta) { eta_ = eta; };

        //  getter
        void getImage(cv::Mat& image) { img_.copyTo(image); };
        void getDisparity(cv::Mat& disparity) { disp_.convertTo(disparity, CV_8U); };
        void getDiffusedImage(cv::Mat& diff);
        
        bool isValid() { return !img_.empty() && !disp_.empty(); };

        //  algorithm
        void reset();
        void update();
        void adaptiveUpdate();
        void intensityUpdate();

        //  utility
        void showInputImages(const string& windowName = "inputImages");
        void showDiffusedImage(const string& windowName = "diffusedImage");


    protected:

        //  initialization
        void updateParameters();        
        void invalidatePointer();
        void freeMemory();

    private:

        //  parameters
        bool evenUpdate_;
        float tau_, eta_;
        int maxDisp_;

        //  properties
        size_t w_, h_, nc_;
        size_t size2_, size2v_, size1_;
        size_t bytes2i_, bytes2fv_, bytes1f_;

        //  kernel
        dim3 dim_, block_, grid_;

        //  host variables
        cv::Mat img_, disp_;

        float* h_img_;
        int* h_disp_;
        float* h_diff_;

        //  device variables
        float* d_u0_, * d_u1_;
        int* d_disp_;
        float* d_diff_;
};


#endif