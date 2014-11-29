#ifndef DISPARITY_ESTIMATOR_H
#define DISPARITY_ESTIMATOR_H


#include "../../common/core/aux.h"


class DisparityEstimator
{
    public:

        //  constructor
        DisparityEstimator();
        ~DisparityEstimator();

        //  setter
        void setLeftImage(const cv::Mat& left) { left.copyTo(img1_); };
        void setRightImage(const cv::Mat& right) { right.copyTo(img2_); };
        void setImagePair(const cv::Mat& img1, const cv::Mat& img2);

        void setParameters(int argc, char* argv[]);
        void setDepth(const size_t depth) { d_ = depth; };
        void setMu(const float mu) { mu_ = mu; };
        void setAlpha(const float alpha) { alpha_ = alpha; };
        void setTheta(const float theta) { theta_ = theta; };
        void setRegularizer(const string& regularizer);

        //  getter
        const size_t getDepth() { return d_; };
        const float getMu() { return mu_; };
        const float getAlpha() { return alpha_; };
        const float getTheta() { return theta_; };
        const string getRegularizer() { return regularizer_; };
        const int getRegularizerId();
        const void getDeviceDisparity(cv::Mat& disparity);

        //  algorithm
        void initialize();  
        void update();

        //  utility
        void showHostInputImages(const string& windowName = "hostInputImages");
        void showDeviceDisparity(const string& windowName = "deviceDisparity");


    protected:

        //  initialization
        void setParameters();
        void allocateHostMemory();
        void allocateDeviceMemory();
        void freeAllMemory();
        void copyImagePairHostToDevice();
        

    private:

        //  utility
        Timer timer_;

        //  parameters
        string regularizer_;
        size_t d_;
        float mu_, alpha_, theta_;

        static const float tau_;
        static const float sigma_;

        //  properties
        size_t w_, h_, nc_;
        size_t size2_, size2v_, size3_, size3v_;
        size_t bytes2i_, bytes2fv_, bytes3f_, bytes3fv_;

        //  kernel
        dim3 dim_, block_, grid_;

        //  input images
        cv::Mat img1_, img2_;
        float* h_img1_, * h_img2_;
        float* d_img1_, * d_img2_;

        //  algorithm variables
        float* h_v_, * d_v_;
        int* h_disp_, * d_disp_;

        float* d_g_;
        float* d_vBar_, * d_vGrad_;
        float* d_Phi_, * d_divPhi_;
};


#endif