#include "../../common/core/Logger.h"

#include "LinearDiffusion.h"
#include "LinearDiffusionKernels.h"


////////////////////////////////////////////////////////////////////////////////
LinearDiffusion::LinearDiffusion() :
    tau_(0.1f),
    eta_(2.0f),
    maxDisp_(255),
    evenUpdate_(true)
{
    Logger logger("LinearDiffusion::LinearDiffusion");

    invalidatePointer();

    //  initialize disparity -> diffusivity 
    size1_ = 256;   //  supposes 8U grayscale images
    bytes1f_ = size1_ * sizeof(float);

    h_diff_ = new float[size1_];
    cudaMalloc(&d_diff_, bytes1f_); CUDA_CHECK;
    cudaDeviceSynchronize();

    memset(h_diff_, 0, bytes1f_);
    cudaMemset(d_diff_, 0, bytes1f_); CUDA_CHECK;
    cudaDeviceSynchronize();
}


////////////////////////////////////////////////////////////////////////////////
//  also update properties and fix all sizes and buffers
void LinearDiffusion::setImage(const cv::Mat& img)
{
    Logger logger("LinearDiffusion::setImage");

    const cv::Size prevSize = img_.size();

    //  convert, save and upload image
    img.convertTo(img_, CV_32F);
    img_ /= 255.0f;

    if (prevSize != img_.size())
    {
        updateParameters();
        freeMemory();

        h_img_ = new float[size2v_];
        h_disp_ = new int[size2_];

        cudaMalloc(&d_u0_, bytes2fv_); CUDA_CHECK;
        cudaMalloc(&d_u1_, bytes2fv_); CUDA_CHECK;
        cudaMalloc(&d_disp_, bytes2i_); CUDA_CHECK;
        cudaDeviceSynchronize();
    }

    convert_mat_to_layered(h_img_, img_);
    cudaMemcpy(d_u0_, h_img_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::setDisparityMap(const cv::Mat& disp) 
{ 
    Logger logger("LinearDiffusion::setDisparityMap");

    if (disp.size() != img_.size()) {
        logger << "disparity size does not match image size"; logger.eol();
        logger.pop(false);
    }

    //  convert, save and upload disparity
    disp.convertTo(disp_, CV_32S);

    cudaMemcpy(d_disp_, disp_.ptr(), bytes2i_, cudaMemcpyHostToDevice); 
    CUDA_CHECK;

    //  save maximal disparity
    double min, max;
    cv::minMaxIdx(disp_, &min, &max);
    maxDisp_ = ceil(max);
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::setParameters(int argc, char* argv[])
{
    Logger logger("LinearDiffusion::setParameters");

    getParam("tau", tau_, argc, argv);
    getParam("eta", eta_, argc, argv);

    logger << "tau: " << tau_; logger.eol();
    logger << "eta: " << eta_; logger.eol();
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::updateParameters()
{
    Logger logger("LinearDiffusion::updateParameters");

    //  image properties
    w_ = img_.cols;
    h_ = img_.rows;
    nc_ = img_.channels();

    //  unit sizes
    size2_ = w_ * h_;
    size2v_ = w_ * h_ * nc_;

    //  memory sizes
    bytes2i_ = size2_ * sizeof(int);
    bytes2fv_ = size2v_ * sizeof(float);

    //  kernel
    dim_ = dim3(w_, h_, nc_);

#ifdef ZORAH
    block_ = dim3(16, 8, 1);
#else
    block_ = dim3(32, 16, 1);
#endif

    grid_.x = (w_ + block_.x - 1) / block_.x;
    grid_.y = (h_ + block_.y - 1) / block_.y;
    grid_.z = 1;

    //  print parameters
    logger << "image size: (" << w_ << " x " << h_ << " x " << nc_ << ")";
    logger << " [" << bytes2fv_ / 1000000.0f << " mb]"; logger.eol();

    logger << "block: " << block_; logger.eol();
    logger << "grid: " << grid_; logger.eol();
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::reset() 
{
    Logger logger("LinearDiffusion::reset"); 

    cudaMemcpy(d_u0_, h_img_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemset(d_diff_, 0, bytes1f_); CUDA_CHECK;

    evenUpdate_ = true; //  start with u0
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::update()
{
    Logger logger("LinearDiffusion::update"); 

    if (evenUpdate_) {
        linear_diffusion<<<grid_, block_>>>(d_u0_, d_u1_, tau_, dim_); 
    } else {
        linear_diffusion<<<grid_, block_>>>(d_u1_, d_u0_, tau_, dim_); 
    }
    CUDA_CHECK;
    cudaDeviceSynchronize();

    evenUpdate_ = !evenUpdate_;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::adaptiveUpdate()
{
    Logger logger("LinearDiffusion::adaptiveUpdate"); 

    if (evenUpdate_) {
        adaptive_linear_diffusion<<<grid_, block_>>>(
            d_u0_, d_u1_, d_disp_, d_diff_, dim_, tau_
        );
    } else {
        adaptive_linear_diffusion<<<grid_, block_>>>(
            d_u1_, d_u0_, d_disp_, d_diff_, dim_, tau_
        );
    }
    CUDA_CHECK;
    cudaDeviceSynchronize();

    evenUpdate_ = !evenUpdate_;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::intensityUpdate()
{
    Logger logger("LinearDiffusion::intensityUpdate"); 

    adaptive_intensity<<<grid_, block_>>>(d_u0_, d_disp_, d_diff_, dim_); 
    CUDA_CHECK;
    cudaDeviceSynchronize();
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::setFocus(const int depth)
{
    Logger logger("LinearDiffusion::setFocus"); 

    for (int i = 0; i <= maxDisp_; i++) {
        const float d = abs(depth - i) / (float)maxDisp_;
        h_diff_[i] = powf(d, eta_);
    }

    //  copy to device
    cudaMemcpy(d_diff_, h_diff_, bytes1f_, cudaMemcpyHostToDevice); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::setFocus(const int x, const int y) {
    const int depth = disp_.at<int>(y, x);
    setFocus(depth);
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::showInputImages(const string& windowName)
{
    cv::Mat disp;
    cv::cvtColor(disp_, disp, CV_GRAY2BGR);
    disp *= (255.0f / maxDisp_);

    cv::Mat img12;
    cv::hconcat(img_, disp, img12);
    cv::imshow(windowName, img12);
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::showDiffusedImage(const string& windowName)
{
    float* h_u0 = new float[size2v_];
    cudaMemcpy(h_u0, d_u0_, bytes2fv_, cudaMemcpyDeviceToHost); CUDA_CHECK;

    cv::Mat img = cv::Mat(h_, w_, CV_32FC3);
    convert_layered_to_mat(img, h_u0);

    cv::imshow(windowName, img);

    delete h_u0;
}

////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::getDiffusedImage(cv::Mat& diff) 
{
    float* h_u0 = new float[size2v_];
    cudaMemcpy(h_u0, d_u0_, bytes2fv_, cudaMemcpyDeviceToHost); CUDA_CHECK;

    diff = cv::Mat(h_, w_, CV_32FC3);
    convert_layered_to_mat(diff, h_u0);
    diff *= 255;
    diff.convertTo(diff, CV_8U);

    delete h_u0;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::invalidatePointer()
{
    Logger logger("LinearDiffusion::invalidatePointer");

    //  invalidate pointers
    d_u0_ = 0;
    d_u1_ = 0;
    d_disp_ = 0;

    h_img_ = 0;
    h_disp_ = 0;
}


////////////////////////////////////////////////////////////////////////////////
void LinearDiffusion::freeMemory()
{
    Logger logger("LinearDiffusion::freeMemory");

    //  free vram   
    if (d_u0_)      cudaFree(d_u0_); CUDA_CHECK;  
    if (d_u1_)      cudaFree(d_u1_); CUDA_CHECK;  
    if (d_disp_)    cudaFree(d_disp_); CUDA_CHECK;

    //  free ram
    if (h_img_)     delete[] h_img_;
    if (h_disp_)    delete[] h_disp_;

    invalidatePointer();
}


////////////////////////////////////////////////////////////////////////////////
LinearDiffusion::~LinearDiffusion()
{
    Logger logger("LinearDiffusion::~LinearDiffusion");

    if (d_diff_)    cudaFree(d_diff_); CUDA_CHECK;
    if (h_diff_)    delete[] h_diff_;

    freeMemory();
}
