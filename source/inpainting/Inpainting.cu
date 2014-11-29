#include "../../common/core/Logger.h"

#include "Inpainting.h"
#include "InpaintingKernels.h"


////////////////////////////////////////////////////////////////////////////////
Inpainting::Inpainting() :
    theta_(0.87f),
    eps_(0.5f),
    maskFlag_(0)
{
    Logger logger("Inpainting::Inpainting");

    invalidatePointer();
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::setParameters(int argc, char* argv[])
{
    Logger logger("Inpainting::setParameters");

    getParam("eps", eps_, argc, argv);
    getParam("theta", theta_, argc, argv);

    logger << "eps: " << eps_; logger.eol();
    logger << "theta: " << theta_; logger.eol();
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::setImage(const cv::Mat& img) 
{
    Logger logger("Inpainting::setImage");

    const cv::Size prevSize = img_.size();

    img.convertTo(img_, CV_32F);
    img_ /= 255.0f;

    if (img_.size() != prevSize)
    {
        updateParameters();

        freeMemory();

        h_img_ = new float[size2v_];
        h_mask_ = new int[size2_];

        cudaMalloc(&d_u_, bytes2fv_); CUDA_CHECK;
        cudaMalloc(&d_mask_, bytes2i_); CUDA_CHECK;
        cudaMalloc(&d_g_, bytes2f_); CUDA_CHECK;
        cudaMalloc(&d_dx_, bytes2fv_); CUDA_CHECK;
        cudaMalloc(&d_dy_, bytes2fv_); CUDA_CHECK;
        cudaDeviceSynchronize();
    }

    //  copy image
    convert_mat_to_layered(h_img_, img_);
    cudaMemcpy(d_u_, h_img_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::setMask(const cv::Mat& mask) 
{ 
    Logger logger("Inpainting::setMask");

    if (mask.size() != img_.size()) {
        logger << "mask size does not match image size"; logger.eol();
        logger.pop(false);
    }

    mask.convertTo(mask_, CV_32S);

    //  copy to device
    cudaMemcpy(d_mask_, mask_.ptr(), bytes2i_, cudaMemcpyHostToDevice); 
    CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::reset() {
    Logger logger("Inpainting::reset");
    cudaMemcpy(d_u_, h_img_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::updateParameters()
{
    Logger logger("Inpainting::updateParameters");

    //  image properties
    w_ = img_.cols;
    h_ = img_.rows;
    nc_ = img_.channels();

    //  unit sizes
    size2_ = w_ * h_;
    size2v_ = w_ * h_ * nc_;

    //  memory sizes
    bytes2i_ = size2_ * sizeof(int);
    bytes2f_ = size2_ * sizeof(float);
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

    set_constant_memory(&w_, &h_, &nc_, &size2_);
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::update()
{
    Logger logger("Inpainting::update"); 

    for (int i = 0; i < 2; i++)
    {
        const bool red = (i % 2 == 0);

        diffusivity_rb<<<grid_, block_>>>(
            d_u_, d_mask_, maskFlag_, d_g_, d_dx_, d_dy_, eps_, !red
        ); 
        CUDA_CHECK;

        cudaDeviceSynchronize();

        inpaint_update_rb_sor<<<grid_, block_>>>(
            d_u_, d_g_, d_mask_, maskFlag_, red, theta_
        ); 
        CUDA_CHECK;

        cudaDeviceSynchronize();
    }
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::setFocus(const int depth)
{
    Logger logger("Inpainting::setFocus"); 

    maskFlag_ = depth;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::setFocus(const int x, const int y) {
    const int depth = mask_.at<int>(y, x);
    setFocus(depth);
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::showInputImages(const string& windowName)
{
    cv::Mat mask;
    cv::cvtColor(mask_, mask, CV_GRAY2BGR);
    double min, max;
    cv::minMaxIdx(mask, &min, &max);
    mask *= (255.0f / max);

    cv::Mat img12;
    cv::hconcat(img_, mask, img12);
    cv::imshow(windowName, img12);
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::showMask(const string& windowName)
{
    cv::Mat mask;
    cv::cvtColor(mask_, mask, CV_GRAY2BGR);
    double min, max;
    cv::minMaxIdx(mask, &min, &max);
    mask *= (255.0f / max);
    cv::imshow(windowName, mask);
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::showInpaintedImage(const string& windowName)
{
    float* h_u = new float[size2v_];
    cudaMemcpy(h_u, d_u_, bytes2fv_, cudaMemcpyDeviceToHost); CUDA_CHECK;

    cv::Mat img = cv::Mat(h_, w_, CV_32FC3);
    convert_layered_to_mat(img, h_u);

    cv::imshow(windowName, img);

    delete h_u;
}

////////////////////////////////////////////////////////////////////////////////
void Inpainting::getInpaintedImage(cv::Mat& inpaint)
{
    float* h_u = new float[size2v_];
    cudaMemcpy(h_u, d_u_, bytes2fv_, cudaMemcpyDeviceToHost); CUDA_CHECK;

    inpaint = cv::Mat(h_, w_, CV_32FC3);
    convert_layered_to_mat(inpaint, h_u);
    inpaint *= 255;
    inpaint.convertTo(inpaint, CV_8U);

    delete h_u;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::invalidatePointer()
{
    Logger logger("Inpainting::invalidatePointer");

    //  invalidate pointers
    d_u_ = 0;
    d_g_ = 0;
    d_mask_ = 0;
    d_dx_ = 0;
    d_dy_ = 0;

    h_img_ = 0;
    h_mask_ = 0;
}


////////////////////////////////////////////////////////////////////////////////
void Inpainting::freeMemory()
{
    Logger logger("Inpainting::freeMemory");

    //  free vram   
    if (d_u_)       cudaFree(d_u_); CUDA_CHECK;  
    if (d_g_)       cudaFree(d_g_); CUDA_CHECK;  
    if (d_mask_)    cudaFree(d_mask_); CUDA_CHECK;
    if (d_dx_)      cudaFree(d_dx_); CUDA_CHECK;
    if (d_dy_)      cudaFree(d_dy_); CUDA_CHECK;

    //  free ram
    if (h_img_)     delete[] h_img_;
    if (h_mask_)    delete[] h_mask_;

    invalidatePointer();
}


////////////////////////////////////////////////////////////////////////////////
Inpainting::~Inpainting()
{
    Logger logger("Inpainting::~Inpainting");

    freeMemory();
}
