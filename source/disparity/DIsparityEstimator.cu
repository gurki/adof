#include "../../common/core/Logger.h"

#include "DisparityEstimator.h"
#include "DisparityEstimatorKernels.h"


//  static class members
// const float DisparityEstimator::tau_ = 1.0f / sqrtf(12.0f);
// const float DisparityEstimator::sigma_ = 1.0f / sqrtf(12.0f);
const float DisparityEstimator::tau_ = 1.0f / 6.0f;
const float DisparityEstimator::sigma_ = 1.0f / 2.0f;


////////////////////////////////////////////////////////////////////////////////
DisparityEstimator::DisparityEstimator() :
    d_(8),
    mu_(50.0f),
    alpha_(5.0f),
    theta_(0.5f),
    regularizer_("huber")
{
    Logger logger("DisparityEstimator::DisparityEstimator");

    //  init pointers
    h_img1_     = 0;
    h_img2_     = 0;
    h_v_        = 0;
    h_disp_     = 0;

    d_img1_     = 0;
    d_img2_     = 0;
    d_v_        = 0;
    d_vBar_     = 0;
    d_vGrad_    = 0;
    d_Phi_      = 0;
    d_divPhi_   = 0;
    d_g_        = 0;
    d_disp_     = 0;
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::setImagePair(const cv::Mat& img1, const cv::Mat& img2)
{
    Logger logger("DisparityEstimator::setImagePair");

    img1_ = img1;
    img2_ = img2;
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::setParameters(int argc, char* argv[])
{
    Logger logger("DisparityEstimator::setParameters");

    getParam("d", d_, argc, argv);
    getParam("mu", mu_, argc, argv);    
    getParam("alpha", alpha_, argc, argv);  
    getParam("theta", theta_, argc, argv);  
    getParam("reg", regularizer_, argc, argv);

    setRegularizer(regularizer_);

    logger << "d: " << d_; logger.eol();
    logger << "mu: " << mu_; logger.eol();
    logger << "alpha: " << alpha_; logger.eol();
    logger << "theta: " << theta_; logger.eol();
    logger << "tau: " << tau_; logger.eol();
    logger << "sigma: " << sigma_; logger.eol();
    logger << "reg: " << regularizer_; logger.eol();
}


////////////////////////////////////////////////////////////////////////////////
const int DisparityEstimator::getRegularizerId()
{
    if (regularizer_ == "huber") {
        return REG_HUBER;
    } else if (regularizer_ == "tv") {
        return REG_TV;
    } else if (regularizer_ == "quadratic") {
        return REG_QUADRATIC;
    }

    return -1;
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::setRegularizer(const string& regularizer)
{
    Logger logger("DisparityEstimator::setRegularizer");

    regularizer_ = regularizer;

    if (getRegularizerId() < 0) {
        logger << "unsupported regularizer " << regularizer_; logger.eol();
        logger.pop(false);
    }
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::setParameters()
{
    Logger logger("DisparityEstimator::setDimensions");

    //  image properties
    w_ = img1_.cols;
    h_ = img1_.rows;
    nc_ = img1_.channels();

    //  unit sizes
    size2_ = w_ * h_;
    size2v_ = w_ * h_ * nc_;
    size3_ = w_ * h_ * d_;
    size3v_ = w_ * h_ * d_ * 3;

    //  memory sizes
    bytes2i_ = size2_ * sizeof(int);
    bytes2fv_ = size2v_ * sizeof(float);
    bytes3f_ = size3_ * sizeof(float);
    bytes3fv_ = size3v_ * sizeof(float);

    //  kernel
    dim_ = dim3(w_, h_, d_);

#ifdef ZORAH
    block_ = dim3(128 / d_, 1, d_);
#else
    block_ = dim3(32, 16, 1);
#endif

    grid_.x = (w_ + block_.x - 1) / block_.x;
    grid_.y = (h_ + block_.y - 1) / block_.y;

#ifdef ZORAH
    grid_.z = 1;
#else
    grid_.z = (d_ + block_.z - 1) / block_.z;
#endif

    //  print parameters
    logger << "disparity size: (" << w_ << " x " << h_ << ")";
    logger << " [" << bytes2i_ / 1000000.0f  << " mb]"; logger.eol();

    logger << "image size: (" << w_ << " x " << h_ << " x " << nc_ << ")";
    logger << " [" << bytes2fv_ / 1000000.0f << " mb]"; logger.eol();

    logger << "volume size: (" << w_ << " x " << h_ << " x " << d_ << ")";
    logger << " [" << bytes3f_ / 1000000.0f  << " mb]"; logger.eol();

    logger << "flux size: (" << w_ << " x " << h_ << " x ";
    logger << d_ << " x " << "3)";
    logger << " [" << bytes3fv_ / 1000000.0f  << " mb]"; logger.eol();

    logger << "block: " << block_; logger.eol();
    logger << "grid: " << grid_; logger.eol();

    //  also, set device parameters
    float tau = tau_;
    float sigma = sigma_;

    setConstantMemory(
        &w_, &h_, &d_, 
        &size2_, &size3_, 
        &tau, &sigma
    );
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::allocateHostMemory()
{
    Logger logger("DisparityEstimator::allocateHostMemory");

    h_img1_ = new float[size2v_];
    h_img2_ = new float[size2v_];
    h_v_ = new float[size3_];
    h_disp_ = new int[size2_];

    memset(h_img1_, 0, bytes2fv_);
    memset(h_img2_, 0, bytes2fv_);
    memset(h_v_, 0, bytes3f_);
    memset(h_disp_, 0, bytes2i_);
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::allocateDeviceMemory()
{
    Logger logger("DisparityEstimator::allocateDeviceMemory");

    //  alloc input images
    cudaMalloc(&d_img1_, bytes2fv_); CUDA_CHECK;
    cudaMalloc(&d_img2_, bytes2fv_); CUDA_CHECK;

    cudaMemset(d_img1_, 0, bytes2fv_); CUDA_CHECK;
    cudaMemset(d_img2_, 0, bytes2fv_); CUDA_CHECK;

    //  alloc algorithm variables
    cudaMalloc(&d_v_, bytes3f_); CUDA_CHECK;
    cudaMalloc(&d_vBar_, bytes3f_); CUDA_CHECK;
    cudaMalloc(&d_vGrad_, bytes3fv_); CUDA_CHECK;
    cudaMalloc(&d_Phi_, bytes3fv_); CUDA_CHECK;
    cudaMalloc(&d_divPhi_, bytes3f_); CUDA_CHECK;
    cudaMalloc(&d_g_, bytes3f_); CUDA_CHECK;
    cudaMalloc(&d_disp_, bytes2i_); CUDA_CHECK;

    cudaMemset(d_v_, 0, bytes3f_); CUDA_CHECK;
    cudaMemset(d_vBar_, 0, bytes3f_); CUDA_CHECK;
    cudaMemset(d_vGrad_, 0, bytes3fv_); CUDA_CHECK;
    cudaMemset(d_Phi_, 0, bytes3fv_); CUDA_CHECK;
    cudaMemset(d_divPhi_, 0, bytes3f_); CUDA_CHECK;
    cudaMemset(d_g_, 0, bytes3f_); CUDA_CHECK;
    cudaMemset(d_disp_, 0, bytes2i_); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::copyImagePairHostToDevice()
{
    Logger logger("DisparityEstimator::copyImagePairHostToDevice");

    cv::Mat img1, img2;

    img1_.convertTo(img1, CV_32F);
    img2_.convertTo(img2, CV_32F);

    img1 /= 255.0f;
    img2 /= 255.0f;

    convert_mat_to_layered(h_img1_, img1);
    convert_mat_to_layered(h_img2_, img2);

    cudaMemcpy(d_img1_, h_img1_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
    cudaMemcpy(d_img2_, h_img2_, bytes2fv_, cudaMemcpyHostToDevice); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::initialize()
{
    Logger logger("DisparityEstimator::initialize"); 

    //  compute missing parameters
    setParameters();

    if (img1_.empty() || img2_.empty()) {
        logger.pop("invalid image asigned");
    }

    //  allocate memory
    freeAllMemory();
    allocateHostMemory();
    allocateDeviceMemory();

    //  initialize gpu variables
    copyImagePairHostToDevice();

    computeDataTerm<<<grid_, block_>>>(d_img1_, d_img2_, d_g_, mu_, nc_); 
    CUDA_CHECK;
    
    initialise_v<<<grid_, block_>>>(d_v_, d_vBar_); 
    CUDA_CHECK;

    cudaDeviceSynchronize();
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::update()
{
    Logger logger("DisparityEstimator::update"); 

    //  primal-dual update: Phi
    update_phi<<<grid_, block_>>>(
        d_g_, d_Phi_, d_vBar_, d_vGrad_, 
        alpha_, getRegularizerId()); 
    CUDA_CHECK;

    cudaDeviceSynchronize();

    //  primal-dual update: v, v_bar
    update_v_vBar<<<grid_, block_>>>(d_v_, d_vBar_, d_Phi_, d_divPhi_); 
    CUDA_CHECK;

    cudaDeviceSynchronize();
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::showHostInputImages(const string& windowName)
{
    cv::Mat img12;
    cv::hconcat(img1_, img2_, img12);
    cv::imshow(windowName, img12);
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::showDeviceDisparity(const string& windowName) {
    cv::Mat disp;
    getDeviceDisparity(disp);
    cv::imshow(windowName, disp);
}


////////////////////////////////////////////////////////////////////////////////
const void DisparityEstimator::getDeviceDisparity(cv::Mat& disparity)
{
    //  reduce volume to disparity map
    cudaMemset(d_disp_, 0, bytes2i_); CUDA_CHECK;
    cudaDeviceSynchronize(); CUDA_CHECK;
    getDepthMap<<<grid_, block_>>>(d_v_, d_disp_, theta_); CUDA_CHECK;

    //  copy to host float array
    cudaMemcpy(h_disp_, d_disp_, bytes2i_, cudaMemcpyDeviceToHost); 
    CUDA_CHECK;

    cv::Mat disp = cv::Mat(h_, w_, CV_32S, h_disp_);

    //  normalize to [0, 1]
    disp.convertTo(disparity, CV_32F);
    disparity /= d_;
        
    //  normalize to [min, max]
    // double min, max;
    // cv::minMaxIdx(disp32f, &min, &max);
    // disp32f -= min;
    // disp32f /= (max - min);
    // disp32f = 1.0f - disp32f; 

    disparity *= 255.0f;
    disparity.convertTo(disparity, CV_8U);
}


////////////////////////////////////////////////////////////////////////////////
void DisparityEstimator::freeAllMemory()
{
    Logger logger("DisparityEstimator::freeAllMemory");

        //  free vram   
    if (d_img1_)    cudaFree(d_img1_); CUDA_CHECK;  
    if (d_img2_)    cudaFree(d_img2_); CUDA_CHECK;

    if (d_v_)       cudaFree(d_v_); CUDA_CHECK;
    if (d_vBar_)    cudaFree(d_vBar_); CUDA_CHECK;
    if (d_vGrad_)   cudaFree(d_vGrad_); CUDA_CHECK;

    if (d_Phi_)     cudaFree(d_Phi_); CUDA_CHECK;
    if (d_divPhi_)  cudaFree(d_divPhi_); CUDA_CHECK;

    if (d_g_)       cudaFree(d_g_); CUDA_CHECK;
    if (d_disp_)    cudaFree(d_disp_); CUDA_CHECK;

    //  free ram
    if (h_img1_)    delete[] h_img1_;
    if (h_img2_)    delete[] h_img2_;

    if (h_v_)       delete[] h_v_;
    if (h_disp_)    delete[] h_disp_;

    //  invalidate pointers
    d_img1_ = 0;
    d_img2_ = 0;
    d_v_ = 0;
    d_vBar_ = 0;
    d_vGrad_ = 0;
    d_Phi_ = 0;
    d_divPhi_ = 0;
    d_g_ = 0;
    d_disp_ = 0;
    h_img1_ = 0;
    h_img2_ = 0;
    h_v_ = 0;
    h_disp_ = 0;
}


////////////////////////////////////////////////////////////////////////////////
DisparityEstimator::~DisparityEstimator()
{
    Logger logger("DisparityEstimator::~DisparityEstimator");

    freeAllMemory();
}
