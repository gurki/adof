#ifndef DISPARITY_ESTIMATOR_KERNELS_H
#define DISPARITY_ESTIMATOR_KERNELS_H


#define REG_TV          0
#define REG_QUADRATIC   1
#define REG_HUBER       2
// #define REG_LIPSCHITZ   4


//  constant device memory
__constant__ size_t c_w;
__constant__ size_t c_h;
__constant__ size_t c_d;

__constant__ size_t c_size2;
__constant__ size_t c_size3;

__constant__ float c_tau;
__constant__ float c_sigma;


void setConstantMemory(
    const size_t* w,
    const size_t* h,
    const size_t* d,
    const size_t* size2,
    const size_t* size3,
    const float* tau,
    const float* sigma
);


// initialisation

__global__ void computeDataTerm(
    const float* img1,  // height * width * 3
    const float* img2,  // height * width * 3
    float* data,        // height * width * depth
    const float mu,     // weighting > 0
    const int nc        // number of image channels
);

__global__ void initialise_v(
    float* v,           // width * height * depth
    float* vBar         // width * height * depth
);


// primal-dual update steps

__global__ void update_phi(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3  in form [qx, qy, qt]
    float* v_bar,       // width * height * depth
    float* v_grad,      // width * height * depth * 3
    const float alpha,  // constant > 0
    const int type      // regularization type REG_*
);

__global__ void update_v_vBar(
    float* v,           // width * height * depth
    float* vBar,        // width * height * depth
    const float* phi,   // width * height * depth * 3  in form [qx, qy, qt]
    float* divPhi       // width * height * depth
);


// gradient / divergence

__device__ void divergence(
    const float* phi,   // width * height * depth * 3  in form [qx, qy, qt]
    float* divPhi,      // width * height * depth
    const size_t id     // current id3
);

__device__ void grad3(
    const float* vBar,  // width * height * depth
    float* dvBar,       // width * height * depth * 3
    const size_t id     // current id3
);


// projections

__device__ void project_phi_TV(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id     // current id3
);

__device__ void project_phi_Quadratic(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id);

__device__ void project_phi_Huber(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const float alpha,  // constant > 0
    const size_t id     // current id3
);


__device__ void project_v(
    float* v,           // width * height * dpeth
    const size_t id     // current id3
);

__device__ float getLambda(
    const float* phi,   // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id     // current id3
);


// getter and helper function

__global__ void getDepthMap(
    const float* v, // depth * height * width
    int* disp, // height * width 
    const float theta
);

__device__ void getId(
    int *x,
    int *y,
    int *z
);


#endif 