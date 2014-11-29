#include "DisparityEstimatorKernels.h"

#include "../../common/core/aux.h"


#ifndef MIN
    #define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
    #define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif


////////////////////////////////////////////////////////////////////////////////
//  constant device memory
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void setConstantMemory(
    const size_t* w,
    const size_t* h,
    const size_t* d,
    const size_t* size2,
    const size_t* size3,
    const float* tau,
    const float* sigma)
{
    cudaMemcpyToSymbol(c_w, w, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(c_h, h, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(c_d, d, sizeof(size_t)); CUDA_CHECK;

    cudaMemcpyToSymbol(c_size2, size2, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(c_size3, size3, sizeof(size_t)); CUDA_CHECK;

    cudaMemcpyToSymbol(c_tau, tau, sizeof(float)); CUDA_CHECK;
    cudaMemcpyToSymbol(c_sigma, sigma, sizeof(float)); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
//  initialisation
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
__global__ void computeDataTerm(
    const float* img1,  // height * width * 3
    const float* img2,  // height * width * 3
    float* data,        // height * width * depth
    const float mu,     // weighting > 0
    const int nc)
{
    int x, y, z;
    getId(&x, &y, &z);

    if(x < c_w && y < c_h && z < c_d) 
    {
        //  get pixel (2d) and volume (3d) indices
        const size_t id2 = x + y * c_w;
        const size_t id3 = id2 + z * c_size2;

        // make sure disparity does not go out of the image
        const int offset = z;
        int dispX = x + offset;
        
        if(dispX >= c_w - 1) {
            dispX = x;
        }

        // sum over all channels, measuring difference between corresponding pixels
        // correspondence given by height
        float sum = 0.0f;

        for (int ch = 0; ch < nc; ch++) 
        {
            const size_t chOff = ch * c_size2;
            const size_t idc1 = id2 + chOff;
            const size_t idc2 = dispX + y * c_w + chOff;

            sum += abs(img1[idc1] - img2[idc2]);
        }

        data[id3] = mu * sum;
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__ void initialise_v(
    float* v,           // width * height * depth
    float* vBar)        // width * height * depth
{
    int x, y, z;
    getId(&x, &y, &z);

    if(x < c_w && y < c_h && z < c_d) 
    {
        const size_t id = x + y * c_w + z * c_size2;

        if (z == 0) {
            v[id] = 1.0f;
            vBar[id] = 1.0f;
        } else {
            v[id] = 0.0f;
            vBar[id] = 0.0f;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
//  primal-dual update steps
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// called with width * height * depth many threads
__global__ void update_phi(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3  in form [qx, qy, qt]
    float* v_bar,       // width * height * depth
    float* v_grad,      // width * height * depth * 3
    const float alpha,  // constant > 0
    const int type)     // regularization type REG_*
{
    int x, y, z;
    getId(&x, &y, &z);

    if(x < c_w && y < c_h && z < c_d) 
    {
        const size_t id = x + y * c_w + z * c_size2;

        const size_t idx = id;
        const size_t idy = idx + c_size3;
        const size_t idt = idy + c_size3;

        grad3(v_bar, v_grad, id);

        phi[idx] += c_sigma * v_grad[idx];
        phi[idy] += c_sigma * v_grad[idy];
        phi[idt] += c_sigma * v_grad[idt];

        //  project phi accordingly
        switch (type) 
        {
            case REG_TV:
                project_phi_TV(data, phi, id);
                break;

            case REG_HUBER:
                project_phi_Huber(data, phi, alpha, id);
                break;

            case REG_QUADRATIC:
                project_phi_Quadratic(data, phi, id);
                break;

            default:
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__ void update_v_vBar(
    float* v,           // width * height * depth
    float* vBar,        // width * height * depth
    const float* phi,   // width * height * depth * 3  in form [qx, qy, qt]
    float* divPhi)      // width * height * depth
{
    int x, y, z;
    getId(&x, &y, &z);

    if(x < c_w && y < c_h && z < c_d)
     {
        const size_t id = x + y * c_w + z * c_size2;

        //  2a)  update vbar
        vBar[id] = -v[id];

        //  1)  update and reproject v
        divergence(phi, divPhi, id);
        v[id] = v[id] + c_tau * divPhi[id];
        project_v(v, id);

        //  2b)  update vbar with v^{n+1}
        vBar[id] += 2 * v[id];
    }
}


////////////////////////////////////////////////////////////////////////////////
//  gradient / divergence
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// backwards divergence
__device__ void divergence(
    const float* phi,   // width * height * depth * 3  in form [qx, qy, qt]
    float* divPhi,      // width * height * depth
    const size_t id)    // current id3
{
    int x, y, z;
    getId(&x, &y, &z);

    const size_t idx = id;
    const size_t idy = idx + c_size3;
    const size_t idt = idy + c_size3;

    float divx = phi[idx];
    float divy = phi[idy];
    float divt = phi[idt];

    //  ! FIXED !
    // calculate x-component
    if(x > 0 && x < c_w - 1) {
        divx -= phi[idx - 1];
    } else if (x == c_w - 1) {
        divx = -phi[idx - 1];
    }

    // calculate y-component
    if(y > 0 && y < c_h - 1) {
        divy -= phi[idy - c_w];
    } else if (y == c_h - 1) {
        divy = -phi[idy - c_w];
    }

    // calculate t-component
    if(z > 0 && z < c_d - 1) {
        divt -= phi[idt - c_size2];
    } else if (z == c_d - 1) {
        divt = -phi[idt - c_size2];
    }

    divPhi[id] = divx + divy + divt;
}


////////////////////////////////////////////////////////////////////////////////
__device__ void grad3(
    const float* vBar,  // width * height * depth
    float* dvBar,       // width * height * depth * 3
    const size_t id)    // current id3
{
    int x, y, z;
    getId(&x, &y, &z);

    const size_t idx = id;
    const size_t idy = idx + c_size3;
    const size_t idt = idy + c_size3;

    // calculate x-vGradient
    if(x < (c_w - 1)) {
        dvBar[idx] = vBar[id + 1] - vBar[id];
    } else {
        dvBar[idx] = 0.0f;
    }

    // calculate y-gradient
    if(y < (c_h - 1)) {
        dvBar[idy] = vBar[id + c_w] - vBar[id];
    } else {
        dvBar[idy] = 0.0f;
    }

    // calculate t-gradient
    if(z < (c_d - 1)) {
        dvBar[idt] = vBar[id + c_size2] - vBar[id];
    } else {
        dvBar[idt] = 0.0f;
    }
}


////////////////////////////////////////////////////////////////////////////////
// Projection functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// called with width * height * depth many threads
__device__ void project_phi_TV(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id)    // current id3
{
    const size_t idx = id;
    const size_t idy = idx + c_size3;
    const size_t idt = idy + c_size3;

    //  add dataterm offset to phit
    const float dataTerm = data[id];
    phi[idt] += dataTerm;

    //  phi should not be projected, if it already is in K
    //  ! FIXED !
    const float qx = phi[idx];
    const float qy = phi[idy];
    const float qt = phi[idt];
    const float qdot = qx * qx + qy * qy;

    const bool phiInK = (qt >= 0 && qdot <= 1);

    if(phiInK == false) 
    { 
        //  pre-fetch and -compute for efficience
        const float qxClamp = MAX(1.0f, sqrtf(qx * qx + qy * qy));

        // projection formula
        phi[idx] /= qxClamp;
        phi[idy] /= qxClamp;
        phi[idt] = MAX(0.0f, qt);
    }

    phi[idt] -= dataTerm;
}

////////////////////////////////////////////////////////////////////////////////
// called with width * height * depth many threads
__device__ void project_phi_Quadratic(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id)    // current id3
{
    const size_t idx = id;
    const size_t idy = idx + c_size3;
    const size_t idt = idy + c_size3;

    //  add dataterm offset to phit
    const float dataTerm = data[id];
    phi[idt] += dataTerm;

    //  phi should not be projected, if it already is in K
    //  ! FIXED !
    const float qx = phi[idx];
    const float qy = phi[idy];
    const float qt = phi[idt];
    const float qdot = qx * qx + qy * qy;

    const bool phiInK = (qt >= qdot / 2.0f);

    if(phiInK == false) 
    { 
        //  pre-fetch and -compute for efficience
        const float lambda = getLambda(phi, id);

        // projection formula
        phi[idx] /= 1 + lambda;
        phi[idy] /= 1 + lambda;
        phi[idt] += lambda;
    }

    phi[idt] -= dataTerm;
}


////////////////////////////////////////////////////////////////////////////////
// called with width * height * depth many threads
__device__ void project_phi_Huber(
    const float* data,  // width * height * depth
    float* phi,         // width * height * depth * 3 in form [qx, qy, qt]
    const float alpha,  // constant > 0
    const size_t id)    // current id3
{
    const size_t idx = id;
    const size_t idy = idx + c_size3;
    const size_t idt = idy + c_size3;

    //  add dataterm offset to phit
    const float dataTerm = data[id];
    phi[idt] += dataTerm;

    //  phi should not be projected, if it already is in K
    //  ! FIXED !
    const float qx = phi[idx];
    const float qy = phi[idy];
    const float qt = phi[idt];
    const float qdot = qx * qx + qy * qy;

    const bool phiInK = (qt >= (alpha / 2.0f) * qdot && qdot <= 1);

    if(phiInK == false) 
    { 
       //  pre-fetch and -compute for efficience
        const float qxAbs = sqrtf(qdot);
        const float qxClamp = MAX(1.0f, qxAbs);

        // projection formula
        if((alpha / 2.0f) <= qt) {
            phi[idx] /= qxClamp;
            phi[idy] /= qxClamp;
        } 
        else 
        {
            const float proj = (alpha / 2.0f) - (1.0f / alpha) * (qxAbs - 1.0f);

            if(proj <= qt && qt < (alpha / 2.0f)) {
                phi[idx] /= qxClamp;
                phi[idy] /= qxClamp;
                phi[idt] = (alpha / 2.0f);
            } 
            else if(qt < proj) 
            {
                //  ! FIXED !
                const float lambda = getLambda(phi, id);

                phi[idx] /= (1.0f + lambda);
                phi[idy] /= (1.0f + lambda);
                phi[idt] += lambda;
            }
        }
    }

    phi[idt] -= dataTerm;
}


////////////////////////////////////////////////////////////////////////////////
// called with width * height * depth many threads
__device__ void project_v(
    float* v,           // width * height * dpeth
    const size_t id)    // current id3

{
    int x, y, z;
    getId(&x, &y, &z);

    // set lowest level 1 and highest 0
    if(z == 0) {
        v[id] = 1.0f;
    } 
    //  ! FIXED !
    else if(z == c_d - 1) {
        v[id] = 0.0f;
    }
    // projection is clipping
    else if(v[id] < 0.0f) {
        v[id] = 0.0f;
    } 
    else if(v[id] > 1.0f) {
        v[id] = 1.0f;
    }
}


////////////////////////////////////////////////////////////////////////////////
__device__ float getLambda(
    const float* phi,   // width * height * depth * 3 in form [qx, qy, qt]
    const size_t id)    // current id3
{
    const float qx = phi[id];
    const float qy = phi[id + c_size3];
    const float qt = phi[id + 2 * c_size3];

    const float qdot = qx * qx + qy * qy;

    //  ! FIXED !
    float lambda = MAX(0.0f, -(2.0f * qt + 1.0f) / 3.0f) + 1.0f;

    while (true) 
    // for (int i = 0; i < 5; i++)
    {
        const float lambda2 = lambda * lambda;
        const float nominator = 
            lambda2 * lambda + 
            lambda2 * (qt + 2.0f) + 
            lambda * (2.0f * qt + 1.0f) + 
            qt - qdot / 2.0f;

        const float denominator = 
            3.0f * lambda2 +
            2.0f * lambda * (qt + 2.0f) +
            2.0f * qt + 1.0f;

        const float dlambda = nominator / denominator;

        //  ! FIXED !
        if(abs(dlambda) < 0.001f) {
            return lambda;
        }

        lambda -= dlambda;
    }   

    return lambda;
}


////////////////////////////////////////////////////////////////////////////////
//  getter and helper functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// thresholds v with theta (without changing v) and writes depth map into depth
__global__ void getDepthMap(
    const float* v,     // width * height * depth
    int* disp,          // width * height
    const float theta   // constant \in [0, 1]
) {
    int x, y, z;
    getId(&x, &y, &z);

    if(x < c_w && y < c_h && z < c_d) 
    {
        const size_t id2 = x + y * c_w;
        const size_t id3 = id2 + z * c_size2;

        const int thresh = (v[id3] >= theta) ? 1 : 0;
    
        atomicAdd(disp + id2, thresh);
    }
}


////////////////////////////////////////////////////////////////////////////////
__device__ void getId(
    int *x,
    int *y,
    int *z
) {
    *x = threadIdx.x + blockDim.x * blockIdx.x;
    *y = threadIdx.y + blockDim.y * blockIdx.y;

#ifdef ZORAH
    *z = threadIdx.z;
#else
    *z = threadIdx.z + blockDim.z * blockIdx.z;
#endif
}