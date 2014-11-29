#include "InpaintingKernels.h"


////////////////
//  optimizations:
//  1. computations only at masked regions: + >50%
//  2. id check only in global: +8%
//  


#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)


////////////////////////////////////////////////////////////////////////////////
//  memory management
void set_constant_memory(
    const size_t* w, 
    const size_t* h, 
    const size_t* nc, 
    const size_t* size)
{
    cudaMemcpyToSymbol(d_w, w, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(d_h, h, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(d_nc, nc, sizeof(size_t)); CUDA_CHECK;
    cudaMemcpyToSymbol(d_size, size, sizeof(size_t)); CUDA_CHECK;
}


////////////////////////////////////////////////////////////////////////////////
//  inpainting
__global__ void inpaint_update_rb_sor(
    float* u, 
    float* g,
    const int* mask, 
    const int maskFlag,
    const bool red,
    const float theta)
{
    //  compute checkerboard index
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const bool evenRow = (y % 2 == 0);
    const int offset = ((evenRow ^ red) ? 0 : 1);
    const int x = 2 * (threadIdx.x + blockIdx.x * blockDim.x) + offset;
    
    const size_t id = x + d_w * y;

    //  check for valid id
    if (x < d_w && y < d_h)
    {
        //  compute update for masked regions only
        if (mask[id] == maskFlag)
        {
            //  compute neighbour ids
            const size_t idr = (x < d_w - 1) ? ((x + 1) + y * d_w) : id;
            const size_t idl = (x > 0)       ? ((x - 1) + y * d_w) : id;
            const size_t idu = (y < d_h - 1) ? (x + (y + 1) * d_w) : id;
            const size_t idd = (y > 0)       ? (x + (y - 1) * d_w) : id;

            //  compute neighbour diffusivities
            const float gr = ((x < d_w - 1) ? g[idr] : 0);
            const float gl = ((x > 0)       ? g[idl] : 0);
            const float gu = ((y < d_h - 1) ? g[idu] : 0);
            const float gd = ((y > 0)       ? g[idd] : 0);

            //  update channel-wise
            for (int ch = 0; ch < d_nc; ch++)
            {
                const size_t choff = ch * d_size;
                const size_t cid = id + choff;

                //  update step
                const float nom = 
                    gr * u[idr + choff] +
                    gl * u[idl + choff] +
                    gu * u[idu + choff] +
                    gd * u[idd + choff];
                const float den = gr + gl + gu + gd;
                const float val = nom / den;

                //  SOR update
                u[cid] = val + theta * (val - u[cid]);
            }
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__ void diffusivity_rb(
    const float* u, 
    const int* mask,
    const int maskFlag,
    float* g, 
    float* dx, 
    float* dy, 
    const float eps,
    const bool red)
{
    //  compute checkerboard index   
    const int y = threadIdx.y + blockIdx.y * blockDim.y;
    const bool evenRow = (y % 2 == 0);
    const int offset = ((evenRow ^ red) ? 0 : 1);
    const int x = 2 * (threadIdx.x + blockIdx.x * blockDim.x) + offset;

    const size_t id = x + d_w * y;

    //  check for valid id
    if (x < d_w && y < d_h)
    {
        //  compute neighbour ids
        const size_t idr = (x < d_w - 1) ? ((x + 1) + y * d_w) : id;
        const size_t idl = (x > 0)       ? ((x - 1) + y * d_w) : id;
        const size_t idu = (y < d_h - 1) ? (x + (y + 1) * d_w) : id;
        const size_t idd = (y > 0)       ? (x + (y - 1) * d_w) : id;

        // compute diffusivity only where needed (mask + 1px offset)
        if (mask[id] == maskFlag || 
            mask[idr] == maskFlag || 
            mask[idl] == maskFlag || 
            mask[idu] == maskFlag || 
            mask[idd] == maskFlag) 
        {
            //  compute gradient
            forward_differences(u, dx, dy, x, y, id);

            //  compute its magnitude
            magnitude(dx, dy, g, id);

            //  compute diff response
            diffusivity(g, eps, id);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__device__ void diffusivity(
    float* g, 
    const float eps, 
    const size_t id)
{
    //  assumes valid id
    g[id] = 1.0f / MAX(eps, g[id]);
}


////////////////////////////////////////////////////////////////////////////////
__device__ void magnitude(
    const float* dx, 
    const float* dy, 
    float* mag, 
    const size_t id) 
{
    //  assumes valid id
    float cmag = 0.0f;

    for(int ch = 0; ch < d_nc; ch++) 
    {
        const size_t cid = id + ch * d_size;
        const float cdx = dx[cid];
        const float cdy = dy[cid];

        cmag += cdx * cdx + cdy * cdy;
    }

    mag[id] = sqrtf(cmag);
}


////////////////////////////////////////////////////////////////////////////////
// calculates forward difference for in and steres result in out
__device__ void forward_differences(
    const float* u, 
    float* dx, 
    float* dy, 
    const int x,
    const int y,
    const size_t id) 
{
    //  assumes valid id
    for (int ch = 0; ch < d_nc; ch++)
    {
        const size_t cid = id + ch * d_size;
        const float u0 = u[cid];

        dx[cid] = (x < d_w - 1) ? (u[cid + 1] - u0) : 0;
        dy[cid] = (y < d_h - 1) ? (u[cid + d_w] - u0) : 0;
    }  
}