#include "LinearDiffusionKernels.h"

//  benchmark: 10000 iterations average, cuda event timer
//  non-constant properties: 1.07721 ms
//  constant properties:     1.08089 ms
//  constant size/tau only:  1.06566 ms
//  constant diff. map:      1.07566 ms

//  no shared memory
//  dt: 1.23649 ms
//  dt: 1.24024 ms
//  dt: 1.24319 ms
//  shared memory, two ifs
//  dt: 2.21866 ms
//  shared memory, one if
//  dt: 2.21273 ms
//  shared memory, inline ifs
//  dt: 2.23558 ms


////////////////////////////////////////////////////////////////////////////////
__global__ void adaptive_intensity(
    float* u0,
    const int* disp,
    const float* diff,
    const dim3 dim) 
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t w = dim.x;
    const size_t h = dim.y;

    if (x < w && y < h) 
    {
        const size_t size = w * h;
        const size_t id = x + y * w;

        const float g = diff[disp[id]];

        for (int ch = 0; ch < dim.z; ch++)
        {
            const size_t cid = id + ch * size;

            u0[cid] *= 1.0f - g;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__ void adaptive_linear_diffusion(
    const float* u0,
    float* u1, 
    const int* disp,
    const float* diff,
    const dim3 dim,
    const float tau) 
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t w = dim.x;
    const size_t h = dim.y;

    if (x < w && y < h) 
    {
        const size_t size = w * h;
        const size_t id = x + y * w;

        //  kronecker delta
        const int kdr = ((x < w - 1) ? 1 : 0);
        const int kdl = ((x > 0)     ? 1 : 0);
        const int kdu = ((y < h - 1) ? 1 : 0);
        const int kdd = ((y > 0)     ? 1 : 0);

        //  neighbour ids
        const size_t idr = (x + kdr) + y * w;
        const size_t idl = (x - kdl) + y * w;
        const size_t idu = x + (y + kdu) * w;
        const size_t idd = x + (y - kdd) * w;

        //  get diffusifity
        const int d = disp[id];
        const float g = diff[d];

        //  diffuse channel-wise
        for (int ch = 0; ch < dim.z; ch++)
        {
            const size_t choff = ch * size;
            const size_t cid = id + choff;

            const float u0val = u0[cid];
 
             //  laplace update step
            const float min = (
                kdr * u0[idr + choff] +
                kdl * u0[idl + choff] +
                kdu * u0[idu + choff] +
                kdd * u0[idd + choff]
            );
            
            const float sub = (kdr + kdl + kdu + kdd) * u0val;

            u1[cid] = u0val + g * tau * (min - sub);
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__ void linear_diffusion(
    const float* u0,
    float* u1, 
    const float tau, 
    const dim3 dim) 
{
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;

    const size_t w = dim.x;
    const size_t h = dim.y;

    const size_t size = w * h;
    const size_t id = x + y * w;

    if (x < dim.x && y < dim.y) 
    {
        //  kronecker delta
        const int kdr = ((x < w - 1) ? 1 : 0);
        const int kdl = ((x > 0)     ? 1 : 0);
        const int kdu = ((y < h - 1) ? 1 : 0);
        const int kdd = ((y > 0)     ? 1 : 0);

        //  neighbour ids
        const size_t idr = (x + kdr) + y * w;
        const size_t idl = (x - kdl) + y * w;
        const size_t idu = x + (y + kdu) * w;
        const size_t idd = x + (y - kdd) * w;

        //  diffuse channel-wise
        for (int ch = 0; ch < dim.z; ch++)
        {
            const size_t choff = ch * size;
            const size_t cid = id + choff;
            const float u0val = u0[cid];

            //  laplace update step
            const float min = (
                kdr * u0[idr + choff] +
                kdl * u0[idl + choff] +
                kdu * u0[idu + choff] +
                kdd * u0[idd + choff]
            );

            const float sub = (kdr + kdl + kdu + kdd) * u0val;

            u1[cid] = u0val + tau * (min - sub);
        }
    }
}