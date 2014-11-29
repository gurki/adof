#ifndef LINEAR_DIFFUSION_KERNELS_H
#define LINEAR_DIFFUSION_KERNELS_H


__global__ void adaptive_intensity(
    float* u0,
    const int* disp,
    const float* diff,
    const dim3 dim);

__global__ void adaptive_linear_diffusion(
    const float* u0,
    float* u1, 
    const int* disp,
    const float* diff,
    const dim3 dim,
    const float tau);

__global__ void linear_diffusion(
    const float* u0, 
    float* u1, 
    const float tau, 
    const dim3 dim);


#endif