#ifndef INPAINTING_KERNELS_H
#define INPAINTING_KERNELS_H

#include "../../common/core/aux.h"


//  memory
__constant__ size_t d_w;
__constant__ size_t d_h;
__constant__ size_t d_nc;
__constant__ size_t d_size;


void set_constant_memory(
    const size_t* w, 
    const size_t* h, 
    const size_t* nc, 
    const size_t* size);


//  inpainting
__global__ void diffusivity_rb(
    const float* u, 
    const int* mask,
    const int maskFlag,
    float* g, 
    float* dx, 
    float* dy, 
    const float eps, 
    const bool red);

__global__ void inpaint_update_rb_sor(
    float* u0, 
    float* g, 
    const int* mask,
    const int maskFlag,
    const bool red,
    const float theta);

__device__ void diffusivity(
    float* g, 
    const float eps, 
    const size_t id);

__device__ void magnitude(
    const float* dx, 
    const float* dy, 
    float* mag, 
    const size_t id);

__device__ void forward_differences(
    const float* u, 
    float* dx, 
    float* dy, 
    const int x,
    const int y,
    const size_t id);


#endif