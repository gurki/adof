#  cuda
CUDA_PATH = /Developer/NVIDIA/CUDA-5.5
CUDA_LIBS = \
    -L$(CUDA_PATH)/lib
CUDA_INCS = \
    -I$(CUDA_PATH)/include

#  append
LIBS += $(CUDA_LIBS)
INCS += $(CUDA_INCS)