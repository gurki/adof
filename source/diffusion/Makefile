include ../../common/defaults.mk
include ../../common/opencv.mk
include ../../common/sfml.mk
include ../../common/cuda.mk


#  files
FILES += \
	LinearDiffusion.cu \
	LinearDiffusionKernels.cu \
	main.cu

default: main

main: 
	nvcc -o main $(FILES) $(LIBS) $(INCS) $(OPTS)