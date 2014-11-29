include ../../common/defaults.mk
include ../../common/opencv.mk
include ../../common/cuda.mk


#  files
FILES += \
	main.cu \
	DisparityEstimator.cu \
	DisparityEstimatorKernels.cu

default: main

main: 
	nvcc -o main $(FILES) $(LIBS) $(INCS) $(OPTS)