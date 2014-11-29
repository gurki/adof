include ../../common/defaults.mk
include ../../common/opencv.mk
include ../../common/cuda.mk
include ../../common/qt.mk


#  files
FILES += \
	main.cpp \
	stereodepthmap.cpp \
	moc_stereodepthmap.cpp \
	../disparity/DisparityEstimator.cu \
	../disparity/DisparityEstimatorKernels.cu \
	../diffusion/LinearDiffusion.cu \
	../diffusion/LinearDiffusionKernels.cu \
	../inpainting/Inpainting.cu \
	../inpainting/InpaintingKernels.cu

default: main

main: 
	nvcc -o main $(FILES) $(LIBS) $(INCS) $(OPTS)