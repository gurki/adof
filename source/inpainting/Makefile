include ../../common/defaults.mk
include ../../common/opencv.mk
include ../../common/sfml.mk
include ../../common/cuda.mk


#  files
FILES += \
	Inpainting.cu \
	InpaintingKernels.cu \
	main.cu

default: main

main: 
	nvcc -o main $(FILES) $(LIBS) $(INCS) $(OPTS)