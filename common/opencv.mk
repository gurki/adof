#  opencv
OPENCV_PATH = /opt/local
OPENCV_LIBS = \
    -L$(OPENCV_PATH)/lib \
    -lopencv_core \
    -lopencv_highgui \
    -lopencv_imgproc \
    -lopencv_video \
    -lopencv_objdetect
OPENCV_INCS = \
    -I$(OPENCV_PATH)/include

#  append
LIBS += $(OPENCV_LIBS)
INCS += $(OPENCV_INCS)