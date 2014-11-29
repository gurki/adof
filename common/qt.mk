#  QT
QT_PATH = /opt/local
QT_LIBS = \
    -L$(QT_PATH)/lib \
    -lQtCore \
    -lQtGui 
QT_INCS = \
    -I$(QT_PATH)/include

#  append
LIBS += $(QT_LIBS)
INCS += $(QT_INCS)