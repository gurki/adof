#  sfml
SFML_PATH = /usr/local
SFML_LIBS = \
	-L$(SFML_PATH)/lib \
	-lsfml-audio \
	-lsfml-graphics \
	-lsfml-network \
	-lsfml-system \
	-lsfml-window
SFML_INCS = \
    -I$(SFML_PATH)/include
SFML_OPTS = \
    -Xlinker -framework,OpenGL,-framework,GLUT

#  append
LIBS += $(SFML_LIBS)
INCS += $(SFML_INCS)
OPTS += $(SFML_OPTS)