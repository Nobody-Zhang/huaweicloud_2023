CUDA_VER=10.2
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

CXX=g++

APP:= deepstream-app-test5-customized

TARGET_DEVICE = $(shell gcc -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=6.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/

ifeq ($(TARGET_DEVICE),aarch64)
  CFLAGS:= -DPLATFORM_TEGRA
endif

SRCS:= test.cpp deepstream_utc.c
SRCS+= /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app.c /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/deepstream_app_config_parser.c
SRCS+= $(wildcard /opt/nvidia/deepstream/deepstream/sources/apps/apps-common/src/*.c)

INCS:= $(wildcard *.h)

PKGS:= gstreamer-1.0 gstreamer-video-1.0 x11 json-glib-1.0 opencv

OBJS:= $(patsubst %.c,%.o, $(patsubst %.cpp,%.o, $(SRCS)))



CFLAGS+= -I/opt/nvidia/deepstream/deepstream/sources/apps/apps-common/includes \
           -I/opt/nvidia/deepstream/deepstream/sources/includes \
           -I/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-app/ -DDS_VERSION_MINOR=0 -DDS_VERSION_MAJOR=5 \
           -I /usr/local/cuda-$(CUDA_VER)/include -std=c++11 \
	   -I /usr/local/include/gstreamer-1.0

LIBS:= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart

LIBS+= -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvdsgst_helper -lnvdsgst_smartrecord -lnvds_utils -lnvds_msgbroker -lm -lnvbufsurface -lnvbufsurftransform\
       -lcuda -pthread -lgstrtspserver-1.0 -lcurl -ljsoncpp -ldl -Wl,-rpath,$(LIB_INSTALL_DIR)

CFLAGS+= $(shell pkg-config --cflags $(PKGS))\

LIBS+= $(shell pkg-config --libs $(PKGS))
LIBS+= -L/usr/local/lib/ -lssl -lcrypto

all: $(APP)

%.o: %.c $(INCS) Makefile
	gcc  -c -o $@ $(CFLAGS)  $<

%.o: %.cpp $(INCS) Makefile
	$(CXX)   -c -o $@ $(CFLAGS)$<

deepstream_utc.o: deepstream_utc.c $(INCS) Makefile
	gcc  -c -o $@ $(CFLAGS)  $<

$(APP): $(OBJS) Makefile
	$(CXX) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)




