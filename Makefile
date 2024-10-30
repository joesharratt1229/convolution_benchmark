CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread
INCFLAGS=-I$(SRCDIR)
DEBUGFLAGS=-G -lineinfo
REGFLAGS=-maxrregcount=30
NVFLAGS=--use_fast_math -arch=sm_80

MODULE := conv2b

SRCDIR := src
OBJDIR := obj

all: $(MODULE)

HEADERS=

SOURCES := $(SRCDIR)/main.cu 

$(MODULE): $(SOURCES) $(HEADERS)
	$(NVCC) $^ $(CFLAGS) $(REGFLAGS) $(NVFLAGS) $(INCFLAGS) -o $@ -DNx=128 -DNy=128 -DKx=7 -DKy=7 -DNi=3 -DNn=16 -DENABLE_BP16


debug_build: CFLAGS += $(DEBUGFLAGS)
debug_build: NVFLAGS += -G --device-debug
debug_build: $(MODULE)

clean:
	rm -f $(MODULE) $(OBJDIR)/*.o