CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread
INCFLAGS=-I$(SRCDIR)
DEBUGFLAGS=-G -lineinfo
REGFLAGS=-maxrregcount=30
NVFLAGS=--use_fast_math

MODULE := conv2b

SRCDIR := src
OBJDIR := obj

all: $(MODULE)

HEADERS=

$(MODULE): $(SRCDIR)/main.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) $(REGFLAGS) $(NVFLAGS) $(INCFLAGS) -o $@ -DNx=2250 -DNy=1250 -DKx=7 -DKy=7 -DNi=3 -DNn=16 -DENABLE_FP32


debug_build: CFLAGS += $(DEBUGFLAGS)
debug_build: NVFLAGS += -G --device-debug
debug_build: $(MODULE)

clean:
	rm -f $(MODULE) $(OBJDIR)/*.o
