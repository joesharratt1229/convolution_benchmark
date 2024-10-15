CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread
DEBUGFLAGS=-G -lineinfo
REGFLAGS=-maxrregcount=30
NVFLAGS=--use_fast_math

MODULE := conv2b

SRCDIR := src
OBJDIR := obj

all: $(MODULE)

HEADERS=

debug: CFLAGS += $(DEBUGFLAGS)
debug: $(MODULE)

$(MODULE): $(SRCDIR)/convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) $(REGFLAGS) $(NVFLAGS) -o $@ -DNx=2250 -DNy=1250 -DKx=7 -DKy=7 -DNi=3 -DNn=768 -DENABLE_BP16


debug_build: CFLAGS += $(DEBUGFLAGS)
debug_build: $(MODULE)

clean:
	rm -f $(MODULE) $(OBJDIR)/*.o
