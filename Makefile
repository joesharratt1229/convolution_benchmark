CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread
DEBUGFLAGS=-G -lineinfo
REGFLAGS=-maxrregcount=30

MODULE := conv2b

all: $(MODULE)

HEADERS=

debug: CFLAGS += $(DEBUGFLAGS)
debug: $(MODULE)

$(MODULE): convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) $(REGFLAGS) -o $@ -DNx=2250 -DNy=1250 -DKx=7 -DKy=7 -DNi=3 -DNn=32

debug_build: convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) $(REGFLAGS) -o $@ -DNx=2250 -DNy=1250 -DKx=7 -DKy=7 -DNi=3 -DNn=768 -G -lineinfo

clean:
	rm -f $(MODULE)
