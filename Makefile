CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread
DEBUGFLAGS=-G -lineinfo

MODULE := conv2b

all: $(MODULE)

HEADERS=

$(MODULE): convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=2250 -DNy=1250 -DKx=7 -DKy=7 -DNi=3 -DNn=768

clean:
	rm -f $(MODULE)
