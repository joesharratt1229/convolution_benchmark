CPP=g++
NVCC=nvcc
CFLAGS=--std=c++11 -g --default-stream per-thread

MODULE := conv2b

all: $(MODULE)

HEADERS=

$(MODULE): convolution.cu $(HEADERS)
	$(NVCC) $^ $(CFLAGS) -o $@ -DNx=32 -DNy=32 -DKx=3 -DKy=3 -DNi=768 -DNn=512

clean:
	rm -f $(MODULE)
