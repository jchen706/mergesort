NVCC=nvcc

all:
	$(NVCC) mergesort.cu
	
clean: 
	rm *.out 