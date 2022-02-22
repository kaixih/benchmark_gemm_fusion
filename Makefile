CUDNN_FLAGS=-I /home/cudnn_frontend/include/ \
					  -DNV_CUDNN_DISABLE_EXCEPTION -lcudnn
#CXXFLAGS=-DDEBUG_MODE

all: cudnn_v8_matmul_bias_gelu_fp16.out \
	   cudnn_v8_matmul_bias_fp16.out \
	   cublaslt_matmul_bias_gelu_fp16.out \
	   cublaslt_matmul_bias_fp16.out

cudnn_v8_matmul_bias_gelu_fp16.out: cudnn_v8_matmul_bias_gelu_fp16.cpp
	nvcc $< -o $@ ${CUDNN_FLAGS} ${CXXFLAGS}

cudnn_v8_matmul_bias_fp16.out: cudnn_v8_matmul_bias_fp16.cpp
	nvcc $< -o $@ ${CUDNN_FLAGS} ${CXXFLAGS}

cublaslt_matmul_bias_gelu_fp16.out: cublaslt_matmul_bias_gelu_fp16.cpp
	nvcc $< -o $@ -lcublasLt ${CXXFLAGS}

cublaslt_matmul_bias_fp16.out: cublaslt_matmul_bias_fp16.cpp
	nvcc $< -o $@ -lcublasLt ${CXXFLAGS}

clean:
	rm -rf *.out *.sqlite *.nsys-rep

