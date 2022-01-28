CUDNN_FRONTEND_DIR=/home/cudnn_frontend/include/
CXXFLAGS=-DNV_CUDNN_DISABLE_EXCEPTION -lcudnn

all: cudnn_v8_matmul_bias_gelu_fp16.out \
	   cudnn_v8_matmul_bias_fp16.out \
	   cublaslt_matmul_bias_gelu_fp16.out \
	   cublaslt_matmul_bias_fp16.out

cudnn_v8_matmul_bias_gelu_fp16.out: cudnn_v8_matmul_bias_gelu_fp16.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_matmul_bias_fp16.out: cudnn_v8_matmul_bias_fp16.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cublaslt_matmul_bias_gelu_fp16.out: cublaslt_matmul_bias_gelu_fp16.cpp
	nvcc $< -o $@ -lcublasLt

cublaslt_matmul_bias_fp16.out: cublaslt_matmul_bias_fp16.cpp
	nvcc $< -o $@ -lcublasLt

clean:
	rm -rf *.out *.sqlite *.nsys-rep

