#include <assert.h>
#include <cublasLt.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <iostream>

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define checkCUBLASLT(expression)                            \
{                                                            \
  cublasStatus_t status = (expression);                      \
  if (status != CUBLAS_STATUS_SUCCESS) {                     \
    std::cerr << "Error on line " << __LINE__ << ": "        \
              << cublasLtGetStatusName(status) << " ("       \
              << cublasLtGetStatusString(status) << ")\n";   \
    std::exit(EXIT_FAILURE);                                 \
  }                                                          \
}

#define FLOAT_T __half

void init_input(FLOAT_T *ptr, int size) {
  FLOAT_T* ptr_host = new FLOAT_T[size];
  for (int i = 0; i < size; i++) {
    float val = static_cast<float>(rand()) / RAND_MAX;
    ptr_host[i]  = static_cast<FLOAT_T>(val);
  }
  checkCUDA(cudaMemcpy(ptr, ptr_host, sizeof(FLOAT_T) * size,
                       cudaMemcpyHostToDevice));
  delete[] ptr_host;
}

void print_output(const FLOAT_T* ptr, int size, const char* message,
                  int lines = 10) {
  checkCUDA(cudaDeviceSynchronize());
  FLOAT_T* ptr_host = new FLOAT_T[size];
  checkCUDA(cudaMemcpy(ptr_host, ptr, sizeof(FLOAT_T) * size,
                       cudaMemcpyDeviceToHost));

  const int num_per_line = 20;
  int limit = INT_MAX;
  if (lines != -1) {
    limit = lines * num_per_line;
  }

  printf("%s (showing %d elements):\n", message, std::min(size, limit));
  for (int i = 0; i < std::min(size, limit); i++) {
    printf("%f, ", static_cast<float>(ptr_host[i]));
    if ((i + 1) % num_per_line == 0) {
      printf("\n");
    }
  }
  printf("\n");
  delete[] ptr_host;
}

int main(int argc, char **argv) {
  int plan_idx = 0;
  if (argc > 1) {
    plan_idx = atoi(argv[1]);
  }

  int M = 64;
  int K = 32;
  int N = 64;
  if (argc > 4) {
    M = atoi(argv[2]);
    K = atoi(argv[3]);
    N = atoi(argv[4]);
  }
  printf("LOG >>> M, K, N: %d, %d, %d\n", M, K, N);

  int a_size = M * K;
  int b_size = K * N;
  int c_size = M * N;
  int z_size = 1 * N; // bias

  int a_bytes = a_size * sizeof(FLOAT_T);
  int b_bytes = b_size * sizeof(FLOAT_T);
  int c_bytes = c_size * sizeof(FLOAT_T);
  int z_bytes = z_size * sizeof(FLOAT_T);

  FLOAT_T *a;
  FLOAT_T *b;
  FLOAT_T *c;
  FLOAT_T *z;
  checkCUDA(cudaMalloc((void**)&a, a_bytes));
  checkCUDA(cudaMalloc((void**)&b, b_bytes));
  checkCUDA(cudaMalloc((void**)&c, c_bytes));
  checkCUDA(cudaMalloc((void**)&z, z_bytes));

  srand(3);
  init_input(a, a_size);
  init_input(b, b_size);
  init_input(c, c_size);

  cudaDataType_t dataType = CUDA_R_16F;
  cudaDataType_t scaleType = CUDA_R_32F;
  cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
  cublasLtEpilogue_t epilogType = CUBLASLT_EPILOGUE_GELU_BIAS;

  cublasLtHandle_t cublaslt;
  checkCUBLASLT(cublasLtCreate(&cublaslt));

  cublasLtMatmulDesc_t matmul_desc;
  checkCUBLASLT(cublasLtMatmulDescCreate(&matmul_desc, computeType, scaleType));
  checkCUBLASLT(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &z, sizeof(z)));
  checkCUBLASLT(cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogType,
      sizeof(epilogType)));

  cublasLtMatrixLayout_t a_desc;
  checkCUBLASLT(cublasLtMatrixLayoutCreate(&a_desc, dataType, M, K, M));
  cublasLtMatrixLayout_t b_desc;
  checkCUBLASLT(cublasLtMatrixLayoutCreate(&b_desc, dataType, K, N, K));
  cublasLtMatrixLayout_t c_desc;
  checkCUBLASLT(cublasLtMatrixLayoutCreate(&c_desc, dataType, M, N, M));

  size_t workspace_size = 1 << 30; // 1 GB
  printf("LOG >>> Max workspace size (bytes): %ld\n", workspace_size);
  cublasLtMatmulPreference_t preference;
  checkCUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
  checkCUBLASLT(cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size,
      sizeof(workspace_size)));

  int num_returned_results = 0;
  cublasLtMatmulHeuristicResult_t* heuristics =
      new cublasLtMatmulHeuristicResult_t[plan_idx + 1];
  checkCUBLASLT(cublasLtMatmulAlgoGetHeuristic(
      cublaslt, matmul_desc, a_desc, b_desc, c_desc, c_desc, preference,
      plan_idx + 1, heuristics, &num_returned_results));
  printf("LOG >>> Number of requested results: %d\n", plan_idx + 1);
  printf("LOG >>> Number of returned results: %d\n", num_returned_results);

  assert(num_returned_results > 0);

  size_t actual_workspace_size = heuristics[0].workspaceSize;
  printf("LOG >>> Actual workspace size (bytes): %ld\n",
         actual_workspace_size);

	void* d_workspace{nullptr};
  if (actual_workspace_size != 0) {
    checkCUDA(cudaMalloc(&d_workspace, actual_workspace_size));
  }

  const float alpha = 1.0f;
  const float beta = 0.0f;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warmup
  checkCUBLASLT(cublasLtMatmul(
      cublaslt, matmul_desc, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc, c,
      c_desc, &heuristics[0].algo, d_workspace, actual_workspace_size, 0));

  cudaEventRecord(start);
  const int num_repeats = 50;
  for (int i = 0; i < num_repeats; i++) {
    checkCUBLASLT(cublasLtMatmul(
        cublaslt, matmul_desc, &alpha, a, a_desc, b, b_desc, &beta, c, c_desc,
        c, c_desc, &heuristics[0].algo, d_workspace, actual_workspace_size, 0));
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("LOG >>> Execution Time (ms): %f\n", milliseconds / num_repeats);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  print_output(c, c_size, "c out:", 1);

  checkCUDA(cudaFree(a));
  checkCUDA(cudaFree(b));
  checkCUDA(cudaFree(c));
  if (actual_workspace_size != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
  delete[] heuristics;

  checkCUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
  checkCUBLASLT(cublasLtMatrixLayoutDestroy(c_desc));
  checkCUBLASLT(cublasLtMatrixLayoutDestroy(b_desc));
  checkCUBLASLT(cublasLtMatrixLayoutDestroy(a_desc));
  checkCUBLASLT(cublasLtMatmulDescDestroy(matmul_desc));
  checkCUBLASLT(cublasLtDestroy(cublaslt));
}
