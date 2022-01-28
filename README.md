# Benchmark the Gemm fusion
This repo is designed to benchmark the Gemm fusion on GPUs. The baseline the Tensorflow implementations.
The target fusion patterns are:
1. MatMul + BiasAdd
2. MatMul + BiasAdd + Gelu
