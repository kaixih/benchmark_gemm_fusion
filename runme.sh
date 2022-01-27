CMD="./cudnn_v8_matmul_bias_gelu_fp16.out 0"
#CMD="python tf_matmul_bias_gelu_fp16.py"
#CMD="./cudnn_v8_matmul_bias_fp16.out 0"
#CMD="python tf_matmul_bias_fp16.py"
for ((m = 10; m < 15; m++))
do
  for ((k = 10; k < 15; k++))
  do
    for ((n = 10; n < 15; n++))
    do
      M=$(( 2 ** $m ))
      K=$(( 2 ** $k ))
      N=$(( 2 ** $n ))
      $CMD $M $K $N
    done
  done
done

