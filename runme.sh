
make

declare -a test_cmds=("./cudnn_v8_matmul_bias_gelu_fp16.out 0"
                      "python tf_matmul_bias_gelu_fp16.py"
                      "./cudnn_v8_matmul_bias_fp16.out 0"
                      "python tf_matmul_bias_fp16.py")

for test_cmd in "${test_cmds[@]}"
do
  for ((m = 10; m < 15; m++))
  do
    for ((k = 10; k < 15; k++))
    do
      for ((n = 10; n < 15; n++))
      do
        M=$(( 2 ** $m ))
        K=$(( 2 ** $k ))
        N=$(( 2 ** $n ))
        $test_cmd $M $K $N
      done
    done
  done
done
                      

