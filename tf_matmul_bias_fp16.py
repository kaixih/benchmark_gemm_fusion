import tensorflow as tf
import time

configs = []
for M_e in range(10, 15):
  for K_e in range(10, 15):
    for N_e in range(10, 15):
      configs.append((2**M_e, 2**K_e, 2**N_e));

for M, K, N in configs:
  print("LOG >>> M, K, N:", M, K, N)
  x = tf.random.normal((M, K), dtype=tf.float16)
  w = tf.random.normal((K, N), dtype=tf.float16)
  b = tf.random.normal((N, ), dtype=tf.float16)

  def matmal_bias(x, w, b):
    y = tf.linalg.matmul(x, w)
    z = tf.nn.bias_add(y, b)
    return z

  def benchmark(repeats):
    for i in range(repeats):
      z = matmal_bias(x, w, b)
    # Sync with a small D2H
    out = tf.math.reduce_sum(z)
    _ = out.numpy()

  burn_iters = 10
  time_iters = 50
  # Warmup
  benchmark(burn_iters)

  start = time.time()
  benchmark(time_iters)
  end = time.time()
  time_in_ms = (end - start) / time_iters * 1000

  print("LOG >>> Execution Time (ms):", time_in_ms)

