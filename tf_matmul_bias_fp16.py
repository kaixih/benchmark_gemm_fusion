import argparse
import tensorflow as tf
import time

from keras.layers import Dense
from tensorflow.keras import mixed_precision

parser = argparse.ArgumentParser(description='Benchmark Gemm+Bias')
parser.add_argument('dims', metavar='N', type=int, nargs='*',
                    help='integers representing M, K, N', default=[64, 32, 64])
args = parser.parse_args()
print("LOG >>> M, K, N:", args.dims)

x = tf.random.normal(shape=(args.dims[0], args.dims[1]), dtype=tf.float16)
w = tf.random.normal(shape=(args.dims[1], args.dims[2]), dtype=tf.float16)
b = tf.random.normal(shape=(args.dims[2],), dtype=tf.float16)

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

