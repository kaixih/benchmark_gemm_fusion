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

# Warmup
repeats = 10
for i in range(repeats):
  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)
# Sync
out = tf.math.reduce_sum(z)
_ = out.numpy()

start = time.time()
repeats = 50
for i in range(repeats):
  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)

# Sync
out = tf.math.reduce_sum(z)
_ = out.numpy()
end = time.time()
time_in_ms = (end - start) / repeats * 1000

print('x.dtype: %s' % x.dtype.name)
print('y.dtype: %s' % z.dtype.name)
print("LOG >>> Execution Time (ms):", time_in_ms)

