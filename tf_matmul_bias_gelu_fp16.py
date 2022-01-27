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

# We use the graph mode (tf.function), since the gelu graph is optimized.
@tf.function
def block(x, w, b):
  y = tf.linalg.matmul(x, w)
  z = tf.nn.bias_add(y, b)
  z = tf.keras.activations.gelu(z)
  return z

# Warmup
repeats = 10
for i in range(repeats):
  z = block(x, w, b)
  #y = tf.linalg.matmul(x, w)
  #z = tf.nn.bias_add(y, b)
  #z = tf.keras.activations.gelu(z)
# Sync
out = tf.math.reduce_sum(z)
_ = out.numpy()

start = time.time()
repeats = 50
for i in range(repeats):
  z = block(x, w, b)
  #y = tf.linalg.matmul(x, w)
  #z = tf.nn.bias_add(y, b)
  #z = tf.keras.activations.gelu(z)
# Sync
out = tf.math.reduce_sum(z)
_ = out.numpy()
end = time.time()
time_in_ms = (end - start) / repeats * 1000

print('x.dtype: %s' % x.dtype.name)
print('y.dtype: %s' % z.dtype.name)
print("LOG >>> Execution Time (ms):", time_in_ms)

