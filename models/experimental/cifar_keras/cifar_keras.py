# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cifar example using Keras for model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
import numpy as np
import os


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')

# Model specific paramenters
flags.DEFINE_integer("batch_size", 128,
                     "Mini-batch size for the computation. Note that this "
                     "is the global batch size and not the per-shard batch.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
flags.DEFINE_string("train_file", "", "Path to cifar10 training data.")
flags.DEFINE_integer("train_steps", 100000,
                     "Total number of steps. Note that the actual number of "
                     "steps is the next multiple of --iterations greater "
                     "than this value.")
flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_string("model_dir", None, "Estimator model_dir")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations per TPU training loop.")
flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")


FLAGS = flags.FLAGS


def model_fn(features, labels, mode, params):
  """Define a CIFAR model in Keras."""
  del params  # unused
  layers = tf.keras.layers

  # Pass our input tensor to initialize the Keras input layer.
  v = layers.Input(tensor=features)
  v = layers.Conv2D(filters=32, kernel_size=5,
                    activation="relu", padding="same")(v)
  v = layers.MaxPool2D(pool_size=2)(v)
  v = layers.Conv2D(filters=64, kernel_size=5,
                    activation="relu", padding="same")(v)
  v = layers.MaxPool2D(pool_size=2)(v)
  v = layers.Flatten()(v)
  fc1 = layers.Dense(units=512, activation="relu")(v)
  logits = layers.Dense(units=10)(fc1)

  # Instead of constructing a Keras model for training, build our loss function
  # and optimizer in Tensorflow.
  #
  # N.B.  This construction omits some features that are important for more
  # complex models (e.g. regularization, batch-norm).  Once
  # `model_to_estimator` support is added for TPUs, it should be used instead.
  loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(
          logits=logits, labels=labels
      )
  )
  optimizer = tf.compat.v1.train.AdamOptimizer()
  if FLAGS.use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

  return tf.contrib.tpu.TPUEstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      predictions={
          "classes": tf.argmax(input=logits, axis=1),
          "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      }
  )


def input_fn(params):
  """Read CIFAR input data from a TF Dataset API"""
  del params
  batch_size = FLAGS.batch_size
  
  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
  
  x_train = (x_train / 255.0) - 0.5
  x_train, y_train = x_train.astype(np.float32), y_train.astype(np.float32)
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  
  dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\
    .shuffle(2000)\
    .prefetch(4 * batch_size).cache().repeat()\
    .batch(batch_size, drop_remainder=True)\
    .prefetch(1)
				          
  return dataset


def main(argv):
  del argv  # Unused.

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      'grpc://' + os.environ['COLAB_TPU_ADDR'])

  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_secs=3600,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_shards),
  )

  estimator = tf.contrib.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      train_batch_size=FLAGS.batch_size)
  estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  app.run(main)
