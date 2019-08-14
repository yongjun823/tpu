# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Example for using Keras Application models using TPU Strategy."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import numpy as np
import tensorflow as tf
import os


# Define a dictionary that maps model names to their model classes inside Keras
MODELS = {
    "vgg16": tf.keras.applications.VGG16,
    "vgg19": tf.keras.applications.VGG19,
    "inceptionv3": tf.keras.applications.InceptionV3,
    "xception": tf.keras.applications.Xception,
    "resnet50": tf.keras.applications.ResNet50,
    "inceptionresnetv2": tf.keras.applications.InceptionResNetV2,
    "mobilenet": tf.keras.applications.MobileNet,
    "densenet121": tf.keras.applications.DenseNet121,
    "densenet169": tf.keras.applications.DenseNet169,
    "densenet201": tf.keras.applications.DenseNet201,
    "nasnetlarge": tf.keras.applications.NASNetLarge,
    "nasnetmobile": tf.keras.applications.NASNetMobile,
}

flags.DEFINE_enum("model", None, MODELS.keys(), "Name of the model to be run",
                  case_sensitive=False)
flags.DEFINE_integer("batch_size", 256 * 8, "Batch size to be used for model")
flags.DEFINE_integer("epochs", 100, "Number of training epochs")
flags.DEFINE_string("mode", 'GPU', "train GPU OR TPU")

FLAGS = flags.FLAGS


class Cifar10Dataset(object):
  """CIFAR10 dataset, including train and test set.
  Each sample consists of a 32x32 color image, and label is from 10 classes.
  Note: Some models such as Xception require larger images than 32x32 so one
  needs to write a tf.data.dataset for Imagenet or use synthetic data.
  """

  def __init__(self, batch_size):
    """Initializes train/test datasets.
    Args:
      batch_size: int, the number of batch size.
    """
    self.input_shape = (32, 32, 3)
    self.num_classes = 10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    self.num_train_images = len(x_train)
    self.num_test_images = len(x_test)

    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    
    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

    self.train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
                          .repeat()
                          .shuffle(2000)
                          .batch(batch_size, drop_remainder=True))
    self.test_dataset = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                         .shuffle(2000)
                         .batch(batch_size, drop_remainder=True))


def run():
  """Run the model training and return evaluation output."""
  resolver = tf.contrib.cluster_resolver.TPUClusterResolver('grpc://' + os.environ['COLAB_TPU_ADDR'])
  tf.contrib.distribute.initialize_tpu_system(resolver)
  strategy = tf.contrib.distribute.TPUStrategy(resolver)

  model_cls = MODELS[FLAGS.model]
  data = Cifar10Dataset(FLAGS.batch_size)

  with strategy.scope():
    model = model_cls(weights=None, input_shape=data.input_shape,
                      classes=data.num_classes)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(loss="categorical_crossentropy",
                  optimizer=optimizer,
                  metrics=["accuracy"])

    history = model.fit(
        data.train_dataset,
        epochs=FLAGS.epochs,
        steps_per_epoch=data.num_train_images // FLAGS.batch_size,
        validation_data=data.test_dataset,
        validation_steps=data.num_test_images // FLAGS.batch_size,
        validation_freq=50)

    return history.history

def run_gpu():
  model_cls = MODELS[FLAGS.model]
  data = Cifar10Dataset(FLAGS.batch_size)

  model = model_cls(weights=None, input_shape=data.input_shape,
                    classes=data.num_classes)

  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  model.compile(loss="categorical_crossentropy",
                optimizer=optimizer,
                metrics=["accuracy"])

  history = model.fit(
      data.train_dataset,
      epochs=FLAGS.epochs,
      steps_per_epoch=data.num_train_images // FLAGS.batch_size,
      validation_data=data.test_dataset,
      validation_steps=data.num_test_images // FLAGS.batch_size,
      validation_freq=50)

  return history.history

def main(argv):
  del argv
  
  if FLAGS.mode == 'GPU':
    run_gpu()
  elif FLAGS.mode == 'TPU':
    run():
  else:
    print('mode error!')


if __name__ == "__main__":
  tf.app.run(main)