# Cifar Keras #

This directory contains an example using the Keras layers API inside an
Estimator/TPUEstimator. If you have a complete Keras model already built,
consider the new experimental Cloud TPU-Keras integration available since TF
1.9. For examples, see [`models/experimental/keras`](https://github.com/tensorflow/tpu/tree/master/models/experimental/keras)

## How to Run
```
python cifar_keras.py --model_dir gs://{checkpoint_gcs_dir}
```
