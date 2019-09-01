# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for bounding box processing."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

EPSILON = 1e-8
BBOX_XFORM_CLIP = np.log(1000. / 16.)


def jitter_boxes(boxes, noise_scale=0.025):
  """Jitter the box coordinates by some noise distribution.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    noise_scale: a python float which specifies the magnitude of noise. The
      rule of thumb is to set this between (0, 0.1]. The default value is found
      to mimic the noisy detections best empirically.

  Returns:
    jittered_boxes: a tensor whose shape is the same as `boxes` representing
      the jittered boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[1]))

  with tf.name_scope('jitter_boxes'):
    bbox_jitters = tf.random_normal([4], stddev=noise_scale)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    width = xmax - xmin
    height = ymax - ymin
    new_center_x = (xmin + xmax) / 2.0 + bbox_jitters[0] * width
    new_center_y = (ymin + ymax) / 2.0 + bbox_jitters[1] * height
    new_width = width * tf.exp(bbox_jitters[2])
    new_height = height * tf.exp(bbox_jitters[3])
    jittered_boxes = tf.concat([
        new_center_y - new_height * 0.5,
        new_center_x - new_width * 0.5,
        new_center_y + new_height * 0.5,
        new_center_x + new_width * 0.5], axis=-1)

    return jittered_boxes


def normalize_boxes(boxes, image_shape):
  """Converts boxes to the normalized coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    normalized_boxes: a tensor whose shape is the same as `boxes` representing
      the normalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[1]))

  with tf.name_scope('normalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height = image_shape[..., 0:1]
      width = image_shape[..., 1:2]

    ymin = boxes[..., 0:1] / height
    xmin = boxes[..., 1:2] / width
    ymax = boxes[..., 2:3] / height
    xmax = boxes[..., 3:4] / width

    normalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return normalized_boxes


def denormalize_boxes(boxes, image_shape):
  """Converts boxes normalized by [height, width] to pixel coordinates.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    denormalized_boxes: a tensor whose shape is the same as `boxes` representing
      the denormalized boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  with tf.name_scope('denormalize_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height, width = tf.split(image_shape, 2, axis=-1)

    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=-1)
    ymin = ymin * height
    xmin = xmin * width
    ymax = ymax * height
    xmax = xmax * width

    denormalized_boxes = tf.concat([ymin, xmin, ymax, xmax], axis=-1)
    return denormalized_boxes


def clip_boxes(boxes, image_shape):
  """Clips boxes to image boundaries.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].

  Returns:
    clipped_boxes: a tensor whose shape is the same as `boxes` representing the
      clipped boxes.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[1]))

  with tf.name_scope('crop_boxes'):
    if isinstance(image_shape, list) or isinstance(image_shape, tuple):
      height, width = image_shape
    else:
      image_shape = tf.cast(image_shape, dtype=boxes.dtype)
      height = image_shape[..., 0:1]
      width = image_shape[..., 1:2]

    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]

    clipped_ymin = tf.maximum(tf.minimum(ymin, height - 1.0), 0.0)
    clipped_ymax = tf.maximum(tf.minimum(ymax, height - 1.0), 0.0)
    clipped_xmin = tf.maximum(tf.minimum(xmin, width - 1.0), 0.0)
    clipped_xmax = tf.maximum(tf.minimum(xmax, width - 1.0), 0.0)

    clipped_boxes = tf.concat(
        [clipped_ymin, clipped_xmin, clipped_ymax, clipped_xmax],
        axis=-1)
    return clipped_boxes


def compute_outer_boxes(boxes, image_shape, scale=1.0):
  """Compute outer box encloses an object with a margin.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    image_shape: a list of two integers, a two-element vector or a tensor such
      that all but the last dimensions are `broadcastable` to `boxes`. The last
      dimension is 2, which represents [height, width].
    scale: a float number specifying the scale of output outer boxes to input
      `boxes`.

  Returns:
    outer_boxes: a tensor whose shape is the same as `boxes` representing the
      outer boxes.
  """
  if scale < 1.0:
    raise ValueError(
        'scale is {}, but outer box scale must be greater than 1.0.'.format(
            scale))
  centers_y = (boxes[..., 0] + boxes[..., 2]) / 2.0
  centers_x = (boxes[..., 1] + boxes[..., 3]) / 2.0
  box_height = (boxes[..., 2] - boxes[..., 0]) * scale
  box_width = (boxes[..., 3] - boxes[..., 1]) * scale
  outer_boxes = tf.stack([centers_y - box_height / 2.0,
                          centers_x - box_width / 2.0,
                          centers_y + box_height / 2.0,
                          centers_x + box_width / 2.0], axis=1)
  outer_boxes = clip_boxes(outer_boxes, image_shape)
  return outer_boxes


def encode_boxes(boxes, anchors, weights=None):
  """Encode boxes to targets.

  Args:
    boxes: a tensor whose last dimension is 4 representing the coordinates
      of boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      encoded box targets.

  Raises:
    ValueError: If the last dimension of boxes is not 4.
  """
  if boxes.shape[-1] != 4:
    raise ValueError(
        'boxes.shape[1] is {:d}, but must be 4.'.format(boxes.shape[1]))

  with tf.name_scope('encode_boxes'):
    boxes = tf.cast(boxes, dtype=anchors.dtype)
    ymin = boxes[..., 0:1]
    xmin = boxes[..., 1:2]
    ymax = boxes[..., 2:3]
    xmax = boxes[..., 3:4]
    box_h = ymax - ymin + 1.0
    box_w = xmax - xmin + 1.0
    box_yc = ymin + 0.5 * box_h
    box_xc = xmin + 0.5 * box_w

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    encoded_dy = (box_yc - anchor_yc) / anchor_h
    encoded_dx = (box_xc - anchor_xc) / anchor_w
    encoded_dh = tf.log(box_h / anchor_h)
    encoded_dw = tf.log(box_w / anchor_w)
    if weights:
      encoded_dy *= weights[0]
      encoded_dx *= weights[1]
      encoded_dh *= weights[2]
      encoded_dw *= weights[3]

    encoded_boxes = tf.concat(
        [encoded_dy, encoded_dx, encoded_dh, encoded_dw],
        axis=-1)
    return encoded_boxes


def decode_boxes(encoded_boxes, anchors, weights=None):
  """Decode boxes.

  Args:
    encoded_boxes: a tensor whose last dimension is 4 representing the
      coordinates of encoded boxes in ymin, xmin, ymax, xmax order.
    anchors: a tensor whose shape is the same as, or `broadcastable` to `boxes`,
      representing the coordinates of anchors in ymin, xmin, ymax, xmax order.
    weights: None or a list of four float numbers used to scale coordinates.

  Returns:
    encoded_boxes: a tensor whose shape is the same as `boxes` representing the
      decoded box targets.
  """
  with tf.name_scope('decode_boxes'):
    encoded_boxes = tf.cast(encoded_boxes, dtype=anchors.dtype)
    dy = encoded_boxes[..., 0:1]
    dx = encoded_boxes[..., 1:2]
    dh = encoded_boxes[..., 2:3]
    dw = encoded_boxes[..., 3:4]
    if weights:
      dy /= weights[0]
      dx /= weights[1]
      dh /= weights[2]
      dw /= weights[3]
    dh = tf.minimum(dh, BBOX_XFORM_CLIP)
    dw = tf.minimum(dw, BBOX_XFORM_CLIP)

    anchor_ymin = anchors[..., 0:1]
    anchor_xmin = anchors[..., 1:2]
    anchor_ymax = anchors[..., 2:3]
    anchor_xmax = anchors[..., 3:4]
    anchor_h = anchor_ymax - anchor_ymin + 1.0
    anchor_w = anchor_xmax - anchor_xmin + 1.0
    anchor_yc = anchor_ymin + 0.5 * anchor_h
    anchor_xc = anchor_xmin + 0.5 * anchor_w

    decoded_boxes_yc = dy * anchor_h + anchor_yc
    decoded_boxes_xc = dx * anchor_w + anchor_xc
    decoded_boxes_h = tf.exp(dh) * anchor_h
    decoded_boxes_w = tf.exp(dw) * anchor_w

    decoded_boxes_ymin = decoded_boxes_yc - 0.5 * decoded_boxes_h
    decoded_boxes_xmin = decoded_boxes_xc - 0.5 * decoded_boxes_w
    decoded_boxes_ymax = decoded_boxes_ymin + decoded_boxes_h - 1.0
    decoded_boxes_xmax = decoded_boxes_xmin + decoded_boxes_w - 1.0

    decoded_boxes = tf.concat(
        [decoded_boxes_ymin, decoded_boxes_xmin,
         decoded_boxes_ymax, decoded_boxes_xmax],
        axis=-1)
    return decoded_boxes
