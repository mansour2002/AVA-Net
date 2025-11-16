"""
Custom loss functions and metrics for image segmentation using TensorFlow/Keras.
This module provides an Intersection-over-Union (IoU) coefficient and its corresponding loss.
"""

import tensorflow as tf
from tensorflow.keras import backend as K

def iou_coef_MC(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the Intersection-over-Union (IoU) coefficient (Jaccard index)
    for multi-class or binary segmentation. This function is designed to work
    with continuous probability outputs from the model (e.g., sigmoid activation).

    Args:
        y_true (tf.Tensor): The ground truth masks. Expected shape (batch, height, width, channels).
                            Should be float32.
        y_pred (tf.Tensor): The predicted masks (probabilities). Expected shape (batch, height, width, channels).
                            Should be float32.

    Returns:
        tf.Tensor: The mean IoU score averaged across the batch and channels.
    """
    smooth: float = 1e-5  # A small constant to prevent division by zero, also acts as smoothing.

    # Ensure tensors are float32 for consistent calculations
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate intersection along spatial dimensions (height and width)
    # K.abs(y_true * y_pred) is effectively y_true * y_pred for non-negative values.
    # Summing over spatial dimensions for each batch item and channel.
    intersection: tf.Tensor = K.sum(K.abs(y_true * y_pred), axis=[1, 2])

    # Calculate union along spatial dimensions for each batch item and channel.
    # Union = Sum(A) + Sum(B) - Intersection(A,B)
    union: tf.Tensor = K.sum(y_true, axis=[1, 2]) + K.sum(y_pred, axis=[1, 2]) - intersection

    # Compute IoU score for each batch item and channel, then take the mean over all
    # dimensions (batch and channels) to get a single scalar loss value.
    iou: tf.Tensor = K.mean((intersection + smooth) / (union + smooth))
    return iou

def iou_loss_MC(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the IoU (Jaccard) loss for multi-class or binary segmentation.
    This loss is defined as 1 - IoU coefficient. Lower values are better.

    Args:
        y_true (tf.Tensor): The ground truth masks. Expected shape (batch, height, width, channels).
        y_pred (tf.Tensor): The predicted masks (probabilities). Expected shape (batch, height, width, channels).

    Returns:
        tf.Tensor: The IoU loss, a scalar value.
    """
    # The casting to float32 is handled internally by iou_coef_MC,
    # so no need to repeat it here.
    return 1.0 - iou_coef_MC(y_true, y_pred)

