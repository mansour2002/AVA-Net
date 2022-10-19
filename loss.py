import tensorflow as tf
from tensorflow.keras import backend as K

def iou_coef_MC_for_loss(y_true, y_pred):
    smooth = 1
    threshold = 0.5
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth))
    return iou


def iou_loss_MC(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    return 1-iou_coef_MC_for_loss(y_true, y_pred)
