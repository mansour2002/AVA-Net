import tensorflow as tf  
from tensorflow.keras import backend as K  

def iou_coef_MC_for_loss(y_true, y_pred):
    smooth = 1  # Smoothing factor to prevent division by zero
    threshold = 0.5  # Threshold value for binary classification
    y_true = tf.cast(y_true, tf.float32)  # Convert true labels to float32
    y_pred = tf.cast(y_pred, tf.float32)  # Convert predicted labels to float32
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1,2])  # Calculate intersection
    union = K.sum(y_true,[1,2])+K.sum(y_pred,[1,2])-intersection  # Calculate union
    iou = K.mean((intersection + smooth) / (union + smooth))  # Compute IoU score
    return iou  

def iou_loss_MC(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Convert true labels to float32
    y_pred = tf.cast(y_pred, tf.float32)  # Convert predicted labels to float32
    
    return 1-iou_coef_MC_for_loss(y_true, y_pred) 
