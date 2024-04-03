import tensorflow as tf  # TensorFlow for deep learning

convkernel = 3  # Convolution kernel size
OUTPUT_CHANNELS = 1  # Number of output channels

# Convolution block
def conv2d_block(input_tensor, n_conv, n_filters, kernel_size=convkernel, dilation_rate=1):
  x = input_tensor
  for i in range(n_conv):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate, activation='relu')(x)
  return x

  
# Dilated convolution block
def conv2d_block_dilated(input_tensor, n_conv, n_filters, kernel_size=convkernel, dilation_rate=1):
  x = input_tensor
  # Dilation rates vary to capture different scales
  x1 = tf.keras.layers.Conv2D(filters=n_filters/4, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=1, activation='relu')(x)
  x2 = tf.keras.layers.Conv2D(filters=n_filters/4, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=2, activation='relu')(x)
  x3 = tf.keras.layers.Conv2D(filters=n_filters/4, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=3, activation='relu')(x)
  x4 = tf.keras.layers.Conv2D(filters=n_filters/4, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=4, activation='relu')(x)
  x = tf.keras.layers.concatenate([x1, x2, x3, x4])  # Combine dilated convolutions
  return x


# Convolution, BatchNormalization, and ReLU block
def CBR_block(input_tensor, n_conv, n_filters, kernel_size=convkernel, dilation_rate=1):
  x = input_tensor
  for i in range(n_conv):
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same', dilation_rate=dilation_rate, use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization()(x)  # Normalize
    x = tf.keras.layers.Activation('relu')(x)  # Activation
  return x


# Encoder block with dilated convolutions
def encoder_block_dilated(inputs, n_conv=2, n_filters=64, pool_size=(2,2)):
  f = conv2d_block(inputs, n_conv-1, n_filters=n_filters)
  f = conv2d_block_dilated(f, 1, n_filters=n_filters)
  p = tf.keras.layers.MaxPooling2D(pool_size=pool_size)(f)  # Pooling
  return f, p


# Encoder architecture
def encoder(inputs):
  f1, p1 = encoder_block_dilated(inputs, n_conv=3, n_filters=64)
  f2, p2 = encoder_block_dilated(p1, n_conv=3, n_filters=128)
  f3, p3 = encoder_block_dilated(p2, n_conv=3, n_filters=256)
  f4, p4 = encoder_block_dilated(p3, n_conv=3, n_filters=512)
  f5, p5 = encoder_block_dilated(p4, n_conv=3, n_filters=512)
  return p5, (f1, f2, f3, f4, f5)


# Bottleneck of the network
def bottleneck(inputs):
  bottle_neck = CBR_block(inputs, n_conv=2, n_filters=512)
  return bottle_neck


# Decoder block
def decoder_block(inputs, conv_output, n_filters=64, kernel_size=convkernel):
  u = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)  # Upsampling
  c = tf.keras.layers.concatenate([u, conv_output])  # Concatenate
  c = CBR_block(input_tensor=c, n_conv=2, n_filters=n_filters, kernel_size=convkernel)
  return c


# Last decoder block
def last_decoder_block(inputs, conv_output, n_filters=64, kernel_size=convkernel):
  u = tf.keras.layers.UpSampling2D(size=(2, 2))(inputs)
  c = CBR_block(input_tensor=u, n_conv=2, n_filters=n_filters, kernel_size=convkernel)
  return c


# Decoder architecture
def decoder(inputs, convs, output_channels):
  f1, f2, f3, f4, f5 = convs
  c5 = decoder_block(inputs, f5, n_filters=256)
  c4 = decoder_block(c5, f4, n_filters=128)
  c3 = decoder_block(c4, f3, n_filters=64)
  c2 = decoder_block(c3, f2, n_filters=32)
  c1 = decoder_block(c2, f1, n_filters=16)
  outputs = tf.keras.layers.Conv2D(filters=output_channels, kernel_size=convkernel, padding='same', activation='sigmoid')(c1)  # Output layer
  return outputs


# AVA_Net model
def AVA_Net():
  inputs = tf.keras.layers.Input(shape=(None, None, 3))  # Input layer
  encoder_output, convs = encoder(inputs)
  bottle_neck = bottleneck(encoder_output)
  outputs = decoder(bottle_neck, convs, output_channels=OUTPUT_CHANNELS)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)  # Model definition
  return model
