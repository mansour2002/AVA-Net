"""
AVA_Net: A U-Net-like architecture for image segmentation, incorporating
dilated convolutions for multi-scale feature extraction.
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, concatenate, Input
from tensorflow.keras.models import Model
from typing import Tuple, List, Union

# Import constants from the shared configuration file
from config import CONV_KERNEL_SIZE, OUTPUT_CHANNELS

# --- Building Blocks ---

def conv2d_block(input_tensor: tf.Tensor, n_conv: int, n_filters: int, kernel_size: int = CONV_KERNEL_SIZE) -> tf.Tensor:
    """
    Applies a series of 2D convolutional layers with ReLU activation.

    Args:
        input_tensor (tf.Tensor): The input tensor to the block.
        n_conv (int): The number of convolutional layers in this block.
        n_filters (int): The number of filters for each convolutional layer.
        kernel_size (int): The size of the convolutional kernel (e.g., 3 for 3x3).

    Returns:
        tf.Tensor: The output tensor after applying the convolutional block.
    """
    x = input_tensor
    for _ in range(n_conv):
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   padding='same', activation='relu')(x)
    return x

def conv2d_block_dilated(input_tensor: tf.Tensor, n_filters: int, kernel_size: int = CONV_KERNEL_SIZE) -> tf.Tensor:
    """
    Applies parallel dilated convolutional layers at different rates (1, 2, 3, 4)
    and concatenates their outputs along the channel axis.
    This aims to capture multi-scale context within the receptive field.

    Args:
        input_tensor (tf.Tensor): The input tensor to the block.
        n_filters (int): The total number of filters for the concatenated output.
                         Each dilated path will use n_filters / 4 filters.
        kernel_size (int): The size of the convolutional kernel.

    Returns:
        tf.Tensor: The output tensor after applying dilated convolutions and concatenation.
    """
    # Ensure n_filters is divisible by 4 to allow equal splitting among dilated branches
    if n_filters % 4 != 0:
        raise ValueError(f"n_filters ({n_filters}) must be divisible by 4 for conv2d_block_dilated.")
    
    filters_per_branch: int = n_filters // 4

    # Apply dilated convolutions with different rates
    x1 = Conv2D(filters=filters_per_branch, kernel_size=(kernel_size, kernel_size),
                padding='same', dilation_rate=1, activation='relu')(input_tensor)
    x2 = Conv2D(filters=filters_per_branch, kernel_size=(kernel_size, kernel_size),
                padding='same', dilation_rate=2, activation='relu')(input_tensor)
    x3 = Conv2D(filters=filters_per_branch, kernel_size=(kernel_size, kernel_size),
                padding='same', dilation_rate=3, activation='relu')(input_tensor)
    x4 = Conv2D(filters=filters_per_branch, kernel_size=(kernel_size, kernel_size),
                padding='same', dilation_rate=4, activation='relu')(input_tensor)
    
    # Concatenate the outputs from all dilated branches
    x = concatenate([x1, x2, x3, x4])
    return x

def CBR_block(input_tensor: tf.Tensor, n_conv: int, n_filters: int, kernel_size: int = CONV_KERNEL_SIZE, dilation_rate: int = 1) -> tf.Tensor:
    """
    Applies a series of Convolution, Batch Normalization, and ReLU activation layers.
    This sequence is common in many deep learning architectures.

    Args:
        input_tensor (tf.Tensor): The input tensor to the block.
        n_conv (int): The number of CBR sequences in this block.
        n_filters (int): The number of filters for each convolutional layer.
        kernel_size (int): The size of the convolutional kernel.
        dilation_rate (int): The dilation rate for the convolutional layers.

    Returns:
        tf.Tensor: The output tensor after applying the CBR block.
    """
    x = input_tensor
    for _ in range(n_conv):
        # use_bias=False when followed by BatchNormalization to avoid redundancy
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),
                   padding='same', dilation_rate=dilation_rate, use_bias=False)(x)
        x = BatchNormalization()(x)  # Normalizes activations for stable training
        x = Activation('relu')(x)    # Applies ReLU activation
    return x

# --- Encoder Section ---

def encoder_block_dilated(inputs: tf.Tensor, n_conv: int = 2, n_filters: int = 64, pool_size: Tuple[int, int] = (2, 2)) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    An encoder block that first applies a standard convolution block, then a dilated
    convolution block, followed by max pooling.

    Args:
        inputs (tf.Tensor): The input tensor for this encoder block.
        n_conv (int): Number of standard convolutional layers in the initial `conv2d_block`.
                      Note: The `conv2d_block_dilated` implicitly has its own 'n_conv' structure.
        n_filters (int): Number of filters used in the convolutional layers within this block.
        pool_size (Tuple[int, int]): Size of the pooling window for downsampling.

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: A tuple containing:
            - The feature map before pooling (f), which serves as a skip connection.
            - The pooled output (p), which is passed to the next encoder block.
    """
    # Apply standard convolutions for initial feature extraction
    f = conv2d_block(inputs, n_conv=n_conv - 1, n_filters=n_filters)
    # Apply dilated convolutions for multi-scale context
    f = conv2d_block_dilated(f, n_filters=n_filters)
    p = MaxPooling2D(pool_size=pool_size)(f)  # Downsample using max pooling
    return f, p

def encoder(inputs: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
    """
    Defines the encoder path of the network, consisting of a series of
    dilated encoder blocks with increasing filter sizes.

    Args:
        inputs (tf.Tensor): The input tensor to the encoder (e.g., image input).

    Returns:
        Tuple[tf.Tensor, List[tf.Tensor]]: A tuple containing:
            - The output of the last encoder block after pooling (p5),
              which feeds into the bottleneck.
            - A list of feature maps from each encoder block before pooling
              (f1, f2, f3, f4, f5), used as skip connections in the decoder.
    """
    # Each call to encoder_block_dilated generates feature maps (fX) and pooled outputs (pX)
    f1, p1 = encoder_block_dilated(inputs, n_conv=3, n_filters=64)
    f2, p2 = encoder_block_dilated(p1, n_conv=3, n_filters=128)
    f3, p3 = encoder_block_dilated(p2, n_conv=3, n_filters=256)
    f4, p4 = encoder_block_dilated(p3, n_conv=3, n_filters=512)
    f5, p5 = encoder_block_dilated(p4, n_conv=3, n_filters=512)
    return p5, [f1, f2, f3, f4, f5]

# --- Bottleneck Section ---

def bottleneck(inputs: tf.Tensor) -> tf.Tensor:
    """
    Defines the bottleneck block of the network, connecting the encoder to the decoder.
    It typically processes the most abstract features.

    Args:
        inputs (tf.Tensor): The input tensor to the bottleneck (output of the encoder).

    Returns:
        tf.Tensor: The output tensor of the bottleneck block.
    """
    bottle_neck = CBR_block(inputs, n_conv=2, n_filters=512)
    return bottle_neck

# --- Decoder Section ---

def decoder_block(inputs: tf.Tensor, conv_output: tf.Tensor, n_filters: int, kernel_size: int = CONV_KERNEL_SIZE) -> tf.Tensor:
    """
    A standard decoder block for upsampling and feature refinement.
    It performs upsampling, concatenates with corresponding encoder features (skip connection),
    and then applies CBR convolutions.

    Args:
        inputs (tf.Tensor): The input tensor from the previous decoder block (or bottleneck).
        conv_output (tf.Tensor): The skip connection feature map from the corresponding encoder block.
        n_filters (int): The number of filters for the CBR block.
        kernel_size (int): The size of the convolutional kernel.

    Returns:
        tf.Tensor: The output tensor of the decoder block.
    """
    # Upsample the input to match the spatial dimensions of the skip connection
    u = UpSampling2D(size=(2, 2))(inputs)
    # Concatenate the upsampled features with the encoder's skip connection
    c = concatenate([u, conv_output], axis=-1) # Explicitly define axis for clarity
    # Apply CBR block for feature learning and refinement
    c = CBR_block(input_tensor=c, n_conv=2, n_filters=n_filters, kernel_size=kernel_size)
    return c

def decoder(inputs: tf.Tensor, convs: List[tf.Tensor], output_channels: int) -> tf.Tensor:
    """
    Defines the decoder path of the network, performing upsampling and combining features
    with skip connections from the encoder to reconstruct the segmentation mask.

    Args:
        inputs (tf.Tensor): The input tensor to the decoder (output of the bottleneck).
        convs (List[tf.Tensor]): A list of feature maps from the encoder (f1-f5)
                                 for skip connections. Expected order: [f1, f2, f3, f4, f5].
        output_channels (int): The number of channels for the final output segmentation mask.

    Returns:
        tf.Tensor: The final output segmentation mask tensor with `output_channels`.
    """
    # Unpack skip connections from the list for easier access
    f1, f2, f3, f4, f5 = convs 

    # Decoder blocks, from coarsest to finest features
    c5 = decoder_block(inputs, f5, n_filters=256)
    c4 = decoder_block(c5, f4, n_filters=128)
    c3 = decoder_block(c4, f3, n_filters=64)
    c2 = decoder_block(c3, f2, n_filters=32)
    c1 = decoder_block(c2, f1, n_filters=16) # This block also uses a skip connection

    # Final convolutional layer to produce the output segmentation mask
    # Sigmoid activation is suitable for binary segmentation or multi-label segmentation.
    # For multi-class (mutually exclusive classes), use 'softmax'.
    outputs = Conv2D(filters=output_channels, kernel_size=CONV_KERNEL_SIZE,
                     padding='same', activation='sigmoid')(c1)
    return outputs

# --- Main Model Definition ---

def AVA_Net(input_shape: Tuple[Union[int, None], Union[int, None], int] = (None, None, 3)) -> Model:
    """
    Constructs the AVA_Net deep learning model for image segmentation.
    This model follows a U-Net-like encoder-decoder architecture with dilated convolutions
    in the encoder and skip connections.

    Args:
        input_shape (Tuple[Union[int, None], Union[int, None], int]): The expected shape of the input images.
                                                                     (None, None, 3) allows for variable image sizes
                                                                     with 3 color channels (RGB).

    Returns:
        tf.keras.Model: The compiled Keras Model instance of AVA_Net.
    """
    # Define the input layer with the specified shape
    inputs = Input(shape=input_shape, name="input_image")

    # Build the encoder path
    encoder_output_pooled, convs_for_skips = encoder(inputs)
    
    # Build the bottleneck
    bottle_neck_output = bottleneck(encoder_output_pooled)
    
    # Build the decoder path
    outputs = decoder(bottle_neck_output, convs_for_skips, output_channels=OUTPUT_CHANNELS)
    
    # Create the Keras Model instance
    model = Model(inputs=inputs, outputs=outputs, name="AVA_Net_Model")
    return model

