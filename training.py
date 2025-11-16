"""
Main script for training the AVA_Net model for image segmentation.
This script handles data loading, augmentation, model compilation, and training.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple

# Keras functions for data augmentation and model building
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

import segmentation_models as sm  # For segmentation metrics and models
# skimage functions for image processing
from skimage import io, transform

# Import custom model and loss function
from AVA_Net import AVA_Net
from loss import iou_loss_MC

# Import configuration settings
from config import (
    PARENT_DIR, TRAIN_CSV_FILENAME, TRAIN_INPUT_DIR_NAME, TRAIN_OUTPUT_DIR_NAME,
    IMAGE_SHAPE, RANDOM_SEED, MAX_EPOCHS, LEARNING_RATE, BATCH_SIZE,
    DATA_AUGMENTATION_ARGS, OUTPUT_CHANNELS
)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Data Loading Functions ---

def load_RGB_image_data(dataframe: pd.DataFrame, base_path: str, target_shape: Tuple[int, int], filename_column: str) -> np.ndarray:
    """
    Loads RGB images from specified paths, resizes them, and returns them as a NumPy array.

    Args:
        dataframe (pd.DataFrame): DataFrame containing image file names.
        base_path (str): Base directory path where images are located.
        target_shape (Tuple[int, int]): Desired output shape for the images (height, width).
        filename_column (str): Column name in the DataFrame containing image filenames.

    Returns:
        np.ndarray: An array of loaded and resized RGB images with shape (num_images, height, width, 3).
    """
    images = []
    logging.info(f"Loading RGB images from: {base_path}")
    for idx, row in dataframe.iterrows():
        image_path = os.path.join(base_path, row[filename_column])
        try:
            temp_img = io.imread(image_path)
            # Ensure image is RGB (3 channels). If grayscale, expand dimensions.
            if len(temp_img.shape) == 2:
                temp_img = np.stack([temp_img]*3, axis=-1)
            elif temp_img.shape[2] == 4: # Handle RGBA images
                temp_img = temp_img[:, :, :3] # Discard alpha channel

            temp_img = transform.resize(temp_img, [target_shape[0], target_shape[1], 3], anti_aliasing=True)
            images.append(temp_img)
            if idx % 100 == 0: # Log progress
                logging.debug(f"Loaded {idx+1}/{len(dataframe)} images.")
        except FileNotFoundError:
            logging.error(f"Error: Image file not found: {image_path}. Skipping this image.")
        except Exception as e:
            logging.error(f"Error loading or processing image {image_path}: {e}. Skipping this image.")
    
    images_array = np.array(images, dtype=np.float32) # Ensure float32 for consistency
    logging.info(f"Finished loading images. Total loaded: {len(images)}. Resultant shape: {images_array.shape}")
    return images_array

def load_grayscale_mask_data(dataframe: pd.DataFrame, base_path: str, target_shape: Tuple[int, int], filename_column: str) -> np.ndarray:
    """
    Loads grayscale mask images from specified paths, resizes them, and returns them as a NumPy array.
    Masks are expanded to have a channel dimension.

    Args:
        dataframe (pd.DataFrame): DataFrame containing mask file names.
        base_path (str): Base directory path where masks are located.
        target_shape (Tuple[int, int]): Desired output shape for the masks (height, width).
        filename_column (str): Column name in the DataFrame containing mask filenames.

    Returns:
        np.ndarray: An array of loaded and resized grayscale masks with shape (num_masks, height, width, 1).
    """
    masks = []
    logging.info(f"Loading grayscale masks from: {base_path}")
    for idx, row in dataframe.iterrows():
        mask_path = os.path.join(base_path, row[filename_column])
        try:
            temp_mask = io.imread(mask_path)
            # If mask has 3 channels, take the first one (assuming it's grayscale repeated)
            if len(temp_mask.shape) == 3 and temp_mask.shape[2] >= 1:
                temp_mask = temp_mask[:, :, 0]
            
            temp_mask = transform.resize(temp_mask, [target_shape[0], target_shape[1]], anti_aliasing=True)
            masks.append(temp_mask)
            if idx % 100 == 0: # Log progress
                logging.debug(f"Loaded {idx+1}/{len(dataframe)} masks.")
        except FileNotFoundError:
            logging.error(f"Error: Mask file not found: {mask_path}. Skipping this mask.")
        except Exception as e:
            logging.error(f"Error loading or processing mask {mask_path}: {e}. Skipping this mask.")

    masks_array = np.array(masks, dtype=np.float32) # Ensure float32 for consistency
    # Expand dimensions to (num_masks, height, width, 1) for Keras compatibility
    masks_array = np.expand_dims(masks_array, axis=-1)
    logging.info(f"Finished loading masks. Total loaded: {len(masks)}. Resultant shape: {masks_array.shape}")
    return masks_array

# --- Main Training Logic ---

def main():
    """
    Main function to orchestrate the training process for the AVA_Net model.
    """
    logging.info("Starting AVA_Net training script.")

    # --- Setup Directories and File Paths ---
    train_csv_full_path: str = os.path.join(PARENT_DIR, TRAIN_CSV_FILENAME)
    image_dir_full_path: str = os.path.join(PARENT_DIR, TRAIN_INPUT_DIR_NAME)
    mask_dir_full_path: str = os.path.join(PARENT_DIR, TRAIN_OUTPUT_DIR_NAME)

    # --- Load Training Data ---
    try:
        df_train = pd.read_csv(train_csv_full_path)
        logging.info(f"Successfully loaded training CSV from: {train_csv_full_path}")
    except FileNotFoundError:
        logging.critical(f"Error: Training CSV file not found at {train_csv_full_path}. Exiting.")
        return
    except Exception as e:
        logging.critical(f"Error reading training CSV: {e}. Exiting.")
        return

    num_imgs: int = len(df_train)
    if num_imgs == 0:
        logging.warning("Training DataFrame is empty. No images to train on. Exiting.")
        return

    X_train = load_RGB_image_data(df_train, image_dir_full_path, IMAGE_SHAPE, 'Input')
    y_train = load_grayscale_mask_data(df_train, mask_dir_full_path, IMAGE_SHAPE, 'AV')

    if X_train.size == 0 or y_train.size == 0:
        logging.critical("No training images or masks were loaded. Exiting.")
        return
    if X_train.shape[0] != y_train.shape[0]:
        logging.critical(f"Mismatch in number of images ({X_train.shape[0]}) and masks ({y_train.shape[0]}). Exiting.")
        return

    # --- Data Augmentation ---
    # Keras ImageDataGenerator for on-the-fly data augmentation
    image_datagen = ImageDataGenerator(**DATA_AUGMENTATION_ARGS)
    # Masks should be augmented identically to images for pixel-wise tasks
    mask_datagen = ImageDataGenerator(**DATA_AUGMENTATION_ARGS)

    logging.info("Fitting ImageDataGenerators to training data for standardization.")
    image_datagen.fit(X_train)
    mask_datagen.fit(y_train)

    logging.info("Setting up data generators for training.")
    image_generator = image_datagen.flow_from_dataframe(
        df_train,
        directory=image_dir_full_path,
        x_col='Input',
        target_size=IMAGE_SHAPE,
        color_mode='rgb',
        batch_size=BATCH_SIZE, # Use configurable BATCH_SIZE
        class_mode=None, # No y for images
        seed=RANDOM_SEED,
        shuffle=True # Shuffle training data
    )
    mask_generator = mask_datagen.flow_from_dataframe(
        df_train,
        directory=mask_dir_full_path,
        x_col='AV',
        target_size=IMAGE_SHAPE,
        color_mode='grayscale', # Masks are grayscale
        batch_size=BATCH_SIZE, # Same batch size as image generator
        class_mode=None, # No y for masks
        seed=RANDOM_SEED,
        shuffle=True # Shuffle masks identically to images
    )


    train_generator = zip(image_generator, mask_generator)

    # --- Model Definition ---
    logging.info("Initializing AVA_Net model.")
    # The Input layer is defined inside AVA_Net, so no need to pass 'inp' explicitly here.
    model = AVA_Net(input_shape=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3)) # Pass the expected input shape

    logging.info("Model summary:")
    model.summary(print_fn=logging.info) # Print summary using logging

    # --- Model Compilation ---
    logging.info("Compiling model with Adam optimizer, IoU loss, and segmentation metrics.")
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=iou_loss_MC, # Custom IoU loss
        metrics=[
            sm.metrics.IOUScore(class_weights=1, threshold=0.5, name='iou_score'), # IoU score metric
            sm.metrics.FScore(class_weights=1, threshold=0.5, name='f1_score'),   # F1-score metric
            'acc' # Pixel accuracy
        ]
    )

    # --- Model Training ---
    logging.info(f"Starting model training for {MAX_EPOCHS} epochs with batch size {BATCH_SIZE}.")
    # Calculate steps per epoch. Ensure it's an integer and handles remainder.
    steps_per_epoch_val = max(1, (num_imgs + BATCH_SIZE - 1) // BATCH_SIZE) # At least 1 step
    logging.info(f"Calculated steps per epoch: {steps_per_epoch_val}")

    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch_val,
        epochs=MAX_EPOCHS,
        verbose=1 # Show progress bar
    )

    logging.info("Training completed.")

    # Save the trained model
    model_save_path = os.path.join(PARENT_DIR, 'trained_ava_net.h5')
    model.save(model_save_path)
    logging.info(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    main()

