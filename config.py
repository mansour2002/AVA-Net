"""
Configuration settings for the AVA-Net training and model.
This file centralizes parameters for easy modification and management.
"""

# --- Data and Image Parameters ---
IMAGE_HEIGHT: int = 320
IMAGE_WIDTH: int = 320
IMAGE_SHAPE: tuple = (IMAGE_HEIGHT, IMAGE_WIDTH) # (Height, Width) for image resizing

# --- Directory Paths ---
# Define parent_dir or leave it empty if paths are relative to the script location
# For absolute paths, update parent_dir = '/path/to/your/project'
PARENT_DIR: str = '' 
TRAIN_CSV_FILENAME: str = 'train_data.csv'
TRAIN_INPUT_DIR_NAME: str = 'Dataset/Train Input'
TRAIN_OUTPUT_DIR_NAME: str = 'Dataset/Train Output' # Assuming this is 'AV' masks

# --- Training Parameters ---
RANDOM_SEED: int = 7
MAX_EPOCHS: int = 5000
LEARNING_RATE: float = 1e-4
BATCH_SIZE: int = 16 # A typical batch size. Original code used full dataset size. Adjust as needed.

# --- Model Architecture Parameters ---
CONV_KERNEL_SIZE: int = 3  # Default kernel size for convolutional layers
OUTPUT_CHANNELS: int = 1   # Number of output channels for the segmentation mask (e.g., 1 for binary segmentation)

# --- Data Augmentation Parameters ---
# Parameters for Keras ImageDataGenerator
DATA_AUGMENTATION_ARGS = dict(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    rotation_range=70,       # Random rotations up to 70 degrees
    shear_range=0.4,         # Shear intensity
    zoom_range=0.5,          # Random zoom range
    brightness_range=[0.1, 1.9], # Random brightness adjustment
    width_shift_range=0.3,   # Horizontal shift
    height_shift_range=0.3,  # Vertical shift
    fill_mode="reflect",     # Strategy for filling in new pixels created by transformations
    horizontal_flip=True,    # Random horizontal flips
    vertical_flip=True       # Random vertical flips
)
